"""Streaming Live RAG Engine orchestrating controller, retrieval, fusion, and grounded streaming."""

import uuid
from typing import Any, Iterator

from avy.config import AVYConfig
from avy.context.session import SessionManager
from avy.core.models import ControllerState, PipelineTimings
from avy.core.timing import Stopwatch
from avy.providers.base import BaseProvider
from avy.providers.models import AgentRequest
from avy.retrieval.controller import RetrievalController
from avy.retrieval.decomposition import MultiIntentDecomposer
from avy.retrieval.embeddings import get_embedding_engine
from avy.retrieval.fusion import EvidenceFusion
from avy.retrieval.grounding import GroundingEngine
from avy.retrieval.multi_query import MultiQueryRetriever
from avy.retrieval.reranker import EvidenceReranker
from avy.retrieval.vectorstore import FAISSVectorStore
from avy.telemetry.events import TelemetryEventNames
from avy.telemetry.logger import TelemetryLogger, get_telemetry_logger


class StreamingLiveRAGEngine:
    """The central orchestrator executing the streaming live RAG pipeline."""

    def __init__(
        self,
        config: AVYConfig | None = None,
        provider: BaseProvider | None = None,
        vectorstore: FAISSVectorStore | None = None,
        session_manager: SessionManager | None = None,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.config = config or AVYConfig.load()
        if provider is None:
            from avy.providers.registry import get_provider

            self.provider = get_provider(self.config.provider, config=self.config)
        else:
            self.provider = provider

        self.session_manager = session_manager or SessionManager()
        self.telemetry = telemetry or get_telemetry_logger(self.config.telemetry_log)

        # Retrieval components
        self.controller = RetrievalController(config=self.config)
        self.decomposer = MultiIntentDecomposer(provider=self.provider)
        self.embeddings = get_embedding_engine(config=self.config)

        if vectorstore is None:
            # Attempt to load saved index if file exists
            try:
                self.vectorstore = FAISSVectorStore.load(
                    index_path=self.config.index_path,
                    metadata_path=self.config.metadata_path,
                )
            except Exception:
                self.vectorstore = FAISSVectorStore(dimension=self.embeddings.dimension)
        else:
            self.vectorstore = vectorstore

        self.retriever = MultiQueryRetriever(
            vectorstore=self.vectorstore,
            embeddings=self.embeddings,
            top_k=self.config.top_k,
            threshold=self.config.similarity_threshold,
        )
        self.fusion = EvidenceFusion(rrf_k=self.config.rrf_k)
        self.reranker = EvidenceReranker(
            enabled=self.config.reranker_enabled,
            model_name=self.config.reranker_model,
        )
        self.grounding = GroundingEngine()

    def process_transcript_stream(
        self,
        transcript_chunk: str,
        session_id: str | None = None,
        accumulated_transcript: str | None = None,
        is_final_chunk: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Process incoming live transcript chunk and stream pipeline events and tokens."""
        pipeline_sw = Stopwatch()
        sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        utterance_id = f"utt_{uuid.uuid4().hex[:8]}"
        session = self.session_manager.get_or_create(sid)

        acc_text = (accumulated_transcript or transcript_chunk).strip()

        # Telemetry: input.chunk
        self.telemetry.log(
            event=TelemetryEventNames.INPUT_CHUNK,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"chunk": transcript_chunk, "accumulated": acc_text, "is_final": is_final_chunk},
        )

        yield {
            "type": "transcript",
            "chunk": transcript_chunk,
            "accumulated": acc_text,
            "session_id": sid,
            "utterance_id": utterance_id,
        }

        # 1. Controller Evaluation
        decision = self.controller.evaluate(
            transcript_chunk=transcript_chunk,
            accumulated_transcript=acc_text,
            session_context=session,
        )

        # Handle Controller State: WAIT
        if decision.state == ControllerState.WAIT:
            self.telemetry.log(
                event=TelemetryEventNames.RETRIEVAL_WAIT,
                session_id=sid,
                utterance_id=utterance_id,
                metadata={"reason": decision.reason, "confidence": decision.confidence},
            )
            yield {
                "type": "controller_decision",
                "state": ControllerState.WAIT.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
            }
            return

        # Handle Controller State: NO_RETRIEVE
        if decision.state == ControllerState.NO_RETRIEVE:
            self.telemetry.log(
                event=TelemetryEventNames.RETRIEVAL_SKIPPED,
                session_id=sid,
                utterance_id=utterance_id,
                metadata={"reason": decision.reason},
            )
            yield {
                "type": "controller_decision",
                "state": ControllerState.NO_RETRIEVE.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
            }

            # Direct conversational stream
            self.telemetry.log(
                event=TelemetryEventNames.SYNTHESIS_STARTED,
                session_id=sid,
                utterance_id=utterance_id,
                metadata={"mode": "conversational_direct"},
            )

            req = AgentRequest(prompt=acc_text, temperature=self.config.temperature, stream=True)
            accumulated_resp = []
            first_token_seen = False
            ttft_ms = 0.0
            synth_sw = Stopwatch()

            for token in self.provider.stream(req):
                if not first_token_seen:
                    first_token_seen = True
                    ttft_ms = pipeline_sw.elapsed_ms
                    self.telemetry.log(
                        event=TelemetryEventNames.SYNTHESIS_FIRST_TOKEN,
                        session_id=sid,
                        utterance_id=utterance_id,
                        metadata={"ttft_ms": round(ttft_ms, 2)},
                    )
                accumulated_resp.append(token)
                yield {"type": "token", "token": token}

            synth_ms = synth_sw.stop()
            total_ms = pipeline_sw.stop()
            full_text = "".join(accumulated_resp)

            self.telemetry.log(
                event=TelemetryEventNames.SYNTHESIS_COMPLETED,
                session_id=sid,
                utterance_id=utterance_id,
                metadata={"total_ms": round(total_ms, 2), "tokens": len(accumulated_resp)},
            )

            timings = PipelineTimings(
                ttft_ms=ttft_ms,
                synthesis_ms=synth_ms,
                total_ms=total_ms,
            )

            self.session_manager.add_turn(
                session_id=sid,
                user_utterance=acc_text,
                controller_state=ControllerState.NO_RETRIEVE.value,
                assistant_response=full_text,
            )

            yield {"type": "timings", "timings": timings.to_dict()}
            return

        # Handle Controller State: RETRIEVE
        self.telemetry.log(
            event=TelemetryEventNames.RETRIEVAL_TRIGGERED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={
                "reason": decision.reason,
                "is_early_retrieval": decision.is_early_retrieval,
                "query": acc_text,
            },
        )
        yield {
            "type": "controller_decision",
            "state": ControllerState.RETRIEVE.value,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "is_early_retrieval": decision.is_early_retrieval,
        }

        # Check session contextualization / refinement
        effective_query = self.session_manager.contextualize_query(session, acc_text)

        # 2. Multi-Intent Query Decomposition
        decomposition = self.decomposer.decompose(effective_query)
        self.telemetry.log(
            event=TelemetryEventNames.QUERY_DECOMPOSED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={
                "subqueries": [sq.subquery_text for sq in decomposition.subqueries],
                "duration_ms": decomposition.duration_ms,
                "is_compound": decomposition.is_compound,
            },
        )
        yield {
            "type": "decomposition",
            "decomposition": decomposition.to_dict(),
        }

        # 3. Multi-Query Retrieval
        self.telemetry.log(
            event=TelemetryEventNames.RETRIEVAL_STARTED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"subquery_count": len(decomposition.subqueries)},
        )
        candidates, retrieval_ms = self.retriever.retrieve(decomposition)
        self.telemetry.log(
            event=TelemetryEventNames.RETRIEVAL_COMPLETED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"candidates_found": len(candidates), "retrieval_ms": round(retrieval_ms, 2)},
        )

        # 4. Evidence Fusion (RRF) & Deduplication
        fused_evidence, fusion_ms = self.fusion.fuse(candidates, top_n=self.config.top_k + 2)
        self.telemetry.log(
            event=TelemetryEventNames.FUSION_COMPLETED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"fused_count": len(fused_evidence), "fusion_ms": round(fusion_ms, 2)},
        )

        # 5. Reranking
        reranked_evidence, rerank_ms = self.reranker.rerank(
            query=effective_query,
            candidates=fused_evidence,
            top_n=self.config.top_k,
        )
        self.telemetry.log(
            event=TelemetryEventNames.RERANK_COMPLETED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={
                "reranked_count": len(reranked_evidence),
                "rerank_ms": round(rerank_ms, 2),
            },
        )

        # Cache evidence in session pool for subsequent follow-up turns
        self.session_manager.update_evidence_pool(sid, reranked_evidence)

        yield {
            "type": "evidence",
            "evidence": [ev.to_dict() for ev in reranked_evidence],
            "count": len(reranked_evidence),
        }

        # 6. Grounded Streaming Synthesis
        self.telemetry.log(
            event=TelemetryEventNames.SYNTHESIS_STARTED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"evidence_count": len(reranked_evidence)},
        )

        sys_prompt, user_prompt = self.grounding.build_synthesis_prompt(
            utterance=effective_query,
            evidence=reranked_evidence,
            session_context=session,
        )

        req = AgentRequest(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            temperature=self.config.temperature,
            stream=True,
        )

        accumulated_resp = []
        first_token_seen = False
        ttft_ms = 0.0
        synth_sw = Stopwatch()

        for token in self.provider.stream(req):
            if not first_token_seen:
                first_token_seen = True
                ttft_ms = pipeline_sw.elapsed_ms
                self.telemetry.log(
                    event=TelemetryEventNames.SYNTHESIS_FIRST_TOKEN,
                    session_id=sid,
                    utterance_id=utterance_id,
                    metadata={"ttft_ms": round(ttft_ms, 2)},
                )
            accumulated_resp.append(token)
            yield {"type": "token", "token": token}

        synth_ms = synth_sw.stop()
        total_ms = pipeline_sw.stop()
        full_text = "".join(accumulated_resp)

        self.telemetry.log(
            event=TelemetryEventNames.SYNTHESIS_COMPLETED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"total_ms": round(total_ms, 2), "tokens": len(accumulated_resp)},
        )

        # 7. Extract Citations
        citations = self.grounding.extract_citations(full_text, reranked_evidence)
        for cit in citations:
            self.telemetry.log(
                event=TelemetryEventNames.CITATION_GENERATED,
                session_id=sid,
                utterance_id=utterance_id,
                metadata=cit.to_dict(),
            )

        yield {
            "type": "citations",
            "citations": [c.to_dict() for c in citations],
        }

        # Record pipeline timings
        timings = PipelineTimings(
            ttft_ms=ttft_ms,
            decomposition_ms=decomposition.duration_ms,
            retrieval_ms=retrieval_ms + fusion_ms,
            reranking_ms=rerank_ms,
            synthesis_ms=synth_ms,
            total_ms=total_ms,
        )

        # Update Session Turn History
        self.session_manager.add_turn(
            session_id=sid,
            user_utterance=acc_text,
            controller_state=ControllerState.RETRIEVE.value,
            decomposed_queries=[sq.subquery_text for sq in decomposition.subqueries],
            retrieved_evidence_ids=[ev.chunk_id for ev in reranked_evidence],
            assistant_response=full_text,
            citations=[c.to_dict() for c in citations],
        )

        yield {
            "type": "timings",
            "timings": timings.to_dict(),
        }
