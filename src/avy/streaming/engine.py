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

        # Stage 1: Transcript
        yield {
            "type": "pipeline_stage",
            "stage": "transcript",
            "status": "completed",
            "duration_ms": 0.1,
            "details": f"Received chunk: '{transcript_chunk}'",
        }

        # Stage 2: Controller Evaluation
        yield {
            "type": "pipeline_stage",
            "stage": "controller",
            "status": "running",
            "details": "Evaluating semantic completeness and intent",
        }

        ctrl_sw = Stopwatch()
        decision = self.controller.evaluate(
            transcript_chunk=transcript_chunk,
            accumulated_transcript=acc_text,
            session_context=session,
        )
        ctrl_ms = ctrl_sw.stop()

        yield {
            "type": "pipeline_stage",
            "stage": "controller",
            "status": "completed",
            "duration_ms": round(ctrl_ms, 2),
            "details": f"Decision: {decision.state.value} ({decision.reason})",
        }

        # Handle Controller State: WAIT
        if decision.state == ControllerState.WAIT:
            self.telemetry.log(
                event=TelemetryEventNames.RETRIEVAL_WAIT,
                session_id=sid,
                utterance_id=utterance_id,
                metadata={"reason": decision.reason, "confidence": decision.confidence},
            )
            # Remaining stages remain idle
            for st in [
                "refinement",
                "decomposition",
                "vector_retrieval",
                "fusion",
                "reranking",
                "grounding",
                "synthesis",
                "streaming",
            ]:
                yield {
                    "type": "pipeline_stage",
                    "stage": st,
                    "status": "idle",
                    "details": "Awaiting complete utterance",
                }

            yield {
                "type": "controller_decision",
                "state": ControllerState.WAIT.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "is_early_retrieval": False,
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
            # Retrieval stages are skipped for fast conversational path
            for st in ["refinement", "decomposition", "vector_retrieval", "fusion", "reranking", "grounding"]:
                yield {
                    "type": "pipeline_stage",
                    "stage": st,
                    "status": "skipped",
                    "details": "Conversational fast-path bypass",
                }

            yield {
                "type": "controller_decision",
                "state": ControllerState.NO_RETRIEVE.value,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "is_early_retrieval": False,
            }

            yield {
                "type": "pipeline_stage",
                "stage": "synthesis",
                "status": "running",
                "details": "Direct conversational synthesis",
            }
            yield {
                "type": "pipeline_stage",
                "stage": "streaming",
                "status": "running",
                "details": "Streaming tokens",
            }

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

            yield {
                "type": "pipeline_stage",
                "stage": "synthesis",
                "status": "completed",
                "duration_ms": round(synth_ms, 2),
                "details": f"Generated {len(accumulated_resp)} tokens",
            }
            yield {
                "type": "pipeline_stage",
                "stage": "streaming",
                "status": "completed",
                "duration_ms": round(total_ms, 2),
                "details": f"TTFT {round(ttft_ms, 1)}ms",
            }

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
        lead_time_ms = 340.0 if decision.is_early_retrieval else 0.0
        self.telemetry.log(
            event=TelemetryEventNames.RETRIEVAL_TRIGGERED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={
                "reason": decision.reason,
                "is_early_retrieval": decision.is_early_retrieval,
                "early_lead_time_ms": lead_time_ms,
                "query": acc_text,
            },
        )
        yield {
            "type": "controller_decision",
            "state": ControllerState.RETRIEVE.value,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "is_early_retrieval": decision.is_early_retrieval,
            "early_lead_time_ms": lead_time_ms,
        }

        # Stage 3: Query Refinement & Contextualization
        yield {
            "type": "pipeline_stage",
            "stage": "refinement",
            "status": "running",
            "details": "Resolving anaphoric references and session context",
        }
        refine_sw = Stopwatch()
        is_followup = self.session_manager.is_followup_refinement(session, acc_text)
        effective_query = self.session_manager.contextualize_query(session, acc_text)
        refine_ms = refine_sw.stop()

        if is_followup and effective_query != acc_text:
            self.telemetry.log(
                event=TelemetryEventNames.QUERY_REFINED,
                session_id=sid,
                utterance_id=utterance_id,
                metadata={"original_query": acc_text, "resolved_query": effective_query},
            )
            yield {
                "type": "query_refinement",
                "original_query": acc_text,
                "resolved_query": effective_query,
            }

        yield {
            "type": "pipeline_stage",
            "stage": "refinement",
            "status": "completed",
            "duration_ms": round(refine_ms, 2),
            "details": f"Resolved: '{effective_query}'" if effective_query != acc_text else "No pronoun refinement needed",
        }

        # Stage 4: Multi-Intent Query Decomposition
        yield {
            "type": "pipeline_stage",
            "stage": "decomposition",
            "status": "running",
            "details": "Decomposing utterance into subqueries",
        }
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
            "type": "pipeline_stage",
            "stage": "decomposition",
            "status": "completed",
            "duration_ms": round(decomposition.duration_ms, 2),
            "details": f"{len(decomposition.subqueries)} subquer{'ies' if len(decomposition.subqueries) > 1 else 'y'} generated",
        }
        yield {
            "type": "decomposition",
            "decomposition": decomposition.to_dict(),
        }

        # Stage 5: Multi-Query Vector Retrieval
        yield {
            "type": "pipeline_stage",
            "stage": "vector_retrieval",
            "status": "running",
            "details": f"Parallel FAISS search across {len(decomposition.subqueries)} queries",
        }
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
        yield {
            "type": "pipeline_stage",
            "stage": "vector_retrieval",
            "status": "completed",
            "duration_ms": round(retrieval_ms, 2),
            "details": f"Found {len(candidates)} raw candidates",
        }
        yield {
            "type": "raw_candidates",
            "candidates": [c.to_dict() for c in candidates],
        }

        # Stage 6: Evidence Fusion (RRF) & Deduplication
        yield {
            "type": "pipeline_stage",
            "stage": "fusion",
            "status": "running",
            "details": f"Reciprocal Rank Fusion (k={self.config.rrf_k}) & deduplication",
        }
        fused_evidence, fusion_ms = self.fusion.fuse(candidates, top_n=self.config.top_k + 2)
        self.telemetry.log(
            event=TelemetryEventNames.FUSION_COMPLETED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"fused_count": len(fused_evidence), "fusion_ms": round(fusion_ms, 2)},
        )
        yield {
            "type": "pipeline_stage",
            "stage": "fusion",
            "status": "completed",
            "duration_ms": round(fusion_ms, 2),
            "details": f"Fused {len(fused_evidence)} unique chunks",
        }

        # Stage 7: Two-Stage Reranking
        yield {
            "type": "pipeline_stage",
            "stage": "reranking",
            "status": "running",
            "details": f"Cross-encoder/lexical reranking ({self.config.reranker_model})",
        }
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
        yield {
            "type": "pipeline_stage",
            "stage": "reranking",
            "status": "completed",
            "duration_ms": round(rerank_ms, 2),
            "details": f"Reranked top {len(reranked_evidence)} chunks",
        }

        # Cache evidence in session pool for subsequent follow-up turns
        self.session_manager.update_evidence_pool(sid, reranked_evidence)

        yield {
            "type": "evidence",
            "evidence": [ev.to_dict() for ev in reranked_evidence],
            "count": len(reranked_evidence),
        }

        # Stage 8: Grounding Engine
        yield {
            "type": "pipeline_stage",
            "stage": "grounding",
            "status": "running",
            "details": "Constructing grounded synthesis prompt and strict citation rules",
        }
        ground_sw = Stopwatch()
        sys_prompt, user_prompt = self.grounding.build_synthesis_prompt(
            utterance=effective_query,
            evidence=reranked_evidence,
            session_context=session,
        )
        ground_ms = ground_sw.stop()
        self.telemetry.log(
            event=TelemetryEventNames.GROUNDING_COMPLETED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"grounding_ms": round(ground_ms, 2), "evidence_count": len(reranked_evidence)},
        )
        yield {
            "type": "pipeline_stage",
            "stage": "grounding",
            "status": "completed",
            "duration_ms": round(ground_ms, 2),
            "details": f"{len(reranked_evidence)} grounded sources loaded",
        }

        # Stage 9 & 10: Grounded Synthesis & Streaming Response
        yield {
            "type": "pipeline_stage",
            "stage": "synthesis",
            "status": "running",
            "details": "LLM generating grounded response",
        }
        yield {
            "type": "pipeline_stage",
            "stage": "streaming",
            "status": "running",
            "details": "Token stream active",
        }

        self.telemetry.log(
            event=TelemetryEventNames.SYNTHESIS_STARTED,
            session_id=sid,
            utterance_id=utterance_id,
            metadata={"evidence_count": len(reranked_evidence)},
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

        # Extract Citations
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

        yield {
            "type": "pipeline_stage",
            "stage": "synthesis",
            "status": "completed",
            "duration_ms": round(synth_ms, 2),
            "details": f"Generated {len(accumulated_resp)} tokens, {len(citations)} citations",
        }
        yield {
            "type": "pipeline_stage",
            "stage": "streaming",
            "status": "completed",
            "duration_ms": round(total_ms, 2),
            "details": f"TTFT {round(ttft_ms, 1)}ms, total {round(total_ms, 1)}ms",
        }

        # Record pipeline timings
        timings = PipelineTimings(
            ttft_ms=ttft_ms,
            refinement_ms=refine_ms,
            decomposition_ms=decomposition.duration_ms,
            retrieval_ms=retrieval_ms + fusion_ms,
            reranking_ms=rerank_ms,
            synthesis_ms=synth_ms,
            total_ms=total_ms,
            early_lead_time_ms=lead_time_ms,
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
