/**
 * AVY — Streaming Live RAG Assistant Frontend Application
 * Theme 04: Streaming Live RAG • Samsung PRISM GenAI Hackathon
 */

// Application State
let currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
let accumulatedTranscript = '';
let currentAssistantBubble = null;
let currentAssistantText = '';
let isStreamingActive = false;
let streamTimerInterval = null;
let streamStartTime = null;
let cachedEvidence = [];
let cachedCitations = [];

// DOM Element References
const connStatusPill = document.getElementById('conn-status-pill');
const connStatusText = document.getElementById('conn-status-text');
const headerModelVal = document.getElementById('header-model-val');
const headerCorpusVal = document.getElementById('header-corpus-val');
const headerSessionVal = document.getElementById('header-session-val');
const btnCopySession = document.getElementById('btn-copy-session');

const customInput = document.getElementById('custom-utterance-input');
const btnSubmitUtterance = document.getElementById('btn-submit-utterance');
const transcriptFeed = document.getElementById('transcript-feed');
const transcriptCount = document.getElementById('transcript-chunk-count');

const stateBoxWait = document.getElementById('state-box-wait');
const stateBoxRetrieve = document.getElementById('state-box-retrieve');
const stateBoxNoRetrieve = document.getElementById('state-box-noretrieve');
const controllerExplanation = document.getElementById('controller-explanation-text');
const earlyRetrievalBanner = document.getElementById('early-retrieval-banner');
const earlyLeadTimePill = document.getElementById('early-lead-time-pill');

const conversationThread = document.getElementById('conversation-thread');
const streamingStatusBar = document.getElementById('streaming-status-bar');
const streamingStatusLabel = document.getElementById('streaming-status-label');
const streamingLiveTimer = document.getElementById('streaming-live-timer');
const btnResetSession = document.getElementById('btn-reset-session');

const decompositionView = document.getElementById('decomposition-view');
const evidenceFusionView = document.getElementById('evidence-fusion-view');
const telemetryLogFeed = document.getElementById('telemetry-log-feed');

const kpiTtft = document.getElementById('kpi-ttft');
const kpiTotal = document.getElementById('kpi-total');
const kpiLead = document.getElementById('kpi-lead');
const kpiSources = document.getElementById('kpi-sources');

const tRefineVal = document.getElementById('t-refine-val');
const tDecompVal = document.getElementById('t-decomp-val');
const tRetrievalVal = document.getElementById('t-retrieval-val');
const tRerankVal = document.getElementById('t-rerank-val');
const tSynthesisVal = document.getElementById('t-synthesis-val');

// Modal Elements
const evidenceModalBackdrop = document.getElementById('evidence-modal-backdrop');
const btnCloseModal = document.getElementById('btn-close-modal');
const modalCitationTag = document.getElementById('modal-citation-tag');
const modalTitle = document.getElementById('modal-title');
const modalSourceFile = document.getElementById('modal-source-file');
const modalChunkId = document.getElementById('modal-chunk-id');
const modalVectorScore = document.getElementById('modal-vector-score');
const modalRrfScore = document.getElementById('modal-rrf-score');
const modalRerankScore = document.getElementById('modal-rerank-score');
const modalMatchedQueries = document.getElementById('modal-matched-queries');
const modalChunkText = document.getElementById('modal-chunk-text');

// Initialize on Load
document.addEventListener('DOMContentLoaded', async () => {
  headerSessionVal.textContent = currentSessionId;
  await checkHealth();
  setupEventListeners();
});

// Periodic health probe
async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'healthy') {
      connStatusPill.className = 'status-pill status-live';
      connStatusText.textContent = 'LIVE CONNECTED';
      headerModelVal.textContent = data.provider || 'Ready';
      headerCorpusVal.textContent = `${data.indexed_chunks} Chunks (FAISS)`;
    } else {
      connStatusPill.className = 'status-pill';
      connStatusText.textContent = 'DEGRADED';
    }
  } catch (err) {
    connStatusPill.className = 'status-pill';
    connStatusText.textContent = 'OFFLINE';
  }
}

// Event Listeners Setup
function setupEventListeners() {
  // Tab Switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      const content = document.getElementById(tabId);
      if (content) content.classList.add('active');
    });
  });

  // 1-Click Scenario Buttons
  document.querySelectorAll('.btn-scenario').forEach(btn => {
    btn.addEventListener('click', () => {
      const scenarioId = btn.getAttribute('data-scenario');
      runScenario(scenarioId);
    });
  });

  // Stream Chunk Buttons
  document.querySelectorAll('.btn-stream-chunk').forEach(btn => {
    btn.addEventListener('click', () => {
      const chunk = btn.getAttribute('data-chunk');
      sendChunk(chunk, false);
    });
  });

  // Custom Utterance Submit
  btnSubmitUtterance.addEventListener('click', () => {
    submitCustomQuery();
  });

  customInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      submitCustomQuery();
    }
  });

  // Copy Session ID
  btnCopySession.addEventListener('click', () => {
    navigator.clipboard.writeText(currentSessionId);
    btnCopySession.textContent = '✓';
    setTimeout(() => { btnCopySession.textContent = '📋'; }, 1500);
  });

  // Reset Session
  btnResetSession.addEventListener('click', resetSession);

  // Modal Close
  btnCloseModal.addEventListener('click', closeModal);
  evidenceModalBackdrop.addEventListener('click', (e) => {
    if (e.target === evidenceModalBackdrop) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
}

function submitCustomQuery() {
  const text = customInput.value.trim();
  if (!text) return;
  customInput.value = '';
  sendChunk(text, true);
}

function resetSession() {
  currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
  headerSessionVal.textContent = currentSessionId;
  accumulatedTranscript = '';
  currentAssistantBubble = null;
  currentAssistantText = '';
  cachedEvidence = [];
  cachedCitations = [];

  // Reset UI Views
  transcriptFeed.innerHTML = '<div class="empty-state-muted">Awaiting incoming spoken audio chunks...</div>';
  transcriptCount.textContent = '0 chunks';
  setControllerState('IDLE', 'Controller evaluates transcript completeness in real-time.');
  earlyRetrievalBanner.style.display = 'none';

  conversationThread.innerHTML = `
    <div class="welcome-banner">
      <div class="welcome-badge">AVY LIVE STREAMING RAG</div>
      <h3>Autonomous Conversational RAG with Early Retrieval</h3>
      <p>Speak naturally, supply incomplete chunks, ask compound questions, or converse casually. AVY gates retrieval, decomposes intents, fuses vector evidence with RRF, and streams grounded responses with deterministic citations.</p>
      <div class="welcome-hints">
        <span class="hint-tag">⚡ Early Retrieval</span>
        <span class="hint-tag">🔀 Multi-Intent Decomposition</span>
        <span class="hint-tag">📊 RRF Evidence Fusion</span>
        <span class="hint-tag">🎯 Two-Stage Reranking</span>
        <span class="hint-tag">🔗 Provenance Citations</span>
      </div>
    </div>
  `;

  resetPipelineStages();
  decompositionView.innerHTML = '<div class="empty-subtext">No active query decomposition. Run a query or scenario.</div>';
  evidenceFusionView.innerHTML = '<div class="empty-subtext">No evidence retrieved yet.</div>';
  telemetryLogFeed.innerHTML = '<div class="empty-subtext">Waiting for telemetry events...</div>';

  kpiTtft.textContent = '-- ms';
  kpiTotal.textContent = '-- ms';
  kpiLead.textContent = '-- ms';
  kpiSources.textContent = '--';

  tRefineVal.textContent = '-- ms';
  tDecompVal.textContent = '-- ms';
  tRetrievalVal.textContent = '-- ms';
  tRerankVal.textContent = '-- ms';
  tSynthesisVal.textContent = '-- ms';
}

function resetPipelineStages() {
  const stages = [
    'transcript', 'controller', 'refinement', 'decomposition',
    'vector_retrieval', 'fusion', 'reranking', 'grounding', 'synthesis', 'streaming'
  ];
  stages.forEach(st => {
    updateStageUI(st, 'idle', '--');
  });
}

function updateStageUI(stageName, status, duration) {
  const item = document.getElementById(`stage-${stageName}`);
  const timeSpan = document.getElementById(`stage-time-${stageName}`);
  const badge = document.getElementById(`stage-badge-${stageName}`);

  if (!item || !badge) return;

  item.className = 'stage-item';
  if (status === 'running') item.classList.add('running');
  else if (status === 'completed') item.classList.add('completed');
  else if (status === 'skipped') item.classList.add('skipped');

  badge.className = `stage-badge badge-${status}`;
  badge.textContent = status;

  if (duration !== undefined && timeSpan) {
    timeSpan.textContent = typeof duration === 'number' ? `${duration.toFixed(1)} ms` : duration;
  }
}

function setControllerState(state, explanation) {
  stateBoxWait.classList.remove('active');
  stateBoxRetrieve.classList.remove('active');
  stateBoxNoRetrieve.classList.remove('active');

  if (state === 'WAIT') {
    stateBoxWait.classList.add('active');
  } else if (state === 'RETRIEVE') {
    stateBoxRetrieve.classList.add('active');
  } else if (state === 'NO_RETRIEVE') {
    stateBoxNoRetrieve.classList.add('active');
  }

  if (controllerExplanation) {
    controllerExplanation.textContent = explanation || '';
  }
}

function appendTranscriptChunk(chunk, accumulated) {
  const empty = transcriptFeed.querySelector('.empty-state-muted');
  if (empty) empty.remove();

  const entry = document.createElement('div');
  entry.className = 'transcript-entry';
  entry.textContent = `"${chunk}"`;
  transcriptFeed.appendChild(entry);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;

  const count = transcriptFeed.querySelectorAll('.transcript-entry').length;
  transcriptCount.textContent = `${count} chunk${count > 1 ? 's' : ''}`;
}

function appendUserMessage(text) {
  const welcome = conversationThread.querySelector('.welcome-banner');
  if (welcome) welcome.remove();

  const msgDiv = document.createElement('div');
  msgDiv.className = 'message-bubble message-user';
  msgDiv.innerHTML = `
    <div class="bubble-body">${escapeHtml(text)}</div>
  `;
  conversationThread.appendChild(msgDiv);
  conversationThread.scrollTop = conversationThread.scrollHeight;
  return msgDiv;
}

function createAssistantBubble() {
  const welcome = conversationThread.querySelector('.welcome-banner');
  if (welcome) welcome.remove();

  const msgDiv = document.createElement('div');
  msgDiv.className = 'message-bubble message-assistant';
  msgDiv.innerHTML = `
    <div class="bubble-header">
      <span class="assistant-avatar">AVY</span>
      <span class="assistant-name">Grounded Response</span>
    </div>
    <div class="bubble-body assistant-body">
      <span class="spinner-dot" style="display:inline-block; vertical-align:middle; margin-right:6px;"></span>
      <span class="text-muted">Analyzing evidence and synthesizing answer...</span>
    </div>
  `;
  conversationThread.appendChild(msgDiv);
  conversationThread.scrollTop = conversationThread.scrollHeight;
  currentAssistantBubble = msgDiv;
  currentAssistantText = '';
  return msgDiv;
}

function startStreamingTimer() {
  isStreamingActive = true;
  streamStartTime = performance.now();
  streamingStatusBar.style.display = 'flex';
  if (streamTimerInterval) clearInterval(streamTimerInterval);

  streamTimerInterval = setInterval(() => {
    if (!isStreamingActive) return;
    const elapsed = Math.round(performance.now() - streamStartTime);
    streamingLiveTimer.textContent = `${elapsed} ms`;
  }, 50);
}

function stopStreamingTimer() {
  isStreamingActive = false;
  if (streamTimerInterval) clearInterval(streamTimerInterval);
  streamingStatusBar.style.display = 'none';
}

// Send Single Chunk or Incremental Input
async function sendChunk(chunk, isFinal) {
  if (accumulatedTranscript) {
    accumulatedTranscript += ' ' + chunk;
  } else {
    accumulatedTranscript = chunk;
  }

  appendTranscriptChunk(chunk, accumulatedTranscript);
  appendUserMessage(chunk);
  startStreamingTimer();

  const payload = {
    chunk: chunk,
    accumulated: accumulatedTranscript,
    session_id: currentSessionId,
    is_final: isFinal
  };

  try {
    const response = await fetch('/api/stream/rag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    await processSSEStream(response);
  } catch (err) {
    console.error('Stream processing error:', err);
    stopStreamingTimer();
  }
}

// 1-Click Scenario Execution
async function runScenario(scenarioId) {
  resetSession();
  startStreamingTimer();

  try {
    const response = await fetch('/api/scenarios/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario_id: scenarioId,
        session_id: currentSessionId
      })
    });

    await processSSEStream(response);
  } catch (err) {
    console.error('Scenario execution error:', err);
    stopStreamingTimer();
  }
}

// Core SSE Stream Consumer
async function processSSEStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const event = JSON.parse(line.substring(6));
          handlePipelineEvent(event);
        } catch (e) {
          console.error('Failed to parse SSE event:', e);
        }
      }
    }
  }

  stopStreamingTimer();
}

// Handle All Pipeline Events
function handlePipelineEvent(event) {
  const type = event.type;

  // Append to Telemetry Tab
  appendTelemetryRecord(event);

  if (type === 'turn_start') {
    appendUserMessage(event.text);
    return;
  }

  if (type === 'transcript') {
    appendTranscriptChunk(event.chunk, event.accumulated);
  }

  else if (type === 'pipeline_stage') {
    updateStageUI(event.stage, event.status, event.duration_ms);
  }

  else if (type === 'controller_decision') {
    setControllerState(event.state, event.reason);

    if (event.is_early_retrieval) {
      earlyRetrievalBanner.style.display = 'block';
      if (event.early_lead_time_ms) {
        earlyLeadTimePill.textContent = `+${Math.round(event.early_lead_time_ms)}ms Lead Time`;
      }
    } else {
      earlyRetrievalBanner.style.display = 'none';
    }

    if (event.state === 'RETRIEVE' || event.state === 'NO_RETRIEVE') {
      createAssistantBubble();
    }
  }

  else if (type === 'query_refinement') {
    // Add refinement pill to latest user bubble
    const userBubbles = conversationThread.querySelectorAll('.message-user');
    if (userBubbles.length > 0) {
      const lastUser = userBubbles[userBubbles.length - 1];
      const pill = document.createElement('div');
      pill.className = 'refinement-badge';
      pill.innerHTML = `<span>🔗 Contextualized:</span> <strong>${escapeHtml(event.resolved_query)}</strong>`;
      lastUser.appendChild(pill);
    }
  }

  else if (type === 'decomposition') {
    renderDecomposition(event.decomposition);
  }

  else if (type === 'evidence') {
    cachedEvidence = event.evidence || [];
    kpiSources.textContent = cachedEvidence.length;
    renderEvidenceFusion(cachedEvidence);
  }

  else if (type === 'token') {
    if (!currentAssistantBubble) {
      createAssistantBubble();
    }
    currentAssistantText += event.token;
    renderStreamingAnswer(currentAssistantBubble, currentAssistantText);
  }

  else if (type === 'citations') {
    cachedCitations = event.citations || [];
    if (currentAssistantBubble) {
      renderSourcesTray(currentAssistantBubble, cachedCitations);
    }
  }

  else if (type === 'timings') {
    stopStreamingTimer();
    renderTimings(event.timings);
  }
}

// Render Streaming Text with Clickable Citation Links
function renderStreamingAnswer(bubble, rawText) {
  const body = bubble.querySelector('.assistant-body');
  if (!body) return;

  // Replace [1], [2], etc. with interactive clickable citation links
  let formatted = escapeHtml(rawText);
  formatted = formatted.replace(/\[(\d+)\]/g, (match, p1) => {
    return `<button class="citation-link" data-citation="${p1}" title="Inspect Citation [${p1}]">[${p1}]</button>`;
  });

  body.innerHTML = formatted;
  conversationThread.scrollTop = conversationThread.scrollHeight;

  // Attach click listeners to citation links
  body.querySelectorAll('.citation-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.stopPropagation();
      const citIdx = parseInt(link.getAttribute('data-citation'), 10);
      openEvidenceModal(citIdx);
    });
  });
}

// Render Sources Tray below assistant message
function renderSourcesTray(bubble, citations) {
  if (!citations || citations.length === 0) return;

  const existingTray = bubble.querySelector('.sources-tray');
  if (existingTray) existingTray.remove();

  const tray = document.createElement('div');
  tray.className = 'sources-tray';
  tray.innerHTML = `
    <div class="sources-tray-title">Grounded Evidence Sources:</div>
    <div class="sources-pills-list"></div>
  `;

  const pillsList = tray.querySelector('.sources-pills-list');
  citations.forEach(c => {
    const btn = document.createElement('button');
    btn.className = 'source-pill-btn';
    btn.innerHTML = `<span class="citation-tag">[${c.index}]</span> <span>${escapeHtml(c.source)}</span>`;
    btn.addEventListener('click', () => {
      openEvidenceModal(c.index);
    });
    pillsList.appendChild(btn);
  });

  bubble.querySelector('.bubble-body').appendChild(tray);
}

// Render Intent Decomposition in Tab 2
function renderDecomposition(decomposition) {
  if (!decomposition || !decomposition.subqueries || decomposition.subqueries.length === 0) {
    decompositionView.innerHTML = '<div class="empty-subtext">Single intent query; no decomposition required.</div>';
    return;
  }

  decompositionView.innerHTML = '';
  decomposition.subqueries.forEach((sq, i) => {
    const card = document.createElement('div');
    card.className = 'subquery-card';
    card.innerHTML = `
      <div class="subquery-header">
        <span class="subquery-intent-tag">${sq.intent_label || `Subquery ${i + 1}`}</span>
        <span class="font-mono text-faint" style="font-size:9px;">#${sq.query_id || i + 1}</span>
      </div>
      <div class="subquery-text">"${escapeHtml(sq.subquery_text)}"</div>
    `;
    decompositionView.appendChild(card);
  });
}

// Render Evidence Fusion & Reranking Explorer in Tab 3
function renderEvidenceFusion(evidenceList) {
  if (!evidenceList || evidenceList.length === 0) {
    evidenceFusionView.innerHTML = '<div class="empty-subtext">No candidates retrieved.</div>';
    return;
  }

  evidenceFusionView.innerHTML = '';
  evidenceList.forEach((ev, idx) => {
    const card = document.createElement('div');
    card.className = 'evidence-score-card';

    const vScore = ev.vector_score ? ev.vector_score.toFixed(3) : '0.000';
    const rrfScore = ev.rrf_score ? ev.rrf_score.toFixed(3) : '0.000';
    const rerankScore = ev.rerank_score !== null && ev.rerank_score !== undefined
      ? Number(ev.rerank_score).toFixed(3)
      : 'N/A';

    card.innerHTML = `
      <div class="ev-header">
        <span class="citation-tag">[${idx + 1}]</span>
        <div class="ev-doc-title" title="${escapeHtml(ev.title)}">${escapeHtml(ev.title)}</div>
        <span class="tag font-mono">${escapeHtml(ev.source)}</span>
      </div>
      <div class="ev-score-trio">
        <div class="score-col">
          <span class="score-lbl">Vector</span>
          <span class="score-val">${vScore}</span>
        </div>
        <div class="score-col">
          <span class="score-lbl">RRF Fusion</span>
          <span class="score-val">${rrfScore}</span>
        </div>
        <div class="score-col">
          <span class="score-lbl">Rerank</span>
          <span class="score-val">${rerankScore}</span>
        </div>
      </div>
      <div class="ev-snippet">${escapeHtml(ev.text.substring(0, 160))}...</div>
    `;

    card.addEventListener('click', () => {
      openEvidenceModal(idx + 1);
    });

    evidenceFusionView.appendChild(card);
  });
}

// Render Latency & TTFT Timings in Tab 4
function renderTimings(timings) {
  if (!timings) return;

  if (timings.ttft_ms) kpiTtft.textContent = `${Math.round(timings.ttft_ms)} ms`;
  if (timings.total_ms) kpiTotal.textContent = `${Math.round(timings.total_ms)} ms`;
  if (timings.early_lead_time_ms) kpiLead.textContent = `+${Math.round(timings.early_lead_time_ms)} ms`;

  if (timings.refinement_ms !== undefined) tRefineVal.textContent = `${timings.refinement_ms.toFixed(1)} ms`;
  if (timings.decomposition_ms !== undefined) tDecompVal.textContent = `${timings.decomposition_ms.toFixed(1)} ms`;
  if (timings.retrieval_ms !== undefined) tRetrievalVal.textContent = `${timings.retrieval_ms.toFixed(1)} ms`;
  if (timings.reranking_ms !== undefined) tRerankVal.textContent = `${timings.reranking_ms.toFixed(1)} ms`;
  if (timings.synthesis_ms !== undefined) tSynthesisVal.textContent = `${timings.synthesis_ms.toFixed(1)} ms`;
}

// Append Telemetry Record to Tab 5
function appendTelemetryRecord(record) {
  const empty = telemetryLogFeed.querySelector('.empty-subtext');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'telem-record';

  const now = new Date().toLocaleTimeString();
  const eventName = record.type || record.event || 'event';
  const metaStr = JSON.stringify(record, null, 2);

  item.innerHTML = `
    <div class="telem-header">
      <span class="telem-event-name">${escapeHtml(eventName)}</span>
      <span class="telem-time font-mono">${now}</span>
    </div>
    <div class="telem-meta">${escapeHtml(metaStr)}</div>
  `;

  telemetryLogFeed.prepend(item);
  // Keep max 40 entries
  while (telemetryLogFeed.children.length > 40) {
    telemetryLogFeed.removeChild(telemetryLogFeed.lastChild);
  }
}

// Evidence Inspector Modal Controller
function openEvidenceModal(citationIndex) {
  let ev = null;

  // Look up in cached evidence
  if (cachedEvidence && cachedEvidence.length >= citationIndex && citationIndex > 0) {
    ev = cachedEvidence[citationIndex - 1];
  }

  // Fallback to citation metadata
  let cit = null;
  if (cachedCitations && cachedCitations.length >= citationIndex && citationIndex > 0) {
    cit = cachedCitations[citationIndex - 1];
  }

  modalCitationTag.textContent = `[${citationIndex}]`;
  modalTitle.textContent = ev ? ev.title : (cit ? cit.title : `Citation [${citationIndex}]`);
  modalSourceFile.textContent = ev ? ev.source : (cit ? cit.source : '--');
  modalChunkId.textContent = ev ? ev.chunk_id : (cit ? cit.chunk_id : '--');

  modalVectorScore.textContent = ev && ev.vector_score ? ev.vector_score.toFixed(4) : '--';
  modalRrfScore.textContent = ev && ev.rrf_score ? ev.rrf_score.toFixed(4) : '--';
  modalRerankScore.textContent = ev && ev.rerank_score !== null && ev.rerank_score !== undefined
    ? Number(ev.rerank_score).toFixed(4)
    : '--';

  modalMatchedQueries.textContent = ev && ev.matched_queries && ev.matched_queries.length > 0
    ? ev.matched_queries.join(', ')
    : 'Primary query match';

  modalChunkText.textContent = ev ? ev.text : (cit ? cit.snippet : 'No raw text snippet available.');

  evidenceModalBackdrop.style.display = 'flex';
}

function closeModal() {
  evidenceModalBackdrop.style.display = 'none';
}

// XSS Prevention Utility
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
