/**
 * AVY — Streaming Live RAG Assistant Frontend Client
 * Samsung PRISM GenAI Hackathon (Theme 04)
 */

let currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
let accumulatedTranscript = '';

// DOM Elements
const statusLabel = document.getElementById('status-label');
const scenarioSelect = document.getElementById('scenario-select');
const btnRunScenario = document.getElementById('btn-run-scenario');
const inputQuery = document.getElementById('input-query');
const btnSend = document.getElementById('btn-send');
const transcriptFeed = document.getElementById('transcript-feed');
const transcriptCount = document.getElementById('transcript-count');
const controllerBadge = document.getElementById('controller-badge');
const controllerReason = document.getElementById('controller-reason');
const earlyRetrievalFlag = document.getElementById('early-retrieval-flag');
const subqueriesList = document.getElementById('subqueries-list');
const evidenceCards = document.getElementById('evidence-cards');
const evidenceCount = document.getElementById('evidence-count');
const responseText = document.getElementById('response-text');
const streamingIndicator = document.getElementById('streaming-indicator');
const citationsList = document.getElementById('citations-list');
const ttftPill = document.getElementById('ttft-pill');
const totalPill = document.getElementById('total-pill');
const sessionDisplay = document.getElementById('session-display');
const btnClearSession = document.getElementById('btn-clear-session');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  sessionDisplay.textContent = currentSessionId;
  await checkHealth();
  await loadScenarios();
  setupEventListeners();
});

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'healthy') {
      statusLabel.textContent = `Ready • ${data.provider} • ${data.indexed_chunks} Chunks`;
    } else {
      statusLabel.textContent = 'Degraded';
    }
  } catch (err) {
    statusLabel.textContent = 'Offline / Connecting';
  }
}

async function loadScenarios() {
  try {
    const res = await fetch('/api/scenarios');
    const scenarios = await res.json();
    scenarioSelect.innerHTML = '<option value="" disabled selected>Select a benchmark scenario...</option>';
    scenarios.forEach(sc => {
      const opt = document.createElement('option');
      opt.value = sc.id;
      opt.textContent = `${sc.name} (${sc.expected_state})`;
      scenarioSelect.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to load scenarios:', err);
  }
}

function setupEventListeners() {
  btnSend.addEventListener('click', () => {
    const q = inputQuery.value.trim();
    if (q) {
      sendStreamChunk(q, true);
      inputQuery.value = '';
    }
  });

  inputQuery.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      btnSend.click();
    }
  });

  // Chip buttons simulating incremental speech chunks
  document.querySelectorAll('.btn-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const chunk = btn.getAttribute('data-chunk');
      sendStreamChunk(chunk, false);
    });
  });

  btnRunScenario.addEventListener('click', runSelectedScenario);

  btnClearSession.addEventListener('click', () => {
    currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
    sessionDisplay.textContent = currentSessionId;
    accumulatedTranscript = '';
    resetViews();
  });
}

function resetViews() {
  transcriptFeed.innerHTML = '<div class="empty-state">Waiting for spoken audio chunks...</div>';
  transcriptCount.textContent = '0 chunks';
  setControllerState('IDLE', 'Controller evaluates transcript completeness in real time.');
  earlyRetrievalFlag.style.display = 'none';
  subqueriesList.innerHTML = '<div class="empty-subtext">No active query decomposition.</div>';
  evidenceCards.innerHTML = '<div class="empty-subtext">Evidence candidates will appear here upon retrieval.</div>';
  evidenceCount.textContent = '0 chunks';
  responseText.innerHTML = '<div class="empty-state">Synthesized answer with citations will stream here...</div>';
  citationsList.innerHTML = '<div class="empty-subtext">No citations generated yet.</div>';
  ttftPill.textContent = 'TTFT: -- ms';
  totalPill.textContent = 'Total: -- ms';
}

function setControllerState(state, reason) {
  controllerBadge.textContent = state;
  controllerBadge.className = 'controller-tag';
  if (state === 'WAIT') controllerBadge.classList.add('state-WAIT');
  else if (state === 'RETRIEVE') controllerBadge.classList.add('state-RETRIEVE');
  else if (state === 'NO_RETRIEVE') controllerBadge.classList.add('state-NO_RETRIEVE');
  controllerReason.textContent = reason || '';
}

function appendTranscriptChunk(chunk, accumulated) {
  const empty = transcriptFeed.querySelector('.empty-state');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'transcript-chunk-item';
  item.textContent = `"${chunk}"`;
  transcriptFeed.appendChild(item);
  transcriptFeed.scrollTop = transcriptFeed.scrollHeight;

  const count = transcriptFeed.querySelectorAll('.transcript-chunk-item').length;
  transcriptCount.textContent = `${count} chunk${count > 1 ? 's' : ''}`;
}

async function sendStreamChunk(chunk, isFinal) {
  if (accumulatedTranscript) {
    accumulatedTranscript += ' ' + chunk;
  } else {
    accumulatedTranscript = chunk;
  }

  appendTranscriptChunk(chunk, accumulatedTranscript);

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
          } catch (e) {}
        }
      }
    }
  } catch (err) {
    console.error('Stream error:', err);
  }
}

async function runSelectedScenario() {
  const scId = scenarioSelect.value;
  if (!scId) return;

  resetViews();
  btnRunScenario.disabled = true;
  btnRunScenario.textContent = 'Running...';

  try {
    const response = await fetch('/api/scenarios/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario_id: scId, session_id: currentSessionId })
    });

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
          } catch (e) {}
        }
      }
    }
  } catch (err) {
    console.error('Scenario run error:', err);
  } finally {
    btnRunScenario.disabled = false;
    btnRunScenario.textContent = '▶ Run Scenario';
  }
}

function handlePipelineEvent(event) {
  const type = event.type;

  if (type === 'transcript') {
    appendTranscriptChunk(event.chunk, event.accumulated);
  } else if (type === 'controller_decision') {
    setControllerState(event.state, event.reason);
    if (event.is_early_retrieval) {
      earlyRetrievalFlag.style.display = 'block';
    } else {
      earlyRetrievalFlag.style.display = 'none';
    }
    if (event.state === 'RETRIEVE' || event.state === 'NO_RETRIEVE') {
      streamingIndicator.style.display = 'inline-block';
      const empty = responseText.querySelector('.empty-state');
      if (empty) responseText.innerHTML = '';
    }
  } else if (type === 'decomposition') {
    const subs = event.decomposition.subqueries || [];
    subqueriesList.innerHTML = '';
    subs.forEach(sq => {
      const item = document.createElement('div');
      item.className = 'subquery-item';
      item.textContent = `• ${sq.subquery_text}`;
      subqueriesList.appendChild(item);
    });
  } else if (type === 'evidence') {
    const evList = event.evidence || [];
    evidenceCards.innerHTML = '';
    evidenceCount.textContent = `${evList.length} chunks`;
    evList.forEach((ev, i) => {
      const card = document.createElement('div');
      card.className = 'evidence-card';
      const rrf = ev.rrf_score ? `RRF: ${ev.rrf_score.toFixed(3)}` : '';
      const rerank = ev.rerank_score ? ` | Rerank: ${ev.rerank_score.toFixed(3)}` : '';
      card.innerHTML = `
        <div class="evidence-meta">
          <span>[${i+1}] ${ev.source} (${ev.chunk_id})</span>
          <span>${rrf}${rerank}</span>
        </div>
        <div class="evidence-title">${ev.title}</div>
        <div class="evidence-text">${ev.text.substring(0, 180)}...</div>
      `;
      evidenceCards.appendChild(card);
    });
  } else if (type === 'token') {
    streamingIndicator.style.display = 'inline-block';
    const empty = responseText.querySelector('.empty-state');
    if (empty) responseText.innerHTML = '';
    responseText.innerHTML += event.token;
  } else if (type === 'citations') {
    streamingIndicator.style.display = 'none';
    const cits = event.citations || [];
    citationsList.innerHTML = '';
    cits.forEach(c => {
      const pill = document.createElement('span');
      pill.className = 'citation-pill';
      pill.title = `${c.title} (${c.source})`;
      pill.textContent = `[${c.index}] ${c.source} : ${c.chunk_id}`;
      citationsList.appendChild(pill);
    });
  } else if (type === 'timings') {
    streamingIndicator.style.display = 'none';
    const t = event.timings;
    if (t.ttft_ms) ttftPill.textContent = `TTFT: ${t.ttft_ms.toFixed(0)} ms`;
    if (t.total_ms) totalPill.textContent = `Total: ${t.total_ms.toFixed(0)} ms`;
  }
}
