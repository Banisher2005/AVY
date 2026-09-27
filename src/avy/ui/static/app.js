/**
 * AVY — Streaming Live RAG Assistant
 * Voice-First Frontend Application
 *
 * State Machine:
 *   IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
 *
 * Transport:
 *   Primary:  WebSocket /ws/{session_id}
 *   Fallback: SSE POST /api/query
 *
 * Voice:
 *   STT: Web Speech API (browser-native, Chrome/Edge)
 *   TTS: Web SpeechSynthesis API (browser-native)
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════
   CONSTANTS
════════════════════════════════════════════════════════════════ */

const State = Object.freeze({
  IDLE:       'IDLE',
  LISTENING:  'LISTENING',
  PROCESSING: 'PROCESSING',
  SPEAKING:   'SPEAKING',
  ERROR:      'ERROR',
});

const STAGE_META = {
  transcript:      { n: '01', label: 'Utterance Received' },
  controller:      { n: '02', label: 'Retrieval Decision' },
  refinement:      { n: '03', label: 'Query Refinement' },
  decomposition:   { n: '04', label: 'Intent Analysis' },
  vector_retrieval:{ n: '05', label: 'Evidence Search' },
  fusion:          { n: '06', label: 'Evidence Fusion (RRF)' },
  reranking:       { n: '07', label: 'Evidence Reranking' },
  grounding:       { n: '08', label: 'Grounding Check' },
  synthesis:       { n: '09', label: 'Generating Answer' },
  streaming:       { n: '10', label: 'Streaming Response' },
};

/* ═══════════════════════════════════════════════════════════════
   APPLICATION STATE
════════════════════════════════════════════════════════════════ */

let appState     = State.IDLE;
let sessionId    = 'sess_' + Math.random().toString(36).substr(2, 10);
let ws           = null;
let wsReady      = false;
let recognition  = null;
let hasMic       = false;
let currentAnswer = null;     // current AVY response bubble element
let currentAnswerText = '';   // accumulated raw answer text
let cachedEvidence = [];      // latest retrieved evidence chunks
let cachedCitations = [];     // latest extracted citations
let speechSynth  = window.speechSynthesis;
let ttsActive    = false;
let ttsQueue     = [];
let chunkBuffer  = [];        // utterance chunks buffered from simulator
let pipelineActive = false;   // guard against duplicate executions

/* ═══════════════════════════════════════════════════════════════
   DOM REFERENCES
════════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

const connPill        = $('conn-pill');
const connDot         = $('conn-dot');
const connLabel       = $('conn-label');
const headerModel     = $('header-model');
const headerCorpus    = $('header-corpus');
const headerSessionId = $('header-session-id');
const btnClear        = $('btn-clear');

const stateBanner     = $('state-banner');
const stateIcon       = $('state-icon');
const stateLabel      = $('state-label');
const stateSub        = $('state-sub');
const transcriptArea  = $('transcript-area');
const liveTranscript  = $('live-transcript');
const conversation    = $('conversation');

const micBtn          = $('mic-btn');
const micLabel        = $('mic-label');
const ttsControl      = $('tts-control');
const btnStopTts      = $('btn-stop-tts');

const textInput       = $('text-input');
const btnSend         = $('btn-send');
const noMicNotice     = $('no-mic-notice');

const traceContainer  = $('trace-container');
const evidenceContainer = $('evidence-container');

/* ═══════════════════════════════════════════════════════════════
   INITIALIZATION
════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  headerSessionId.textContent = sessionId;
  initWebSocket();
  initSpeechRecognition();
  initEventListeners();
  loadScenarios();
  checkHealth();
  resetTraceStages();
});

/* ═══════════════════════════════════════════════════════════════
   STATE MACHINE
════════════════════════════════════════════════════════════════ */

function setState(newState, label, sub) {
  appState = newState;

  const icons = {
    [State.IDLE]:       '🎙',
    [State.LISTENING]:  '🎙',
    [State.PROCESSING]: '🧠',
    [State.SPEAKING]:   '🔊',
    [State.ERROR]:      '⚠️',
  };

  stateIcon.textContent  = icons[newState] || '●';
  stateLabel.textContent = label;
  stateSub.textContent   = sub || '';

  // Banner class
  stateBanner.className = 'state-banner state-' + newState.toLowerCase();

  // Mic button class
  micBtn.className = 'mic-btn ' + newState.toLowerCase();

  // Mic label
  const micLabels = {
    [State.IDLE]:       'TAP TO SPEAK',
    [State.LISTENING]:  'LISTENING — TAP TO STOP',
    [State.PROCESSING]: 'PROCESSING...',
    [State.SPEAKING]:   'SPEAKING — TAP TO INTERRUPT',
    [State.ERROR]:      'TAP TO RETRY',
  };
  micLabel.textContent = micLabels[newState] || '';

  // TTS control visibility
  ttsControl.classList.toggle('hidden', newState !== State.SPEAKING);
  btnSend.disabled = (newState === State.PROCESSING);
}

/* ═══════════════════════════════════════════════════════════════
   WEBSOCKET
════════════════════════════════════════════════════════════════ */

function initWebSocket() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${location.host}/ws/${sessionId}`;

  try {
    ws = new WebSocket(url);
  } catch (e) {
    console.warn('WebSocket unavailable, using SSE fallback');
    return;
  }

  ws.onopen = () => {
    wsReady = true;
    setConnected(true);
    // Start health refresh
    checkHealth();
    // Heartbeat
    setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  };

  ws.onmessage = e => {
    try {
      handleServerEvent(JSON.parse(e.data));
    } catch (err) {
      console.error('WS parse error:', err);
    }
  };

  ws.onclose = () => {
    wsReady = false;
    setConnected(false);
    // Reconnect after 3s
    setTimeout(initWebSocket, 3000);
  };

  ws.onerror = () => {
    wsReady = false;
    setConnected(false);
  };
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
    return true;
  }
  return false;
}

function setConnected(ok) {
  connPill.className = 'conn-pill ' + (ok ? 'live' : 'offline');
  connLabel.textContent = ok ? 'LIVE CONNECTED' : 'RECONNECTING...';
}

/* ═══════════════════════════════════════════════════════════════
   HEALTH CHECK
════════════════════════════════════════════════════════════════ */

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const d = await res.json();
    if (d.status === 'healthy') {
      setConnected(true);
      headerModel.textContent = d.provider || '—';
      headerCorpus.textContent = `${d.indexed_chunks} chunks`;
    }
  } catch (e) {
    setConnected(false);
  }
}

/* ═══════════════════════════════════════════════════════════════
   SPEECH RECOGNITION (STT)
════════════════════════════════════════════════════════════════ */

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    hasMic = false;
    noMicNotice.classList.remove('hidden');
    micBtn.style.opacity = '0.4';
    micBtn.title = 'Speech recognition not available — use text input';
    return;
  }

  hasMic = true;
  recognition = new SpeechRecognition();
  recognition.continuous    = false;
  recognition.interimResults = true;
  recognition.lang          = 'en-US';
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    setState(State.LISTENING, 'Listening...', 'Speak now — I\'ll answer when you\'re done');
    transcriptArea.classList.remove('hidden');
    liveTranscript.textContent = '—';
  };

  recognition.onresult = event => {
    let interim = '';
    let finalText = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        finalText += event.results[i][0].transcript;
      } else {
        interim  += event.results[i][0].transcript;
      }
    }

    liveTranscript.textContent = finalText || interim || '—';

    if (finalText) {
      // Finalize and process
      transcriptArea.classList.add('hidden');
      processUtterance(finalText.trim());
    }
  };

  recognition.onerror = event => {
    transcriptArea.classList.add('hidden');
    if (event.error === 'not-allowed') {
      hasMic = false;
      noMicNotice.classList.remove('hidden');
      noMicNotice.textContent = '⚠️ Microphone permission denied — using text input mode.';
      setState(State.IDLE, 'Ready', 'Type your question below');
    } else if (event.error === 'no-speech') {
      setState(State.IDLE, 'Ready', 'No speech detected — press the mic to try again');
    } else if (event.error === 'aborted') {
      setState(State.IDLE, 'Ready', 'Press the microphone to speak');
    } else {
      setState(State.ERROR, 'Microphone Error', event.error);
      setTimeout(() => setState(State.IDLE, 'Ready', 'Press to speak'), 3000);
    }
  };

  recognition.onend = () => {
    if (appState === State.LISTENING) {
      setState(State.IDLE, 'Ready', 'Press the microphone to speak');
    }
    transcriptArea.classList.add('hidden');
  };
}

function startListening() {
  if (!hasMic) return;
  if (appState === State.SPEAKING) stopTTS();
  if (appState === State.LISTENING) {
    recognition.stop();
    return;
  }
  if (appState !== State.IDLE && appState !== State.ERROR) return;
  try {
    recognition.start();
  } catch (e) {
    // Already started — stop and restart
    recognition.stop();
    setTimeout(() => { try { recognition.start(); } catch (_) {} }, 300);
  }
}

/* ═══════════════════════════════════════════════════════════════
   UTTERANCE PROCESSING
════════════════════════════════════════════════════════════════ */

/**
 * Process a COMPLETE utterance through the RAG pipeline.
 * One utterance → one pipeline execution → one answer.
 */
function processUtterance(text) {
  if (!text || !text.trim()) return;
  if (pipelineActive) {
    console.warn('Pipeline already active — ignoring duplicate request');
    return;
  }

  // Stop TTS if AVY was speaking
  if (appState === State.SPEAKING) stopTTS();

  pipelineActive = true;
  setState(State.PROCESSING, 'Understanding...', `"${text.length > 60 ? text.substr(0, 60) + '…' : text}"`);

  // Add user message to conversation
  addUserMessage(text);

  // Reset pipeline UI
  resetTraceStages();
  clearEvidence();

  // Create AVY response bubble (with typing indicator)
  currentAnswer = createAnswerBubble();
  currentAnswerText = '';
  cachedEvidence = [];
  cachedCitations = [];

  if (wsReady) {
    // Primary: WebSocket
    wsSend({ type: 'utterance', text });
  } else {
    // Fallback: SSE
    processViaSSE(text);
  }
}

/* ── SSE Fallback ──────────────────────────────────────────── */
async function processViaSSE(text) {
  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, session_id: sessionId }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        if (part.startsWith('data: ')) {
          try {
            handleServerEvent(JSON.parse(part.slice(6)));
          } catch (e) {}
        }
      }
    }
    onPipelineCompleted();
  } catch (e) {
    onPipelineError(e.message);
  }
}

/* ═══════════════════════════════════════════════════════════════
   SERVER EVENT HANDLER
════════════════════════════════════════════════════════════════ */

function handleServerEvent(ev) {
  const t = ev.type;

  if (t === 'pong') return;

  if (t === 'pipeline.started') {
    setState(State.PROCESSING, 'Retrieving & Thinking...', 'Running the RAG pipeline');
    return;
  }

  if (t === 'pipeline.completed') {
    onPipelineCompleted();
    return;
  }

  if (t === 'pipeline.cancelled') {
    pipelineActive = false;
    setState(State.IDLE, 'Ready', 'Press the microphone to speak');
    return;
  }

  if (t === 'session.cleared') {
    setState(State.IDLE, 'Ready', 'Session reset — press the mic to start');
    return;
  }

  if (t === 'utterance.buffered') {
    // Show buffering state in transcript area
    transcriptArea.classList.remove('hidden');
    liveTranscript.textContent = ev.accumulated || ev.chunk;
    return;
  }

  if (t === 'utterance.finalized') {
    transcriptArea.classList.add('hidden');
    return;
  }

  if (t === 'turn_start') {
    // Multi-turn scenario: new turn starting
    if (ev.text) addUserMessage(ev.text);
    currentAnswer = createAnswerBubble();
    currentAnswerText = '';
    resetTraceStages();
    clearEvidence();
    return;
  }

  if (t === 'pipeline_stage') {
    updateTraceStage(ev.stage, ev.status, ev.duration_ms, ev.details);
    return;
  }

  if (t === 'controller_decision') {
    updateControllerDecision(ev);
    return;
  }

  if (t === 'query_refinement') {
    showRefinementBadge(ev.original_query, ev.resolved_query);
    return;
  }

  if (t === 'evidence') {
    cachedEvidence = ev.evidence || [];
    renderEvidence(cachedEvidence);
    $('kpi-sources').textContent = cachedEvidence.length;
    return;
  }

  if (t === 'decomposition') {
    // Log decomposition info — could expand to dedicated panel
    const sqs = ev.decomposition?.subqueries || [];
    if (sqs.length > 1) {
      appendTelemetryNote(`Decomposed into ${sqs.length} subqueries: ${sqs.map(s => '"' + s.subquery_text + '"').join(', ')}`);
    }
    return;
  }

  if (t === 'token') {
    if (!currentAnswer) currentAnswer = createAnswerBubble();
    currentAnswerText += ev.token;
    renderAnswerText(currentAnswer, currentAnswerText);
    return;
  }

  if (t === 'citations') {
    cachedCitations = ev.citations || [];
    if (currentAnswer) renderSourcesTray(currentAnswer, cachedCitations);
    return;
  }

  if (t === 'timings') {
    renderTimings(ev.timings);
    return;
  }

  if (t === 'error') {
    onPipelineError(ev.message || 'Unknown error');
    return;
  }
}

/* ═══════════════════════════════════════════════════════════════
   PIPELINE LIFECYCLE
════════════════════════════════════════════════════════════════ */

function onPipelineCompleted() {
  pipelineActive = false;
  const hasAnswer = currentAnswerText.trim().length > 0;

  if (hasAnswer) {
    // Start TTS
    speakResponse(currentAnswerText);
  } else {
    setState(State.IDLE, 'Ready', 'Press the microphone to speak');
  }
}

function onPipelineError(msg) {
  pipelineActive = false;
  const isNetworkFailure = String(msg).toLowerCase().includes('failed to fetch') || String(msg).toLowerCase().includes('network');
  const userMessage = isNetworkFailure
    ? `Unable to reach AVY backend on ${location.host}. Please verify scripts/demo_server.py is running on port ${location.port || 8001}.`
    : `Pipeline error: ${msg}`;

  setState(State.ERROR, 'Backend Error', userMessage);

  if (currentAnswer) {
    const body = currentAnswer.querySelector('.msg-bubble');
    if (body) {
      body.innerHTML = `
        <div style="color:var(--red); font-size:13px; line-height:1.6;">
          <strong>⚠️ ${escHtml(userMessage)}</strong>
          ${!isNetworkFailure ? `<details style="margin-top:6px; opacity:0.8; font-size:11px;"><summary>Technical Details</summary><pre style="white-space:pre-wrap; margin-top:4px;">${escHtml(msg)}</pre></details>` : ''}
        </div>
      `;
    }
  }

  setTimeout(() => setState(State.IDLE, 'Ready', 'Press to speak'), 6000);
}

/* ═══════════════════════════════════════════════════════════════
   TEXT-TO-SPEECH
════════════════════════════════════════════════════════════════ */

function speakResponse(rawText) {
  if (!speechSynth) {
    setState(State.IDLE, 'Ready', 'Press the microphone to speak');
    return;
  }

  // Strip citation markers and clean text for speech
  const cleanText = rawText
    .replace(/\[\d+\]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  if (!cleanText) {
    setState(State.IDLE, 'Ready', 'Press the microphone to speak');
    return;
  }

  setState(State.SPEAKING, 'Speaking...', 'Tap to interrupt');

  // Split into sentence-sized chunks for natural TTS
  const sentences = cleanText.match(/[^.!?]+[.!?]+|\S[^.!?]*/g) || [cleanText];

  ttsQueue = [...sentences];
  ttsActive = true;
  speakNext();
}

function speakNext() {
  if (!ttsActive || ttsQueue.length === 0) {
    ttsActive = false;
    setState(State.IDLE, 'Ready', 'Press the microphone to speak');
    return;
  }

  const sentence = ttsQueue.shift().trim();
  if (!sentence) { speakNext(); return; }

  const utter = new SpeechSynthesisUtterance(sentence);
  utter.rate  = 1.0;
  utter.pitch = 1.0;
  utter.volume = 1.0;
  utter.onend = () => {
    if (ttsActive) speakNext();
  };
  utter.onerror = () => {
    if (ttsActive) speakNext();
  };

  speechSynth.speak(utter);
}

function stopTTS() {
  ttsActive = false;
  ttsQueue  = [];
  if (speechSynth) speechSynth.cancel();
  setState(State.IDLE, 'Ready', 'Press the microphone to speak');
}

/* ═══════════════════════════════════════════════════════════════
   CONVERSATION UI
════════════════════════════════════════════════════════════════ */

function addUserMessage(text) {
  removeWelcome();
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-avatar">U</div>
    <div class="msg-bubble">${escHtml(text)}</div>
  `;
  conversation.appendChild(row);
  scrollDown();
  return row;
}

function createAnswerBubble() {
  removeWelcome();
  const row = document.createElement('div');
  row.className = 'msg-row avy';
  row.innerHTML = `
    <div class="msg-avatar">A</div>
    <div class="msg-bubble">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  conversation.appendChild(row);
  scrollDown();
  return row;
}

function renderAnswerText(bubble, rawText) {
  const body = bubble.querySelector('.msg-bubble');
  if (!body) return;

  // Convert [1], [2] to clickable links; escape HTML first
  let html = escHtml(rawText).replace(/\[(\d+)\]/g, (_, n) =>
    `<button class="cit-link" data-cit="${n}" title="View source [${n}]">[${n}]</button>`
  );

  body.innerHTML = html;

  // Attach click listeners
  body.querySelectorAll('.cit-link').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      openEvidenceModal(parseInt(btn.getAttribute('data-cit'), 10));
    });
  });

  scrollDown();
}

function renderSourcesTray(bubble, citations) {
  if (!citations || citations.length === 0) return;
  const body = bubble.querySelector('.msg-bubble');
  if (!body) return;

  // Remove existing tray
  const existing = body.querySelector('.sources-tray');
  if (existing) existing.remove();

  const tray = document.createElement('div');
  tray.className = 'sources-tray';
  citations.forEach(c => {
    const btn = document.createElement('button');
    btn.className = 'source-pill';
    btn.textContent = `[${c.index}] ${c.source}`;
    btn.addEventListener('click', () => openEvidenceModal(c.index));
    tray.appendChild(btn);
  });
  body.appendChild(tray);
  scrollDown();
}

function showRefinementBadge(original, resolved) {
  const lastUser = [...conversation.querySelectorAll('.msg-row.user')].pop();
  if (!lastUser) return;
  const existing = lastUser.querySelector('.refinement-badge');
  if (existing) return;
  const badge = document.createElement('div');
  badge.className = 'refinement-badge';
  badge.textContent = `🔗 Resolved: "${resolved}"`;
  lastUser.querySelector('.msg-bubble')?.appendChild(badge);
}

function removeWelcome() {
  const w = conversation.querySelector('.welcome');
  if (w) w.remove();
}

function scrollDown() {
  requestAnimationFrame(() => { conversation.scrollTop = conversation.scrollHeight; });
}

/* ═══════════════════════════════════════════════════════════════
   PIPELINE TRACE
════════════════════════════════════════════════════════════════ */

function resetTraceStages() {
  traceContainer.innerHTML = '';
  Object.entries(STAGE_META).forEach(([key, meta]) => {
    const el = document.createElement('div');
    el.className = 'trace-stage idle';
    el.id = `stage-${key}`;
    el.innerHTML = `
      <span class="trace-num">${meta.n}</span>
      <span class="trace-name">${meta.label}</span>
      <span class="trace-dur" id="stage-dur-${key}">—</span>
      <span class="trace-badge badge-idle" id="stage-badge-${key}">idle</span>
    `;
    traceContainer.appendChild(el);
  });

  // Reset controller banner if exists
  const cb = traceContainer.querySelector('.controller-banner');
  if (cb) cb.remove();
}

function updateTraceStage(stageName, status, durationMs, details) {
  const el = document.getElementById(`stage-${stageName}`);
  if (!el) return;

  el.className = `trace-stage ${status}`;
  const badge = document.getElementById(`stage-badge-${stageName}`);
  if (badge) { badge.className = `trace-badge badge-${status}`; badge.textContent = status; }

  const dur = document.getElementById(`stage-dur-${stageName}`);
  if (dur && durationMs !== undefined && durationMs !== null) {
    dur.textContent = `${Number(durationMs).toFixed(1)} ms`;
  }
}

function updateControllerDecision(ev) {
  const state = (ev.state || '').toLowerCase();

  // Remove existing banner
  const existing = traceContainer.querySelector('.controller-banner');
  if (existing) existing.remove();

  const labels = {
    retrieve:    `⚡ RETRIEVE${ev.is_early_retrieval ? ' (Early)' : ''}`,
    no_retrieve: '💬 NO_RETRIEVE — Conversational fast-path',
    wait:        '⏳ WAIT — Utterance incomplete',
  };
  const banner = document.createElement('div');
  banner.className = `controller-banner ${state}`;
  banner.textContent = labels[state] || ev.state;

  // Insert before first non-controller stage
  const firstStage = traceContainer.querySelector('.trace-stage');
  if (firstStage) {
    traceContainer.insertBefore(banner, firstStage);
  } else {
    traceContainer.appendChild(banner);
  }

  // Update KPI
  $('kpi-controller').textContent = ev.state || '—';

  // For WAIT state, show transparent response
  if (state === 'wait') {
    pipelineActive = false;
    if (currentAnswer) {
      const body = currentAnswer.querySelector('.msg-bubble');
      if (body) body.innerHTML = `<div class="wait-response">⏳ Still listening… Utterance seems incomplete.</div>`;
    }
    setState(State.IDLE, 'Ready', 'Utterance seemed incomplete — please rephrase or continue');
  }
}

/* ═══════════════════════════════════════════════════════════════
   EVIDENCE PANEL
════════════════════════════════════════════════════════════════ */

function clearEvidence() {
  evidenceContainer.innerHTML = '<div class="trace-empty">Evidence will appear after retrieval.</div>';
}

function renderEvidence(evidence) {
  if (!evidence || evidence.length === 0) { clearEvidence(); return; }

  evidenceContainer.innerHTML = '';
  evidence.forEach((ev, i) => {
    const card = document.createElement('div');
    card.className = 'ev-card';
    const vScore = ev.vector_score ? Number(ev.vector_score).toFixed(3) : '—';
    const rrf    = ev.rrf_score   ? Number(ev.rrf_score).toFixed(3)   : '—';
    const rerank = ev.rerank_score !== undefined && ev.rerank_score !== null
      ? Number(ev.rerank_score).toFixed(3) : '—';

    card.innerHTML = `
      <div class="ev-card-header">
        <span class="ev-idx">[${i + 1}]</span>
        <span class="ev-title" title="${escHtml(ev.title)}">${escHtml(ev.title)}</span>
        <span class="ev-source">${escHtml(ev.source || '')}</span>
      </div>
      <div class="ev-scores">
        <span class="ev-score">Vector <strong>${vScore}</strong></span>
        <span class="ev-score">RRF <strong>${rrf}</strong></span>
        <span class="ev-score">Rerank <strong>${rerank}</strong></span>
      </div>
      <div class="ev-snippet">${escHtml((ev.text || '').substring(0, 160))}${ev.text?.length > 160 ? '…' : ''}</div>
    `;
    card.addEventListener('click', () => openEvidenceModal(i + 1));
    evidenceContainer.appendChild(card);
  });

  // Open evidence panel automatically when we have results
  openPanelIfCollapsed('evidence');
}

/* ═══════════════════════════════════════════════════════════════
   ANALYTICS
════════════════════════════════════════════════════════════════ */

function renderTimings(timings) {
  if (!timings) return;

  const fmt = v => v !== undefined && v !== null ? Math.round(v) + ' ms' : '—';
  $('kpi-ttft').textContent      = fmt(timings.ttft_ms);
  $('kpi-total').textContent     = fmt(timings.total_ms);
  $('kpi-retrieval').textContent = fmt(timings.retrieval_ms);
  $('kpi-reranking').textContent = fmt(timings.reranking_ms);

  const details = $('timing-details');
  if (details) {
    details.innerHTML = [
      timings.refinement_ms   ? `Refinement: <strong>${timings.refinement_ms.toFixed(1)} ms</strong>` : '',
      timings.decomposition_ms ? `Decomposition: <strong>${timings.decomposition_ms.toFixed(1)} ms</strong>` : '',
      timings.retrieval_ms    ? `Retrieval+Fusion: <strong>${timings.retrieval_ms.toFixed(1)} ms</strong>` : '',
      timings.reranking_ms    ? `Reranking: <strong>${timings.reranking_ms.toFixed(1)} ms</strong>` : '',
      timings.synthesis_ms    ? `Synthesis: <strong>${timings.synthesis_ms.toFixed(1)} ms</strong>` : '',
      timings.early_lead_time_ms ? `Early Lead: <strong>+${timings.early_lead_time_ms.toFixed(1)} ms</strong>` : '',
    ].filter(Boolean).join('<br>');
  }
}

/* ═══════════════════════════════════════════════════════════════
   SCENARIO RUNNER
════════════════════════════════════════════════════════════════ */

async function loadScenarios() {
  try {
    const res = await fetch('/api/scenarios');
    const scenarios = await res.json();
    const grid = $('scenario-grid');
    if (!grid) return;
    grid.innerHTML = '';

    scenarios.forEach(sc => {
      const btn = document.createElement('button');
      btn.className = 'scenario-btn';
      const stateClass = sc.expected_state.toLowerCase();
      btn.innerHTML = `
        <strong>${sc.id}</strong>: ${escHtml(sc.name.replace(/Scenario [A-Z]: /, ''))}
        <span class="scenario-state ${stateClass}">${sc.expected_state}</span>
      `;
      btn.addEventListener('click', () => runScenario(sc.id));
      grid.appendChild(btn);
    });
  } catch (e) {
    console.warn('Could not load scenarios:', e);
  }
}

async function runScenario(scenarioId) {
  if (pipelineActive) return;

  resetSession(false);
  pipelineActive = true;
  setState(State.PROCESSING, 'Running Scenario...', scenarioId);
  openPanelIfCollapsed('trace');

  try {
    const res = await fetch('/api/scenarios/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario_id: scenarioId, session_id: sessionId }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        if (part.startsWith('data: ')) {
          try { handleServerEvent(JSON.parse(part.slice(6))); } catch (e) {}
        }
      }
    }

    onPipelineCompleted();
  } catch (e) {
    onPipelineError(e.message);
  }
}

/* ═══════════════════════════════════════════════════════════════
   CHUNK SIMULATOR
════════════════════════════════════════════════════════════════ */

function initChunkSimulator() {
  document.querySelectorAll('.btn-chunk').forEach(btn => {
    btn.addEventListener('click', () => {
      const chunk   = btn.getAttribute('data-chunk');
      const isFinal = btn.getAttribute('data-final') === 'true';

      chunkBuffer.push(chunk);

      // Show in transcript area
      transcriptArea.classList.remove('hidden');
      liveTranscript.textContent = chunkBuffer.join(' ');

      if (wsReady) {
        wsSend({ type: 'chunk', chunk, is_final: isFinal });
      }

      if (isFinal) {
        // Also handle via SSE if no WS, using accumulated buffer
        if (!wsReady) {
          const fullText = chunkBuffer.join(' ');
          chunkBuffer = [];
          processViaSSE(fullText).then(onPipelineCompleted);
        } else {
          chunkBuffer = [];
        }
      }
    });
  });

  $('btn-clear-chunks')?.addEventListener('click', () => {
    chunkBuffer = [];
    transcriptArea.classList.add('hidden');
    liveTranscript.textContent = '—';
    if (wsReady) wsSend({ type: 'clear' });
  });
}

/* ═══════════════════════════════════════════════════════════════
   SESSION MANAGEMENT
════════════════════════════════════════════════════════════════ */

function resetSession(sendClear = true) {
  pipelineActive = false;
  stopTTS();

  sessionId = 'sess_' + Math.random().toString(36).substr(2, 10);
  headerSessionId.textContent = sessionId;
  chunkBuffer = [];
  cachedEvidence = [];
  cachedCitations = [];
  currentAnswer = null;
  currentAnswerText = '';

  // Reset UI
  conversation.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">🎙</div>
      <h2 class="welcome-title">Speak to AVY</h2>
      <p class="welcome-desc">Session reset. Press the mic or type a new question.</p>
    </div>
  `;

  transcriptArea.classList.add('hidden');
  resetTraceStages();
  clearEvidence();

  ['kpi-ttft','kpi-total','kpi-retrieval','kpi-reranking','kpi-sources','kpi-controller']
    .forEach(id => { const el = $(id); if (el) el.textContent = '—'; });
  const td = $('timing-details');
  if (td) td.innerHTML = '';

  setState(State.IDLE, 'Ready', 'Press the microphone to speak');

  if (sendClear && wsReady) {
    // Re-init WS with new session ID
    initWebSocket();
  }
}

/* ═══════════════════════════════════════════════════════════════
   EVIDENCE MODAL
════════════════════════════════════════════════════════════════ */

function openEvidenceModal(citationIndex) {
  const ev  = cachedEvidence[citationIndex - 1] || null;
  const cit = cachedCitations.find(c => c.index === citationIndex) || null;

  const backdrop = $('modal-backdrop');
  if (!backdrop) return;

  $('modal-citation-tag').textContent = `[${citationIndex}]`;
  $('modal-title').textContent = ev?.title || cit?.title || `Citation [${citationIndex}]`;
  $('modal-source').textContent  = ev?.source  || cit?.source  || '—';
  $('modal-chunk-id').textContent = ev?.chunk_id || cit?.chunk_id || '—';

  $('modal-vector-score').textContent = ev?.vector_score ? Number(ev.vector_score).toFixed(4) : '—';
  $('modal-rrf-score').textContent    = ev?.rrf_score    ? Number(ev.rrf_score).toFixed(4)    : '—';
  $('modal-rerank-score').textContent = ev?.rerank_score !== undefined && ev?.rerank_score !== null
    ? Number(ev.rerank_score).toFixed(4) : '—';

  $('modal-text').textContent = ev?.text || cit?.snippet
    || 'Evidence text not available for this citation index.';

  backdrop.classList.remove('hidden');
}

function closeModal(e) {
  if (e && e.target !== $('modal-backdrop')) return;
  $('modal-backdrop')?.classList.add('hidden');
}

/* ═══════════════════════════════════════════════════════════════
   PANEL COLLAPSING
════════════════════════════════════════════════════════════════ */

function togglePanel(name) {
  const panel   = document.getElementById(`panel-${name}`);
  const chevron = document.getElementById(`chevron-${name}`);
  if (!panel) return;
  panel.classList.toggle('collapsed');
  if (chevron) chevron.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
}

function openPanelIfCollapsed(name) {
  const panel = document.getElementById(`panel-${name}`);
  if (panel && panel.classList.contains('collapsed')) togglePanel(name);
}

/* ═══════════════════════════════════════════════════════════════
   EVENT LISTENERS
════════════════════════════════════════════════════════════════ */

function initEventListeners() {
  // Mic button
  micBtn.addEventListener('click', () => {
    if (!hasMic) return;
    if (appState === State.SPEAKING) {
      stopTTS();
      return;
    }
    startListening();
  });

  // Stop TTS
  btnStopTts?.addEventListener('click', stopTTS);

  // Text input
  btnSend?.addEventListener('click', submitText);
  textInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) submitText();
  });

  // Clear session
  btnClear?.addEventListener('click', () => resetSession(true));

  // Modal close
  $('modal-backdrop')?.addEventListener('click', closeModal);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') $('modal-backdrop')?.classList.add('hidden');
  });

  // Chunk simulator
  initChunkSimulator();
}

function submitText() {
  const text = textInput?.value.trim();
  if (!text) return;
  textInput.value = '';
  processUtterance(text);
}

/* ═══════════════════════════════════════════════════════════════
   UTILITIES
════════════════════════════════════════════════════════════════ */

function escHtml(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function appendTelemetryNote(msg) {
  // Light telemetry for decomposition etc — just console for now
  console.log('[AVY]', msg);
}
