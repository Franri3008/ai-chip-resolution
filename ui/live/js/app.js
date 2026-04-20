// AI Chip Resolution — live dashboard. SSE with polling fallback.

const STAGE_ORDER = [
  'fetch_models', 'modelcards', 'github_urls', 'arxiv_urls',
  'eval_github', 'eval_arxiv', 'classify', 'resolve'
];

const _stageStates = {};
let _builtStages = false;

function buildStages(data) {
  const pipeline = document.getElementById('pipeline');
  const labels = data.stage_labels || {};
  STAGE_ORDER.forEach((key, idx) => {
    if (idx > 0) {
      const conn = document.createElement('div');
      conn.className = 'connector';
      conn.innerHTML = '<svg viewBox="0 0 48 16" preserveAspectRatio="none">' +
        '<line x1="0" y1="8" x2="38" y2="8" />' +
        '<polyline points="32,3 42,8 32,13" />' +
        '</svg>';
      pipeline.appendChild(conn);
    }
    const stage = document.createElement('div');
    stage.className = 'stage';
    stage.id = 'stage-' + key;
    stage.dataset.stage = key;
    stage.innerHTML = `
      <div class="stage-head">
        <span class="step-num">${idx + 1}</span>
        <div>
          <div class="stage-name">${labels[key] || key}</div>
          <div class="stage-status" id="badge-${key}">pending</div>
        </div>
      </div>
      <div class="stage-body">
        <div class="stat">
          <span class="sv" id="count-${key}">—</span>
          <span class="sl">done</span>
        </div>
        <div class="stat err" id="errstat-${key}" hidden>
          <span class="sv" id="errs-${key}">0</span>
          <span class="sl">errors</span>
        </div>
        <div class="stat">
          <span class="sv sm" id="time-${key}">—</span>
          <span class="sl">time</span>
        </div>
      </div>
      <div class="stage-current" id="current-${key}"></div>
    `;
    pipeline.appendChild(stage);
  });
  _builtStages = true;
}

function animateParticle(fromKey, toKey) {
  const fromEl = document.getElementById('stage-' + fromKey);
  const toEl = document.getElementById('stage-' + toKey);
  const particle = document.getElementById('particle');
  const pipeline = document.querySelector('.pipeline');
  if (!fromEl || !toEl || !particle || !pipeline) return;

  const pr = pipeline.getBoundingClientRect();
  const fr = fromEl.getBoundingClientRect();
  const tr = toEl.getBoundingClientRect();

  const fromX = fr.left + fr.width / 2 - pr.left - 11;
  const fromY = fr.top + fr.height / 2 - pr.top - 11;
  const toX = tr.left + tr.width / 2 - pr.left - 11;
  const toY = tr.top + tr.height / 2 - pr.top - 11;

  particle.textContent = '🔍';
  particle.style.transition = 'none';
  particle.style.transform = `translate(${fromX}px, ${fromY}px)`;
  particle.style.opacity = '1';

  setTimeout(() => {
    particle.style.transition = 'transform 0.6s cubic-bezier(0.4,0,0.2,1), opacity 0.3s ease 0.5s';
    particle.style.transform = `translate(${toX}px, ${toY}px)`;
    particle.style.opacity = '0';
  }, 30);
}

function applyState(data) {
  if (!_builtStages) buildStages(data);

  setText('run-id', data.run_id);
  setText('args-summary', data.args_summary || '');
  setText('elapsed', fmtTime(data.elapsed_s));

  const total = data.total_models || 0;
  const resolved = (data.summary && data.summary.resolved) || 0;
  const pct = total > 0 ? Math.min(100, (resolved / total) * 100) : 0;
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('progress-label').textContent =
    total > 0 ? `${resolved} / ${total}` : '—';

  STAGE_ORDER.forEach((key) => {
    const s = data.stages[key];
    if (!s) return;
    updateStage(key, s);
  });

  setText('resolved-count', resolved + ' resolved');
  updateLatest(data.llm, data.summary);
  updateRecent(data.llm);
  updateSummary(data.summary, data.llm);
}

function updateStage(key, s) {
  const el = document.getElementById('stage-' + key);
  const badge = document.getElementById('badge-' + key);
  if (!el || !badge) return;

  const prev = _stageStates[key];
  _stageStates[key] = s.status;
  if (prev !== 'running' && s.status === 'running') {
    const idx = STAGE_ORDER.indexOf(key);
    if (idx > 0) animateParticle(STAGE_ORDER[idx - 1], key);
  }

  el.classList.remove('active', 'done', 'error');
  if (s.status === 'running') el.classList.add('active');
  else if (s.status === 'done') el.classList.add('done');
  else if (s.status === 'error') el.classList.add('error');
  badge.textContent = s.status;

  setText('count-' + key, s.completed != null ? s.completed : '—');
  setText('time-' + key, fmtSec(s.elapsed_s));
  setText('current-' + key, truncate(s.current || s.note || '', 26));

  const errStat = document.getElementById('errstat-' + key);
  if (s.errors && s.errors > 0) {
    errStat.hidden = false;
    setText('errs-' + key, s.errors);
  } else {
    errStat.hidden = true;
  }
}

function updateLatest(llm, summary) {
  const box = document.getElementById('latest-resolution');
  if (!llm || !llm.last_model_id) {
    box.innerHTML = '<div class="lr-empty">Waiting for first model…</div>';
    return;
  }
  const chipCls = (!llm.last_chip || llm.last_chip === 'unknown') ? 'chip-badge unknown' : 'chip-badge';
  box.innerHTML = `
    <div class="lr-model-id">${escapeHtml(llm.last_model_id)}</div>
    <div class="lr-grid">
      <div>
        <div class="lr-kv-label">Chip</div>
        <div class="lr-kv-value"><span class="${chipCls}">${escapeHtml(llm.last_chip || 'unknown')}</span></div>
      </div>
      <div>
        <div class="lr-kv-label">Confidence</div>
        <div class="lr-kv-value">${(llm.last_confidence || 0).toFixed(2)}</div>
      </div>
      <div>
        <div class="lr-kv-label">Source</div>
        <div class="lr-kv-value">${escapeHtml(llm.last_source || '—')}</div>
      </div>
      <div>
        <div class="lr-kv-label">LLM fallbacks</div>
        <div class="lr-kv-value">${llm.chip_fallback_calls || 0}</div>
      </div>
    </div>
  `;
}

function updateRecent(llm) {
  const list = document.getElementById('recent-list');
  const items = (llm && llm.recent_results) || [];
  if (items.length === 0) {
    list.innerHTML = '<div class="lr-empty">No results yet.</div>';
    return;
  }
  list.innerHTML = items.map(r => {
    const unknown = !r.chip || r.chip === 'unknown';
    return `
      <div class="recent-row">
        <span class="recent-id" title="${escapeHtml(r.id)}">${escapeHtml(r.id)}</span>
        <span class="recent-chip${unknown ? ' unknown' : ''}">${escapeHtml(r.chip || 'unknown')}</span>
        <span class="recent-conf">${(r.confidence || 0).toFixed(2)} · ${escapeHtml(r.source || '—')}</span>
      </div>
    `;
  }).join('');
}

function updateSummary(summary, llm) {
  if (!summary) return;
  const byChip = summary.by_chip || {};
  const bySrc = summary.by_source || {};
  document.getElementById('summary-chip').innerHTML =
    Object.entries(byChip).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `<span class="sg-pill">${escapeHtml(k)}<span class="sg-pill-count">${v}</span></span>`)
      .join('') || '<span class="sg-pill">—</span>';
  document.getElementById('summary-source').innerHTML =
    Object.entries(bySrc).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `<span class="sg-pill">${escapeHtml(k)}<span class="sg-pill-count">${v}</span></span>`)
      .join('') || '<span class="sg-pill">—</span>';
  setText('llm-calls', (llm && llm.chip_fallback_calls) || 0);
  const cost = (llm && llm.chip_fallback_cost) || 0;
  setText('llm-cost', '$' + cost.toFixed(cost < 0.01 ? 6 : 4));
}

// Helpers
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val != null && val !== '' ? val : '—';
}

function fmtSec(s) {
  if (s == null || s === 0) return '—';
  return s.toFixed(1) + 's';
}

function fmtTime(s) {
  if (s == null) return '0.0s';
  if (s < 60) return s.toFixed(1) + 's';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + 'm ' + sec + 's';
}

function truncate(str, len) {
  if (!str) return '';
  return str.length > len ? str.slice(0, len) + '…' : str;
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Connect
async function tick() {
  try {
    const res = await fetch('status.json?' + Date.now());
    if (!res.ok) return;
    applyState(await res.json());
  } catch {}
}

if (location.protocol === 'http:' || location.protocol === 'https:') {
  const es = new EventSource('/api/status/stream');
  es.onmessage = (e) => {
    try { applyState(JSON.parse(e.data)); } catch {}
  };
  es.onerror = () => {
    es.close();
    tick();
    setInterval(tick, 1000);
  };
} else {
  tick();
  setInterval(tick, 1000);
}
