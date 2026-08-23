/* ═══════════════════════════════════════════════
   DOM Helpers
   ═══════════════════════════════════════════════ */
function $(id) {
  return document.getElementById(id);
}

function $$(sel) {
  return document.querySelectorAll(sel);
}

/* ═══════════════════════════════════════════════
   App State
   ═══════════════════════════════════════════════ */
const appState = {
  currentPage: 'dashboard',
  currentTopicId: null,
  currentReportId: null,
  isResearchRunning: false,
  runId: null,               // 本次研究的 run_id（断线恢复用）
  userCancelled: false,      // 用户是否主动取消（主动取消不做断线轮询）
  runPollTimer: null,
  researchStartTime: 0,
  navHistory: [],
  currentTopic: null,
  trackingTopicId: null,
  incrementalTopicId: null,
  abortController: null,
  eventSource: null,
  lastReportText: '',
  confirmResolver: null,
  hitlThreadId: null,
  hitlInterruptPoint: null,
  streamingReportText: '',
  reportCollapsed: false,
  activeSteps: [],
  pipelineReached: -1,
  nodeDetails: {},
  tocObserver: null,
  reportRenderTimer: null,
};

const STEPS = [
  'knowledge_lookup', 'planner', 'searcher',
  'validator', 'synthesizer', 'reflector',
  'knowledge_store', 'diff_analyzer', 'change_notifier'
];

const CORE_RESEARCH_STEPS = ['planner', 'searcher', 'validator', 'synthesizer', 'reflector'];
const KNOWLEDGE_RESEARCH_STEPS = ['knowledge_lookup', ...CORE_RESEARCH_STEPS, 'knowledge_store'];

let researchViz = null;

class ResearchEvidenceMap {
  static BOTTOM_INSET = 34;
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.running = false;
    this.startedAt = 0;
    this.phase = 'idle';
    this.branches = [];
    this.sources = [];
    this.gaps = new Set();
    this.synthesisStartedAt = 0;
    this.reflection = null;
    this.hasKnowledge = false;
    this.stored = false;
    this.changeCount = 0;
    this.events = [];
    this.sourceCount = 0;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    this.resize();
    this.frame = this.frame.bind(this);
    requestAnimationFrame(this.frame);
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
  }

  start(question) {
    this.running = true;
    this.startedAt = performance.now();
    this.phase = 'start';
    this.branches = [];
    this.sources = [];
    this.gaps.clear();
    this.synthesisStartedAt = 0;
    this.reflection = null;
    this.hasKnowledge = false;
    this.stored = false;
    this.changeCount = 0;
    this.events = [];
    this.sourceCount = 0;
    this.question = question;
    this.updateMetrics();
    this.renderNarrative('start', {});
    $('executionDetails').open = false;
    requestAnimationFrame(() => this.resize());
  }

  update(node, detail = {}) {
    this.phase = node;
    if (node === 'knowledge_lookup') this.hasKnowledge = !!detail.has_knowledge;

    if (node === 'planner' && detail.sub_questions) {
      this.branches = detail.sub_questions.map((item, index, list) => ({
        question: item.question || `研究分支 ${index + 1}`,
        query: item.search_query || '',
        angle: -Math.PI / 2 + (Math.PI * 2 * index / Math.max(list.length, 1)),
        born: performance.now() + index * 110,
        status: 'open',
      }));
    }

    if (node === 'searcher') {
      this.branches.forEach(branch => { if (branch.status === 'open') branch.status = 'searching'; });
      this.sourceCount += detail.results_count || 0;
      const previews = detail.results_preview || [];
      previews.forEach((item, index) => {
        this.sources.push({
          title: item.title || '来源信号',
          branch: this.branches.length ? index % this.branches.length : 0,
          seed: Math.random(),
          born: performance.now() + index * 70,
          score: item.score || 0,
        });
      });
      if (!previews.length && detail.results_count) {
        const count = Math.min(detail.results_count, 18);
        for (let i = 0; i < count; i++) {
          this.sources.push({ title: '来源信号', branch: this.branches.length ? i % this.branches.length : 0, seed: Math.random(), born: performance.now() + i * 45, score: 0 });
        }
      }
      this.sources = this.sources.slice(-60);
    }

    if (node === 'validator') {
      this.gaps = new Set((detail.gaps || []).map(g => g.question));
      this.branches.forEach(branch => {
        branch.status = this.gaps.has(branch.question) ? 'gap' : 'verified';
      });
    }

    if (node === 'synthesizer') {
      this.synthesisStartedAt = performance.now();
      this.branches.forEach(branch => { if (branch.status !== 'gap') branch.status = 'gathering'; });
    }
    if (node === 'reflector') this.reflection = { pass: !!detail.reflection_pass, round: detail.reflection_round || 0, at: performance.now() };
    if (node === 'knowledge_store') this.stored = true;
    if (node === 'diff_analyzer') this.changeCount = detail.changes_count || 0;
    this.updateMetrics(detail.results_count);
    this.renderNarrative(node, detail);
  }

  complete() {
    this.running = false;
    this.phase = 'complete';
    this.branches.forEach(branch => { if (branch.status !== 'gap') branch.status = 'verified'; });
    this.renderNarrative('complete', {});
  }

  updateMetrics(reportedSourceCount) {
    $('metricBranches').textContent = this.branches.length;
    this.sourceCount = Math.max(this.sourceCount, reportedSourceCount || 0, this.sources.length);
    $('metricSources').textContent = this.sourceCount;
    $('metricGaps').textContent = this.gaps.size;
  }

  eventText(node, detail) {
    if (node === 'start') return '研究问题已进入分析队列';
    if (node === 'knowledge_lookup') return detail.has_knowledge ? '已找到可复用的历史知识' : '未发现历史知识，将执行全新研究';
    if (node === 'planner') return `生成 ${this.branches.length} 个研究分支`;
    if (node === 'searcher') return `接入 ${detail.results_count || 0} 条来源信号`;
    if (node === 'validator') return this.gaps.size ? `发现 ${this.gaps.size} 个证据缺口` : '全部研究分支证据充足';
    if (node === 'synthesizer') return '证据开始向报告结构收束';
    if (node === 'reflector') return detail.reflection_pass ? `第 ${detail.reflection_round || 0} 轮质量检查通过` : `第 ${detail.reflection_round || 0} 轮检查需要补充研究`;
    if (node === 'knowledge_store') return '研究结果已写入知识库';
    if (node === 'diff_analyzer') return `检测到 ${detail.changes_count || 0} 项变化`;
    if (node === 'change_notifier') return detail.notification_sent ? '关键变化通知已发送' : '本轮无需发送通知';
    if (node === 'complete') return '所有证据已收束，研究报告生成完成';
    return '研究状态已更新';
  }

  renderNarrative(node, detail) {
    const meta = NODE_META[node];
    const phaseIndex = appState.activeSteps.indexOf(node);
    $('executionPhaseIndex').textContent = node === 'complete' ? 'OK' : phaseIndex >= 0 ? String(phaseIndex + 1).padStart(2, '0') : '00';
    $('executionPhaseTitle').textContent = node === 'start' ? '建立研究上下文' : node === 'complete' ? '研究完成' : meta ? meta.title : '研究进行中';
    const eventText = this.eventText(node, detail);
    $('executionPhaseSummary').textContent = eventText;

    this.events.push({ text: eventText, at: new Date() });
    this.events = this.events.slice(-4);
    const eventsEl = $('executionEvents');
    eventsEl.replaceChildren(...this.events.map(event => {
      const row = document.createElement('div');
      row.className = 'execution-event';
      const dot = document.createElement('span');
      dot.className = 'execution-event-dot';
      const text = document.createElement('span');
      text.textContent = event.text;
      const time = document.createElement('time');
      time.textContent = event.at.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      row.append(dot, text, time);
      return row;
    }));

    $('executionBranchCount').textContent = this.branches.length;
    const branchesEl = $('executionBranches');
    if (!this.branches.length) {
      const placeholder = document.createElement('div');
      placeholder.className = 'execution-placeholder';
      placeholder.textContent = '规划完成后生成研究分支';
      branchesEl.replaceChildren(placeholder);
      return;
    }

    const statusLabel = { open: '待搜索', searching: '搜索中', gap: '需补充', verified: '已验证', gathering: '汇总中' };
    branchesEl.replaceChildren(...this.branches.map((branch, index) => {
      const row = document.createElement('div');
      row.className = `execution-branch is-${branch.status}`;
      const number = document.createElement('span');
      number.className = 'execution-branch-number';
      number.textContent = String(index + 1).padStart(2, '0');
      const copy = document.createElement('span');
      copy.className = 'execution-branch-copy';
      const question = document.createElement('strong');
      question.textContent = branch.question;
      const query = document.createElement('small');
      query.textContent = branch.query || '等待搜索词';
      copy.append(question, query);
      const status = document.createElement('span');
      status.className = 'execution-branch-status';
      status.textContent = statusLabel[branch.status] || '处理中';
      row.append(number, copy, status);
      return row;
    }));
  }

  /** 舞台底部 34px 是指标状态条，几何中心相应上移，避免图形贴住状态条 */
  get centerY() { return (this.height - ResearchEvidenceMap.BOTTOM_INSET) / 2; }

  pointForBranch(branch, index) {
    const cx = this.width / 2;
    const cy = this.centerY;
    const rx = Math.min(this.width * (this.width < 600 ? .29 : .36), 390);
    const ry = Math.min((this.height - ResearchEvidenceMap.BOTTOM_INSET) * .38, 168);
    const angle = branch ? branch.angle : -Math.PI / 2 + index;
    return { x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry };
  }

  drawHex(x, y, radius, fill, stroke, rotation = 0) {
    const ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = rotation + Math.PI / 3 * i;
      const px = x + Math.cos(angle) * radius;
      const py = y + Math.sin(angle) * radius;
      if (!i) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.stroke(); }
  }

  frame(now) {
    this.draw(now);
    requestAnimationFrame(this.frame);
  }

  draw(now) {
    if (!this.width || !this.height) return;
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    const cx = w / 2;
    const cy = this.centerY;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = '#0d1311';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(77, 133, 121, .08)';
    ctx.lineWidth = 1;
    for (let x = (now * .006) % 32; x < w; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

    const radarRadius = Math.min(w, h - ResearchEvidenceMap.BOTTOM_INSET) * .42;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.strokeStyle = 'rgba(60, 139, 122, .12)';
    [0.35, 0.68, 1].forEach(scale => { ctx.beginPath(); ctx.ellipse(0, 0, radarRadius * scale, radarRadius * scale * .58, 0, 0, Math.PI * 2); ctx.stroke(); });
    ctx.rotate((now * .00012) % (Math.PI * 2));
    const sweep = ctx.createLinearGradient(0, 0, radarRadius, 0);
    sweep.addColorStop(0, 'rgba(65, 201, 178, .16)');
    sweep.addColorStop(1, 'rgba(65, 201, 178, 0)');
    ctx.fillStyle = sweep;
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, radarRadius, -.18, .18); ctx.closePath(); ctx.fill();
    ctx.restore();

    if (this.hasKnowledge) {
      ctx.strokeStyle = 'rgba(223, 77, 63, .22)';
      for (let i = 0; i < 3; i++) {
        ctx.beginPath();
        ctx.arc(cx, cy, 76 + i * 11 + Math.sin(now * .002 + i) * 3, Math.PI * .72, Math.PI * 1.28);
        ctx.stroke();
      }
    }

    this.branches.forEach((branch, index) => {
      const target = this.pointForBranch(branch, index);
      const progress = Math.max(0, Math.min(1, (now - branch.born) / 650));
      const x = cx + (target.x - cx) * progress;
      const y = cy + (target.y - cy) * progress;
      const gap = branch.status === 'gap';
      const verified = branch.status === 'verified';
      const color = gap ? '#d49a38' : verified ? '#55bb86' : '#43bcae';

      ctx.save();
      ctx.strokeStyle = gap ? 'rgba(212,154,56,.58)' : verified ? 'rgba(85,187,134,.48)' : 'rgba(67,188,174,.42)';
      ctx.lineWidth = gap ? 1.5 : 1;
      if (gap) ctx.setLineDash([5, 7]);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
      ctx.setLineDash([]);
      this.drawHex(x, y, gap ? 9 : 7, '#13201c', color, Math.PI / 6);
      if (gap) {
        ctx.strokeStyle = 'rgba(212,154,56,.32)';
        ctx.beginPath(); ctx.arc(x, y, 15 + Math.sin(now * .006) * 3, 0, Math.PI * 2); ctx.stroke();
      }

      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = color;
      ctx.font = '700 8px ui-monospace, monospace';
      ctx.fillText(String(index + 1), x, y + .5);
      ctx.restore();
    });

    this.sources.forEach((source, index) => {
      const branch = this.branches[source.branch];
      if (!branch) return;
      const target = this.pointForBranch(branch, source.branch);
      const angle = branch.angle + (source.seed - .5) * .75;
      const outer = Math.max(w, h) * .62;
      const origin = { x: cx + Math.cos(angle) * outer, y: cy + Math.sin(angle) * outer * .55 };
      let t = ((now - source.born) * .00024 + source.seed) % 1;
      let tx = target.x;
      let ty = target.y;
      if (this.synthesisStartedAt) {
        const gather = Math.min(1, (now - this.synthesisStartedAt) / 2400);
        tx += (cx - tx) * gather;
        ty += (cy - ty) * gather;
        t = Math.max(t, gather);
      }
      const x = origin.x + (tx - origin.x) * t;
      const y = origin.y + (ty - origin.y) * t;
      ctx.fillStyle = source.score > .6 ? '#72dbc8' : 'rgba(114,219,200,.65)';
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
      if (index < 14) {
        ctx.strokeStyle = 'rgba(114,219,200,.08)';
        ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(x, y); ctx.stroke();
      }
    });

    const pulse = 1 + Math.sin(now * .003) * .06;
    ctx.save();
    ctx.shadowColor = this.reflection && !this.reflection.pass ? 'rgba(212,154,56,.7)' : 'rgba(67,188,174,.62)';
    ctx.shadowBlur = this.running ? 22 : 10;
    this.drawHex(cx, cy, 72 * pulse, 'rgba(18,29,25,.96)', this.reflection && !this.reflection.pass ? '#d49a38' : '#43bcae', Math.PI / 6);
    ctx.restore();
    this.drawHex(cx, cy, 60, 'rgba(32,49,43,.8)', 'rgba(119,219,195,.24)', Math.PI / 6);

    if (this.reflection) {
      const age = now - this.reflection.at;
      ctx.save();
      ctx.strokeStyle = this.reflection.pass ? 'rgba(85,187,134,.75)' : 'rgba(212,154,56,.75)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 8]);
      ctx.beginPath(); ctx.arc(cx, cy, 82 + (age * .025) % 42, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();
    }

    if (this.stored) {
      ctx.fillStyle = 'rgba(85,187,134,.85)';
      ctx.fillRect(cx - 16, cy + 56, 32, 2);
    }
    if (this.changeCount) {
      ctx.fillStyle = '#d49a38';
      for (let i = 0; i < Math.min(this.changeCount, 8); i++) ctx.fillRect(cx - 28 + i * 8, cy - 66, 4, 4);
    }

    if (this.startedAt) {
      const seconds = Math.max(0, Math.floor((now - this.startedAt) / 1000));
      $('observatoryTime').textContent = String(Math.floor(seconds / 60)).padStart(2, '0') + ':' + String(seconds % 60).padStart(2, '0');
    }
  }
}

/* ═══════════════════════════════════════════════
   Toast Notification System
   ═══════════════════════════════════════════════ */
const TOAST_ICONS = {
  success: '✓',
  error: '✕',
  warning: '⚠',
  info: 'ℹ',
};

function toast(message, type = 'info', title = '', duration = 4000) {
  const container = $('toastContainer');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `
    <span class="toast-icon" aria-hidden="true">${TOAST_ICONS[type]}</span>
    <div class="toast-body">
      ${title ? `<div class="toast-title">${esc(title)}</div>` : ''}
      <div class="toast-message">${esc(message)}</div>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()" aria-label="关闭通知">✕</button>
    <div class="toast-progress" style="animation-duration:${duration}ms"></div>
  `;
  container.appendChild(el);

  setTimeout(() => {
    el.classList.add('removing');
    setTimeout(() => el.remove(), 250);
  }, duration);
}

/* ═══════════════════════════════════════════════
   Confirm Dialog
   ═══════════════════════════════════════════════ */
function showConfirm(title, message) {
  return new Promise(resolve => {
    appState.confirmResolver = resolve;
    $('confirmTitle').textContent = title;
    $('confirmMessage').textContent = message;
    $('confirmOverlay').classList.add('visible');
    $('confirmOk').focus();
  });
}

function resolveConfirm(result) {
  $('confirmOverlay').classList.remove('visible');
  if (appState.confirmResolver) {
    appState.confirmResolver(result);
    appState.confirmResolver = null;
  }
}

/* ═══════════════════════════════════════════════
   Clipboard Helper
   ═══════════════════════════════════════════════ */
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('已复制到剪贴板', 'success');
  } catch {
    toast('复制失败', 'error');
  }
}

/* ═══════════════════════════════════════════════
   Navigation
   ═══════════════════════════════════════════════ */
function navigate(page, opts = {}) {
  if (page === 'topic-detail' && opts.topicId) {
    appState.currentTopicId = opts.topicId;
    appState.navHistory.push(appState.currentPage);
  } else if (page === 'report-detail' && opts.reportId) {
    appState.currentReportId = opts.reportId;
    appState.navHistory.push(appState.currentPage);
  }

  appState.currentPage = page;

  // Hide all pages
  $$('.page').forEach(p => p.style.display = 'none');
  const pageEl = $('page-' + page);
  if (pageEl) pageEl.style.display = '';

  // Update nav active state
  $$('.nav-item').forEach(n => n.classList.remove('active'));
  const navMap = {
    'dashboard': 'dashboard',
    'research': 'research',
    'topics': 'topics',
    'topic-detail': 'topics',
    'tracking': 'tracking',
    'report-detail': 'topics',
  };
  const activeNav = document.querySelector(`.nav-item[data-page="${navMap[page] || page}"]`);
  if (activeNav) activeNav.classList.add('active');

  // Load page data
  if (page === 'dashboard') loadDashboard();
  if (page === 'topics') loadTopics();
  if (page === 'topic-detail') loadTopicDetail(opts.topicId);
  if (page === 'tracking') loadTrackingPage();
  if (page === 'report-detail') loadReportDetail(opts.reportId);
  if (page === 'research') refreshResearchModes();

  // Update breadcrumb
  updateBreadcrumb();

  // Close mobile sidebar
  $('sidebar').classList.remove('open');
  $('sidebarOverlay').classList.remove('open');
}

function goBack() {
  navigate(appState.navHistory.pop() || 'topics');
}

function toggleSidebar() {
  $('sidebar').classList.toggle('open');
  $('sidebarOverlay').classList.toggle('open');
}

function showModal(n) {
  const overlay = $('modal-' + n);
  overlay.classList.add('visible');
  // Focus first input
  const firstInput = overlay.querySelector('.input, textarea');
  if (firstInput) setTimeout(() => firstInput.focus(), 100);
}

function hideModal(n) {
  $('modal-' + n).classList.remove('visible');
}

/* ═══════════════════════════════════════════════
   Breadcrumb
   ═══════════════════════════════════════════════ */
function updateBreadcrumb() {
  const bc = $('breadcrumb');
  const page = appState.currentPage;
  const crumbs = [];

  if (page === 'dashboard') {
    bc.innerHTML = '';
    return;
  }

  crumbs.push({ label: '概览', page: 'dashboard' });

  if (page === 'research') {
    crumbs.push({ label: '新建研究', current: true });
  } else if (page === 'topics') {
    crumbs.push({ label: '研究主题', current: true });
  } else if (page === 'topic-detail') {
    crumbs.push({ label: '研究主题', page: 'topics' });
    if (appState.currentTopic) {
      crumbs.push({ label: appState.currentTopic.name, current: true });
    } else {
      crumbs.push({ label: '主题详情', current: true });
    }
  } else if (page === 'report-detail') {
    crumbs.push({ label: '研究主题', page: 'topics' });
    crumbs.push({ label: '报告详情', current: true });
  } else if (page === 'tracking') {
    crumbs.push({ label: '追踪任务', current: true });
  }

  bc.innerHTML = crumbs.map((c, i) => {
    if (c.current) {
      return `<span class="breadcrumb-current">${esc(c.label)}</span>`;
    }
    return `<a class="breadcrumb-link" onclick="navigate('${c.page}')">${esc(c.label)}</a>` +
      (i < crumbs.length - 1 ? '<span class="breadcrumb-sep">›</span>' : '');
  }).join('');
}

/* ═══════════════════════════════════════════════
   Tag Input
   ═══════════════════════════════════════════════ */
function initTagInput(wId, iId) {
  const w = $(wId);
  const i = $(iId);
  i.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      const v = i.value.trim();
      if (!v) return;
      const t = document.createElement('span');
      t.className = 'tag';
      t.innerHTML = `${esc(v)} <span class="tag-remove" onclick="this.parentElement.remove()">✕</span>`;
      w.insertBefore(t, i);
      i.value = '';
    }
  });
}

function getTags(wId) {
  return Array.from($(wId).querySelectorAll('.tag'))
    .map(t => t.textContent.replace('✕', '').trim());
}

function setTags(wId, iId, tags) {
  const w = $(wId);
  const i = $(iId);
  w.querySelectorAll('.tag').forEach(t => t.remove());
  tags.forEach(v => {
    const t = document.createElement('span');
    t.className = 'tag';
    t.innerHTML = `${esc(v)} <span class="tag-remove" onclick="this.parentElement.remove()">✕</span>`;
    w.insertBefore(t, i);
  });
}

initTagInput('topicKwWrap', 'topicKwInput');
initTagInput('trackKwWrap', 'trackKwInput');

/* ═══════════════════════════════════════════════
   API
   ═══════════════════════════════════════════════ */
async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(e.error || r.statusText);
  }
  return r.json();
}

