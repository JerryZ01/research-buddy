/* ═══════════════════════════════════════════════
   Init + 断线恢复（localStorage run_id）
   ═══════════════════════════════════════════════ */
const LAST_RUN_KEY = 'rb:lastRun';

function saveLastRun(runId) {
  try { localStorage.setItem(LAST_RUN_KEY, JSON.stringify({ id: runId, at: Date.now() })); } catch (e) {}
}

function clearLastRun() {
  try { localStorage.removeItem(LAST_RUN_KEY); } catch (e) {}
}

function getLastRun() {
  try {
    const v = JSON.parse(localStorage.getItem(LAST_RUN_KEY) || 'null');
    return v && v.id ? v : null;
  } catch (e) { return null; }
}

/** 页面加载/刷新后：上次研究若未正常结束且记录仍在，自动恢复结果 */
async function restoreSavedRun() {
  const saved = getLastRun();
  if (!saved || appState.isResearchRunning) return;
  try {
    const res = await fetch('/research/run/' + saved.id);
    if (res.status === 404) { clearLastRun(); return; }
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === 'done') {
      clearLastRun();
      if (data.result) {
        showReport(data.result);
        setAllDone();
        addLog('<span class="text-green">✅ 上次中断的研究已完成，结果已恢复</span>');
        toast('上次研究已完成，结果已恢复', 'success');
      }
    } else if (data.status === 'error') {
      clearLastRun();
      addLog(`<span class="text-amber">上次研究已失败: ${esc(data.error || '未知')}</span>`);
    } else {
      // 仍在 running：后台继续跑，轮询恢复
      appState.runId = saved.id;
      addLog('<span class="text-amber">⚠ 上次研究仍在后台进行，正在自动恢复…</span>');
      $('foundryStatus').textContent = '后台研究中，正在恢复…';
      startPollRun();
    }
  } catch (e) { /* 网络问题，下次加载再试 */ }
}

async function init() {
  if (window.lucide) {
    window.lucide.createIcons();
  } else {
    window.addEventListener('load', () => window.lucide && window.lucide.createIcons(), { once: true });
  }

  researchViz = new ResearchEvidenceMap($('researchCanvas'));

  // Load dashboard as landing page
  loadDashboard();

  // Populate research modes
  refreshResearchModes();

  // 断线/刷新恢复上次未完成的研究
  restoreSavedRun();
}

init();
