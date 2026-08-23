/* ═══════════════════════════════════════════════
   Topics
   ═══════════════════════════════════════════════ */
async function loadTopics() {
  const l = $('topicsList');
  l.innerHTML = '<div class="skeleton-grid"><div class="skeleton skeleton-card"></div><div class="skeleton skeleton-card"></div><div class="skeleton skeleton-card"></div></div>';

  try {
    const topics = await api('/topics');
    $('topicCount').textContent = topics.length;

    if (!topics.length) {
      l.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon" aria-hidden="true">◫</div>
          <div class="empty-text">还没有研究主题，创建一个开始吧</div>
          <button class="btn btn-primary" onclick="showModal('createTopic')">+ 新建主题</button>
        </div>
      `;
      return;
    }

    l.innerHTML = '<div class="topic-grid">' + topics.map(t => {
      const kw = (t.tracking_keywords || []).map(k => `<span class="badge badge-muted">${esc(k)}</span>`).join('');
      const tb = t.tracking_enabled ? '<span class="badge badge-green">追踪中</span>' : '';
      return `
        <div class="card topic-card" onclick="navigate('topic-detail',{topicId:'${t.id}'})">
          <div class="topic-actions">
            <button class="btn btn-ghost btn-sm btn-danger" onclick="event.stopPropagation();deleteTopic('${t.id}')">删除</button>
          </div>
          <div class="topic-name">${esc(t.name)}</div>
          <div class="topic-desc">${esc(t.description || '暂无描述')}</div>
          <div class="topic-meta">${kw}${tb}<span class="badge badge-muted">${new Date(t.created_at).toLocaleDateString('zh-CN')}</span></div>
        </div>
      `;
    }).join('') + '</div>';
  } catch (e) {
    l.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">⚠</div>
        <div class="empty-text">加载失败: ${esc(e.message)}</div>
      </div>
    `;
  }
}

async function createTopic() {
  const n = $('topicName').value.trim();
  if (!n) return;

  const d = $('topicDesc').value.trim();
  const k = getTags('topicKwWrap');

  try {
    await api('/topics', {
      method: 'POST',
      body: { name: n, description: d, tracking_keywords: k },
    });
    hideModal('createTopic');
    $('topicName').value = '';
    $('topicDesc').value = '';
    setTags('topicKwWrap', 'topicKwInput', []);
    loadTopics();
    toast('主题创建成功', 'success');
  } catch (e) {
    toast('创建失败: ' + e.message, 'error');
  }
}

async function deleteTopic(id) {
  const confirmed = await showConfirm('删除主题', '确定删除此主题及其所有报告？此操作不可撤销。');
  if (!confirmed) return;

  try {
    await api('/topics/' + id, { method: 'DELETE' });
    loadTopics();
    toast('主题已删除', 'success');
  } catch (e) {
    toast(e.message, 'error');
  }
}

/* ═══════════════════════════════════════════════
   Topic Detail
   ═══════════════════════════════════════════════ */
async function loadTopicDetail(topicId) {
  const h = $('topicDetailHeader');

  try {
    const t = await api('/topics/' + topicId);
    appState.currentTopic = t;

    h.innerHTML = `
      <div class="topic-icon" aria-hidden="true">◫</div>
      <div class="topic-info">
        <h2>${esc(t.name)}</h2>
        <p>${esc(t.description || '暂无描述')}</p>
      </div>
      <div class="topic-actions-bar">
        <button class="btn btn-sm" onclick="startResearchForTopic('${topicId}')">◈ 新研究</button>
        <button class="btn btn-sm" onclick="showIncrementalModal('${topicId}')">📈 增量</button>
        <button class="btn btn-sm" onclick="openTrackingConfig('${topicId}')">◉ 追踪</button>
        <button class="btn btn-sm btn-danger" onclick="deleteTopicFromDetail('${topicId}')">删除</button>
      </div>
    `;

    loadTopicReports(topicId);
    loadTopicTrackingConfig(topicId);
    loadTopicTrackingLogs(topicId);
    updateBreadcrumb();
  } catch (e) {
    h.innerHTML = `<p class="text-red">加载失败: ${esc(e.message)}</p>`;
  }
}

function switchTab(el, name) {
  el.parentElement.querySelectorAll('.tab').forEach(t => {
    t.classList.remove('active');
    t.setAttribute('aria-selected', 'false');
    t.setAttribute('tabindex', '-1');
  });
  el.classList.add('active');
  el.setAttribute('aria-selected', 'true');
  el.setAttribute('tabindex', '0');

  $$('.tab-panel').forEach(p => p.classList.remove('active'));
  $('panel-' + name).classList.add('active');
}

async function loadTopicReports(topicId) {
  const p = $('panel-reports');
  p.innerHTML = '<div class="skeleton" style="height:200px;border-radius:var(--r2)"></div>';

  try {
    const reports = await api('/topics/' + topicId + '/reports');
    if (!reports.length) {
      p.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon" aria-hidden="true">▯</div>
          <div class="empty-text">暂无研究报告</div>
        </div>
      `;
      return;
    }

    p.innerHTML = '<div class="timeline">' + reports.map(r => {
      const d = new Date(r.created_at).toLocaleString('zh-CN');
      const b = [
        r.is_incremental ? '<span class="badge badge-cyan">增量</span>' : '<span class="badge badge-accent">全新</span>',
        r.confidence ? `<span class="badge badge-${r.confidence === '高' ? 'green' : r.confidence === '中' ? 'amber' : 'red'}">${r.confidence}</span>` : '',
        `<span class="badge badge-muted">${r.search_results_count || 0} src</span>`,
      ].filter(Boolean).join('');
      const pv = (r.report || '').slice(0, 150).replace(/[#*`]/g, '');
      return `
        <div class="timeline-item${r.is_incremental ? ' incremental' : ''}">
          <div class="tl-date">${d}</div>
          <div class="tl-title" onclick="navigate('report-detail',{reportId:'${r.id}'})">${esc(r.question)}</div>
          <div class="tl-badges">${b}</div>
          <div class="tl-preview">${esc(pv)}…</div>
        </div>
      `;
    }).join('') + '</div>';
  } catch (e) {
    p.innerHTML = '<p class="text-muted">加载失败</p>';
  }
}

async function loadTopicTrackingConfig(topicId) {
  const p = $('panel-tracking');
  const t = appState.currentTopic;
  if (!t) return;

  const en = t.tracking_enabled;
  const cr = t.tracking_cron || '';
  const kw = t.tracking_keywords || [];

  p.innerHTML = `
    <div class="card">
      <div class="card-body">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
          <div>
            <h3 style="font-size:.98rem;font-weight:700">${'自动追踪'}</h3>
            <p class="text-muted" style="font-size:.8rem;margin-top:4px">定期自动搜索领域变化</p>
          </div>
          <span class="badge ${en ? 'badge-green' : 'badge-muted'}">${en ? '已启用' : '未启用'}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
          <div>
            <div class="form-label">Cron</div>
            <div class="text-secondary font-mono" style="font-size:.86rem">${esc(cr || '未配置')}</div>
          </div>
          <div>
            <div class="form-label">关键词</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap">
              ${kw.map(k => `<span class="badge badge-accent">${esc(k)}</span>`).join('') || '<span class="text-muted" style="font-size:.82rem">未配置</span>'}
            </div>
          </div>
        </div>
        <button class="btn btn-sm" onclick="openTrackingConfig('${topicId}')">✏ 编辑</button>
        <button class="btn btn-sm" style="margin-left:8px" onclick="runTrackingNow('${topicId}', this)">▶ 立即追踪</button>
      </div>
    </div>
  `;
}

async function loadTopicTrackingLogs(topicId) {
  const p = $('panel-changes');

  try {
    const logs = await api('/topics/' + topicId + '/tracking-logs');
    if (!logs.length) {
      p.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon" aria-hidden="true">◉</div>
          <div class="empty-text">暂无追踪记录</div>
        </div>
      `;
      return;
    }

    let h = '<div class="card"><div style="overflow:hidden">';
    for (const l of logs) {
      const d = new Date(l.triggered_at).toLocaleString('zh-CN');
      const sc = l.status === 'completed' ? 'completed' : l.status === 'failed' ? 'failed' : 'running';
      h += `
        <div class="tracking-log-item" onclick="loadChanges('${l.id}')">
          <div class="log-status ${sc}"></div>
          <div class="log-info">
            <div class="log-time">${d}</div>
            <div class="log-summary">${esc(l.change_summary || l.status)}</div>
          </div>
          <div class="log-changes">${l.changes_detected || 0} 项变化</div>
        </div>
      `;
    }
    h += '</div></div>';
    p.innerHTML = h;
  } catch (e) {
    p.innerHTML = '<p class="text-muted">加载失败</p>';
  }
}

async function loadChanges(logId) {
  try {
    const changes = await api('/tracking-logs/' + logId + '/changes');
    if (!changes.length) return;

    let h = '<div class="change-list">';
    for (const c of changes) {
      const tm = {
        new_info: ['badge-green', '新增'],
        contradiction: ['badge-red', '矛盾'],
        update: ['badge-amber', '更新'],
        removal: ['badge-muted', '删除'],
      };
      const [cl, lb] = tm[c.change_type] || ['badge-muted', c.change_type];
      h += `
        <div class="change-item">
          <div class="change-header">
            <span class="badge ${cl}">${lb}</span>
            ${c.significance ? `<span class="badge badge-${c.significance === 'high' ? 'pink' : c.significance === 'medium' ? 'amber' : 'muted'}">${c.significance}</span>` : ''}
          </div>
          <div class="change-desc">${esc(c.description)}</div>
          ${c.old_content || c.new_content ? `
            <div class="change-diff">
              ${c.old_content ? `<span class="diff-old">${esc(c.old_content.slice(0, 80))}</span>` : ''}
              ${c.new_content ? `<span class="diff-new">${esc(c.new_content.slice(0, 80))}</span>` : ''}
            </div>
          ` : ''}
        </div>
      `;
    }
    h += '</div>';
    $('panel-changes').innerHTML = `
      <h3 class="section-title">变更详情</h3>
      ${h}
      <button class="btn btn-sm btn-ghost" onclick="loadTopicTrackingLogs('${appState.currentTopicId}')" style="margin-top:10px">← 返回列表</button>
    `;
  } catch (e) {
    toast('加载变更失败', 'error');
  }
}

/* ═══════════════════════════════════════════════
   Tracking Config
   ═══════════════════════════════════════════════ */
function openTrackingConfig(topicId) {
  const t = appState.currentTopic;
  if (t) {
    setTags('trackKwWrap', 'trackKwInput', t.tracking_keywords || []);
    $('trackCron').value = t.tracking_cron || '';
    $('trackEnabled').checked = !!t.tracking_enabled;
  }
  appState.trackingTopicId = topicId;
  showModal('trackingConfig');
}

async function saveTrackingConfig() {
  const tid = appState.trackingTopicId;
  const kw = getTags('trackKwWrap');
  // 清洗 cron：只取前 5 段，忽略用户可能附加的注释（如 "0 9 * * * == 每天9点"）
  let cr = $('trackCron').value.trim().split(/\s+/).slice(0, 5).join(' ');
  const en = $('trackEnabled').checked;

  if (cr && cr.split(/\s+/).length !== 5) {
    toast('Cron 表达式格式错误，应为 5 段：分 时 日 月 周', 'warning');
    return;
  }

  try {
    const saved = await api('/topics/' + tid, {
      method: 'PUT',
      body: { tracking_keywords: kw, tracking_cron: cr, tracking_enabled: en },
    });
    hideModal('trackingConfig');
    loadTopicDetail(tid);
    // 后端返回 tracking_scheduled 表示定时任务是否真的注册进调度器
    if (saved.tracking_warning) {
      toast(saved.tracking_warning, 'warning');
    } else if (en && cr && !saved.tracking_scheduled) {
      toast('配置已保存，但定时任务未注册，请检查服务端日志', 'warning');
    } else if (en && cr) {
      toast('追踪配置已保存，定时任务已生效', 'success');
    } else {
      toast('追踪配置已保存', 'success');
    }
  } catch (e) {
    toast('保存失败: ' + e.message, 'error');
  }
}

async function testNotification() {
  try {
    const r = await api('/tracking/test-notification', { method: 'POST' });
    if (r.sent) {
      toast('测试通知已发送，请检查接收端', 'success');
    } else {
      toast('通知未发送：请检查通知配置（环境变量）', 'warning');
    }
  } catch (e) {
    toast('测试通知失败: ' + e.message, 'error');
  }
}

async function runTrackingNow(topicId, btn) {
  const origText = btn.textContent;
  btn.disabled = true;
  btn.textContent = '⏳ 追踪中…';
  try {
    const r = await api('/tracking/run', {
      method: 'POST',
      body: { topic_id: topicId },
    });
    toast(`追踪完成：检测到 ${r.changes_detected} 项变化${r.notification_sent ? '，已发送通知' : ''}`, 'info');
    loadTopicDetail(topicId);
  } catch (e) {
    toast('追踪失败: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = origText;
  }
}

/* ═══════════════════════════════════════════════
   Incremental Research
   ═══════════════════════════════════════════════ */
function startResearchForTopic(topicId) {
  navigate('research');
  $('researchMode').value = topicId;
  $('question').focus();
}

async function showIncrementalModal(topicId) {
  try {
    const s = await api('/topics/' + topicId + '/reports');
    const l = s && s[0];
    $('incrementalContext').innerHTML = l
      ? `<strong>最近研究：</strong>${esc(l.question)}<br><span class="text-muted">${new Date(l.created_at).toLocaleString('zh-CN')} · ${l.is_incremental ? '增量' : '全新'} · ${l.confidence || '未知'}</span>`
      : '暂无历史研究记录';
    $('incrementalQuestion').value = '';
    appState.incrementalTopicId = topicId;
    showModal('incrementalResearch');
  } catch (e) {
    toast('加载历史研究失败', 'error');
  }
}

function startIncrementalResearch() {
  const q = $('incrementalQuestion').value.trim();
  if (!q) {
    $('incrementalQuestion').focus();
    return;
  }
  hideModal('incrementalResearch');
  navigate('research');
  $('question').value = q;
  $('researchMode').value = appState.incrementalTopicId;
  startResearch(appState.incrementalTopicId, true);
}



/* ═══════════════════════════════════════════════
   Delete Topic From Detail
   ═══════════════════════════════════════════════ */
async function deleteTopicFromDetail(id) {
  const confirmed = await showConfirm('删除主题', '确定删除此主题及其所有报告？此操作不可撤销。');
  if (!confirmed) return;

  try {
    await api('/topics/' + id, { method: 'DELETE' });
    toast('主题已删除', 'success');
    navigate('topics');
  } catch (e) {
    toast(e.message, 'error');
  }
}

/* ═══════════════════════════════════════════════
   Delete Report
   ═══════════════════════════════════════════════ */
async function deleteReport() {
  const id = appState.currentReportId;
  if (!id) return;
  const confirmed = await showConfirm('删除报告', '确定删除此报告？此操作不可撤销。');
  if (!confirmed) return;

  try {
    await api('/reports/' + id, { method: 'DELETE' });
    toast('报告已删除', 'success');
    goBack();
  } catch (e) {
    toast('删除失败: ' + e.message, 'error');
  }
}

/* ═══════════════════════════════════════════════
   Tracking Page
   ═══════════════════════════════════════════════ */
async function loadTrackingPage() {
  const c = $('trackingList');
  c.innerHTML = '<div class="skeleton" style="height:200px;border-radius:var(--r2)"></div>';

  try {
    const [topics, jobs] = await Promise.all([
      api('/topics'),
      api('/tracking/jobs'),
    ]);

    $('trackingCount').textContent = jobs.length;
    const tr = topics.filter(t => t.tracking_enabled);

    if (!tr.length && !jobs.length) {
      c.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon" aria-hidden="true">◉</div>
          <div class="empty-text">暂无追踪任务。在主题详情中启用追踪即可。</div>
        </div>
      `;
      return;
    }

    let h = '';

    if (tr.length) {
      h += '<h3 class="section-title">正在追踪</h3><div class="topic-grid" style="margin-bottom:28px">';
      for (const t of tr) {
        const kw = (t.tracking_keywords || []).map(k => `<span class="badge badge-accent">${esc(k)}</span>`).join('');
        h += `
          <div class="card topic-card" onclick="navigate('topic-detail',{topicId:'${t.id}'})">
            <div class="topic-name">${esc(t.name)}</div>
            <div style="margin:6px 0;display:flex;gap:5px;flex-wrap:wrap">${kw}</div>
            <div style="display:flex;gap:6px;align-items:center">
              <span class="badge badge-green">追踪中</span>
              <span class="badge badge-muted font-mono">${esc(t.tracking_cron || '')}</span>
            </div>
          </div>
        `;
      }
      h += '</div>';
    }

    if (jobs.length) {
      h += '<h3 class="section-title">调度任务</h3><div class="card"><div style="overflow:hidden">';
      for (const j of jobs) {
        const jobTopic = topics.find(t => t.id === j.topic_id);
        h += `
          <div class="tracking-log-item">
            <div class="log-status completed"></div>
            <div class="log-info">
              <div class="log-time">${esc(j.topic_name || (jobTopic && jobTopic.name) || j.topic_id || '未知')}</div>
              <div class="log-summary">下次运行: ${esc(j.next_run || '未安排')}</div>
            </div>
          </div>
        `;
      }
      h += '</div></div>';
    }

    c.innerHTML = h;
  } catch (e) {
    c.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">⚠</div>
        <div class="empty-text">加载失败: ${esc(e.message)}</div>
      </div>
    `;
  }
}

