/* ═══════════════════════════════════════════════
   Research
   ═══════════════════════════════════════════════ */
function elapsed() {
  const s = Math.floor((Date.now() - appState.researchStartTime) / 1000);
  return s < 60 ? s + 's' : Math.floor(s / 60) + 'm ' + (s % 60) + 's';
}

function addLog(html) {
  const c = $('logContent');
  const l = document.createElement('div');
  l.className = 't-line';
  l.innerHTML = `<span class="t-prompt" aria-hidden="true">▸</span><span class="t-text">${html}</span><span class="t-time">${elapsed()}</span>`;
  c.appendChild(l);
  const body = $('progressLog').querySelector('.terminal-body');
  body.scrollTop = body.scrollHeight;
}

function setStep(node, detail = {}) {
  const flow = appState.activeSteps.length ? appState.activeSteps : STEPS;
  const idx = flow.indexOf(node);
  if (idx < 0) return;
  const meta = NODE_META[node];
  if ($('foundryStatus')) $('foundryStatus').textContent = meta ? meta.status : node;
  renderPipelineStrip(flow, node);
  if (researchViz) researchViz.update(node, detail);
}

/**
 * 画布底部状态条右侧的节点流水线：一眼看出当前停在哪一步。
 * 补搜/重写会让节点重复经过，所以「已完成」只按首次到达的最远位置算。
 */
function renderPipelineStrip(flow, activeNode) {
  const host = $('observatoryPipeline');
  if (!host) return;
  const activeIdx = flow.indexOf(activeNode);
  appState.pipelineReached = Math.max(appState.pipelineReached || 0, activeIdx);
  const frag = document.createDocumentFragment();
  flow.forEach((step, i) => {
    if (i > 0) {
      const sep = document.createElement('i');
      sep.className = 'om-sep';
      sep.textContent = '›';
      frag.appendChild(sep);
    }
    const el = document.createElement('i');
    el.className = 'om-step' + (i === activeIdx ? ' is-active' : (i <= appState.pipelineReached ? ' is-done' : ''));
    el.textContent = (NODE_META[step] || {}).title || step;
    frag.appendChild(el);
  });
  host.replaceChildren(frag);
}

function setAllDone() {
  if (researchViz) researchViz.complete();
  if ($('foundryStatus')) $('foundryStatus').textContent = '证据已收束，报告完成';
  const flow = appState.activeSteps.length ? appState.activeSteps : STEPS;
  appState.pipelineReached = flow.length - 1;
  renderPipelineStrip(flow, flow[flow.length - 1]);
}

/* ── AI 问题润色（输入框辅助） ─────────────────────── */
let _lastOriginalQuestion = '';

async function refineQuestion() {
  const q = $('question').value.trim();
  if (!q) {
    toast('先输入研究问题再润色', 'warning');
    $('question').focus();
    return;
  }
  const btn = $('refineBtn');
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = '润色中…';
  try {
    const res = await fetch('/research/refine-question', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || '润色失败');
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    if (!candidates.length) throw new Error('没有返回候选');
    _lastOriginalQuestion = q;
    renderRefineCandidates(candidates);
  } catch (e) {
    toast('润色失败：' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

function renderRefineCandidates(candidates) {
  const list = $('refineCandidateList');
  list.replaceChildren();
  candidates.forEach((c, i) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'refine-candidate';
    card.onclick = () => useRefinedCandidate(i);
    const qEl = document.createElement('div');
    qEl.className = 'rc-q';
    qEl.textContent = c.refined_question;
    const meta = document.createElement('div');
    meta.className = 'rc-meta';
    const tags = [];
    if (c.style) tags.push(c.style);
    if (c.intent) tags.push('意图：' + c.intent);
    if (Array.isArray(c.tips) && c.tips.length) tags.push(c.tips.join('；'));
    meta.textContent = tags.join(' · ');
    card.append(qEl, meta);
    list.appendChild(card);
  });
  $('refinePanel').style.display = 'block';
}

function useRefinedCandidate(index) {
  const cards = $('refineCandidateList').children;
  if (!cards[index]) return;
  const qEl = cards[index].querySelector('.rc-q');
  if (!qEl || !qEl.textContent) return;
  $('question').value = qEl.textContent;
  hideRefinePanel();
  toast('已采用润色结果', 'success');
}

function regenRefine() {
  hideRefinePanel();
  refineQuestion();
}

function hideRefinePanel() {
  $('refinePanel').style.display = 'none';
}

function undoRefine() {
  if (!_lastOriginalQuestion) return;
  $('question').value = _lastOriginalQuestion;
  hideRefinePanel();
  _lastOriginalQuestion = '';
  toast('已还原原问题', 'info');
}

function startResearch(topicId, isIncremental) {
  const question = $('question').value.trim();
  if (!question) {
    $('question').focus();
    return;
  }
  if (appState.isResearchRunning) return;

  const mode = $('researchMode').value;
  const useTopic = topicId || (mode !== 'normal' ? mode : '');

  // HITL mode manages its own running state and UI setup.
  if (mode === 'hitl') {
    startHITLResearch();
    return;
  }

  appState.isResearchRunning = true;
  appState.researchStartTime = Date.now();
  appState.runId = null;
  appState.userCancelled = false;
  appState.reportShown = false;
  stopPollRun();
  appState.activeSteps = useTopic ? KNOWLEDGE_RESEARCH_STEPS : CORE_RESEARCH_STEPS;
  appState.pipelineReached = -1;
  appState.nodeDetails = {};
  renderPipelineStrip(appState.activeSteps, '');
  document.body.classList.add('research-active');
  $('foundryQuestion').textContent = question;
  $('foundryStatus').textContent = '正在建立研究图';
  if (researchViz) researchViz.start(question);

  // Update UI
  $('submitBtn').disabled = true;
  $('spinner').style.display = 'block';
  $('btnText').textContent = '研究中...';
  $('cancelBtn').style.display = 'inline-flex';
  $('emptyState').style.display = 'none';
  $('stepPipeline').style.display = 'flex';
  $('progressLog').classList.add('visible');
  $('logContent').innerHTML = '';
  resetNodePanels();
  $('reportView').classList.remove('visible');

  addLog(`启动研究: <strong>${esc(question)}</strong>${isIncremental ? ' <span class="text-cyan">[增量模式]</span>' : ''}`);

  if (useTopic) {
    startKnowledgeResearchSSE(question, useTopic, isIncremental);
    return;
  }

  // Create abort controller
  appState.abortController = new AbortController();
  const style = ($('writingStyle') && $('writingStyle').value) || 'tech-blog';
  const es = new EventSource('/research/stream?question=' + encodeURIComponent(question)
    + '&style=' + encodeURIComponent(style));
  appState.eventSource = es;
  setupSSEHandlers(es);
}

function cancelResearch() {
  appState.userCancelled = true;
  stopPollRun();
  clearLastRun();  // 主动取消，不保留恢复记录
  if (appState.eventSource) {
    appState.eventSource.close();
    appState.eventSource = null;
  }
  if (appState.abortController) {
    appState.abortController.abort();
    appState.abortController = null;
  }
  addLog('<span class="text-amber">研究已取消</span>');
  $('foundryStatus').textContent = '研究已取消';
  if (researchViz) researchViz.running = false;
  finishResearch();
  toast('研究已取消', 'warning');
}

function startKnowledgeResearchSSE(question, topicId, isIncremental) {
  appState.abortController = new AbortController();

  fetch('/research/knowledge/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      topic_id: topicId,
      is_incremental: isIncremental !== false,
      style: ($('writingStyle') && $('writingStyle').value) || 'tech-blog',
    }),
    signal: appState.abortController.signal,
  })
    .then(res => {
      if (!res.ok) throw new Error('Request failed');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let retryCount = 0;

      function read() {
        reader.read()
          .then(({ done, value }) => {
            if (done) {
              finishResearch();
              return;
            }
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop() || '';
            for (let i = 0; i < lines.length; i++) {
              if (lines[i].startsWith('event: ')) {
                const ev = lines[i].slice(7).trim();
                const dl = lines[i + 1];
                if (dl && dl.startsWith('data: ')) {
                  handleSSEEvent(ev, dl.slice(6));
                }
              }
            }
            read();
          })
          .catch(err => {
            if (err.name === 'AbortError') return;
            if (retryCount < 2) {
              retryCount++;
              addLog(`<span class="text-amber">连接中断，重试中 (${retryCount}/2)...</span>`);
              setTimeout(read, 3000);
            } else {
              addLog(`<span class="text-red">连接失败: ${esc(err.message)}</span>`);
              handleSSEConnectionLost();
            }
          });
      }

      read();
    })
    .catch(err => {
      if (err.name === 'AbortError') return;
      addLog(`<span class="text-red">连接失败: ${esc(err.message)}</span>`);
      handleSSEConnectionLost();
    });
}

function setupSSEHandlers(es) {
  let retryCount = 0;

  es.addEventListener('progress', e => handleSSEEvent('progress', e.data));
  es.addEventListener('message', e => handleSSEEvent('message', e.data));
  es.addEventListener('report_chunk', e => handleSSEEvent('report_chunk', e.data));
  es.addEventListener('report_reset', e => handleSSEEvent('report_reset', e.data));
  es.addEventListener('report', e => handleSSEEvent('report', e.data));
  es.addEventListener('done', () => {
    es.close();
    appState.eventSource = null;
    clearLastRun();  // 正常完成，无需再恢复
    finishResearch();
  });
  es.addEventListener('error', e => {
    // SSE 的具名 error 事件（服务端主动发的）和 EventSource 的传输错误
    // 共用同一个监听器。带 data 的是服务端消息，必须交给 handleSSEEvent
    // 显示出来 —— 否则后端「搜索层不可用」这类错误会被当成断线，
    // 用户只看到「SSE 重连中」，真正的原因永远不显示。
    if (e && typeof e.data === 'string' && e.data) {
      handleSSEEvent('error', e.data);
      es.close();
      appState.eventSource = null;
      finishResearch();
      return;
    }
    // Don't immediately close on transient errors
    if (es.readyState === EventSource.CLOSED) {
      appState.eventSource = null;
      handleSSEConnectionLost();
    } else if (retryCount < 2) {
      retryCount++;
      addLog(`<span class="text-amber">SSE 重连中 (${retryCount}/2)...</span>`);
    } else {
      es.close();
      appState.eventSource = null;
      handleSSEConnectionLost();
    }
  });
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) {
      appState.eventSource = null;
      handleSSEConnectionLost();
    }
  };
}

/** SSE 意外断开：有 run_id 且未完成 → 后台线程仍在跑，轮询取回结果 */
function handleSSEConnectionLost() {
  if (appState.reportShown) { finishResearch(); return; }
  if (appState.runId && !appState.userCancelled) {
    addLog('<span class="text-amber">⚠ 连接中断，研究仍在后台进行，正在自动恢复结果…</span>');
    $('foundryStatus').textContent = '连接中断，后台研究中…';
    startPollRun();
  } else {
    finishResearch();
  }
}

function startPollRun() {
  if (!appState.runId) { finishResearch(); return; }
  pollResearchRun();
  appState.runPollTimer = setInterval(pollResearchRun, 5000);
}

function stopPollRun() {
  if (appState.runPollTimer) {
    clearInterval(appState.runPollTimer);
    appState.runPollTimer = null;
  }
}

async function pollResearchRun() {
  const runId = appState.runId;
  if (!runId || appState.userCancelled) { stopPollRun(); return; }
  try {
    const res = await fetch('/research/run/' + runId);
    if (!res.ok) { stopPollRun(); finishResearch(); return; }
    const data = await res.json();
    if (data.status === 'done') {
      stopPollRun();
      if (data.result) {
        showReport(data.result);
        setAllDone();
        addLog('<span class="text-green">✅ 连接中断后研究已在后台完成，结果已恢复</span>');
        toast('研究已在后台完成，结果已恢复', 'success');
      }
      finishResearch();
    } else if (data.status === 'error') {
      stopPollRun();
      addLog(`<span class="text-red">研究失败: ${esc(data.error || '未知')}</span>`);
      finishResearch();
    }
    // running：保持轮询
  } catch (e) {
    // 网络暂不可用，保持轮询
  }
}

/**
 * 记住每个节点最近一次的 detail，报告区的指标卡和来源清单要用。
 * 搜索来源逐轮累积，用于区分「本次检索」和「历史知识/仅报告引用」。
 */
function rememberNodeDetail(node, detail) {
  appState.nodeDetails[node] = detail;
}

function handleSSEEvent(type, raw) {
  try {
    const d = JSON.parse(raw);

    if (type === 'progress') {
      if (d.run_id) { appState.runId = d.run_id; saveLastRun(d.run_id); }
      if (STEPS.includes(d.node)) {
        setStep(d.node, d.detail || {});
        rememberNodeDetail(d.node, d.detail || {});
      }
      if (d.message) addLog(esc(d.message));
      if (d.summary) {
        const nodeMeta = NODE_META[d.node];
        const label = nodeMeta ? `<strong style="color:var(--${nodeMeta.color})">${nodeMeta.title}</strong> ` : '';
        addLog(`${label}${esc(d.summary)}`);
      }
      if (d.detail) showNodeDetail(d.node, d.detail);
    } else if (type === 'message') {
      if (d.text) addLog(esc(d.text));
    } else if (type === 'report') {
      showReport(d);
      setAllDone();
      addLog('<span class="text-green">研究报告生成完成</span>');
    } else if (type === 'report_chunk') {
      handleReportChunk(d);
    } else if (type === 'report_reset') {
      // 改进模式重写：清空已显示的文章，重新流式输出，避免新旧稿拼接
      resetReportStreaming();
    } else if (type === 'interrupt') {
      handleInterrupt(d);
    } else if (type === 'error') {
      $('executionDetails').open = true;
      addLog(`<span class="text-red">错误: ${esc(d.message || '未知')}</span>`);
      toast(d.message || '研究出错', 'error', '研究错误');
    }
  } catch (e) {
    // Partial JSON — ignore
  }
}



/* ═══════════════════════════════════════════════
   Node Detail Rendering — step card history
   ═══════════════════════════════════════════════ */
const NODE_META = {
  knowledge_lookup: { icon: '◎', title: '知识查询', status: '正在读取历史知识', color: 'cyan' },
  planner: { icon: '◈', title: '问题规划', status: '研究问题正在形成分支', color: 'purple' },
  searcher: { icon: '◉', title: '并行搜索', status: '外部来源信号正在汇入', color: 'purple' },
  validator: { icon: '✓', title: '结果验证', status: '正在标记证据缺口', color: 'green' },
  editorial_planner: { icon: '≡', title: '编辑规划', status: '正在建立证据账本与写作主线', color: 'cyan' },
  synthesizer: { icon: '▯', title: '报告综合', status: '证据正在向核心收束', color: 'purple' },
  language_editor: { icon: '≡', title: '语言审校', status: '正在清理模板化表达', color: 'cyan' },
  article_editor: { icon: '✓', title: '事实审校', status: '正在逐项核对断言与证据', color: 'green' },
  reflector: { icon: '↻', title: '质量反思', status: '正在扫描结论稳定性', color: 'amber' },
  knowledge_store: { icon: '◈', title: '知识存储', status: '研究结果正在归档', color: 'green' },
  diff_analyzer: { icon: '⇄', title: '变化分析', status: '正在标记时间差异', color: 'amber' },
  change_notifier: { icon: '◉', title: '变更通知', status: '正在发送关键变化', color: 'green' },
};

/**
 * 构建某个节点的卡片正文与角标。
 * 只负责产出 HTML，不碰 DOM —— 由 showNodeDetail 决定放到哪张面板里。
 */
function buildNodeBody(node, detail) {
  let body = '';
  let badge = '';

  // ── Knowledge Lookup ──
  if (node === 'knowledge_lookup') {
    const has = detail.has_knowledge;
    body = `
      <div class="nd-kl">
        <div class="nd-kl-item">
          <div class="nd-kl-dot ${has ? 'yes' : 'no'}"></div>
          <span class="${has ? 'text-green' : 'text-muted'}">${has ? '发现历史知识' : '全新研究'}</span>
        </div>
        <div class="nd-kl-item text-muted">${detail.knowledge_context_length || 0} chars context</div>
      </div>
    `;
    badge = has ? '发现历史知识' : '全新研究';
  }

  // ── Planner ──
  if (node === 'planner' && detail.sub_questions) {
    badge = `${detail.sub_questions.length} 个子问题`;
    body = detail.sub_questions.map((sq, i) => `
      <div class="nd-sq">
        <div class="nd-sq-num">${i + 1}</div>
        <div class="nd-sq-copy">
          <div class="nd-sq-q">${esc(sq.question)}</div>
          <div class="nd-sq-kw">▸ ${esc(sq.search_query)}</div>
        </div>
      </div>
    `).join('');
  }

  // ── Searcher ──
  if (node === 'searcher') {
    const count = detail.results_count || 0;
    badge = `${count} 条结果`;
    body = `<div class="nd-info">找到 <strong class="text-cyan">${count}</strong> 条搜索结果</div>`;
    if (detail.results_preview && detail.results_preview.length) {
      body += '<div class="nd-sr-grid">' +
        detail.results_preview.map(r => `
          <div class="nd-sr">
            <div class="nd-sr-title">${esc(r.title || 'Untitled')}</div>
            <div class="nd-sr-url">${esc((r.url || '').replace(/^https?:\/\//, '').slice(0, 40))}</div>
          </div>
        `).join('') + '</div>';
    }
  }

  // ── Validator ──
  if (node === 'validator') {
    const gc = detail.gaps_count || 0;
    badge = gc > 0 ? `${gc} 个不足` : '✓ 充足';
    if (gc > 0) {
      body = `<div class="nd-warn">⚠ ${gc} 个子问题需要补充搜索</div>`;
      if (detail.gaps) {
        body += detail.gaps.map((g, i) => `
          <div class="nd-gap">
            <div class="nd-gap-num">${String(i + 1).padStart(2, '0')}</div>
            <div class="nd-gap-copy">
              <div class="nd-gap-q">${esc(g.question)}</div>
              <div class="nd-gap-kw">▸ ${esc(g.search_query)}</div>
            </div>
          </div>
        `).join('');
      }
    } else {
      body = `<div class="nd-ok">✓ 所有子问题搜索结果充足</div>`;
    }
  }

  // ── Editorial Planner ──
  if (node === 'editorial_planner') {
    const count = detail.evidence_count || 0;
    const sections = detail.section_count || 0;
    badge = detail.degraded ? '已降级' : `${sections} 个章节`;
    body = detail.degraded
      ? `<div class="nd-warn">编辑简报生成失败，将使用 ${count} 条证据直接写作</div>`
      : `<div class="nd-info">证据账本 <strong class="text-cyan">${count}</strong> 条，规划 <strong class="text-purple2">${sections}</strong> 个章节</div>
         <div class="nd-sq"><div class="nd-sq-copy"><div class="nd-sq-q">${esc(detail.thesis || '')}</div></div></div>`;
  }

  // ── Synthesizer ──
  if (node === 'synthesizer') {
    const len = detail.report_length || 0;
    const selected = detail.selected_images_count || 0;
    const cached = detail.cached_images_count || 0;
    const embedded = detail.embedded_images_count || 0;
    badge = `${len > 1000 ? (len / 1000).toFixed(1) + 'k' : len} 字符`;
    body = `<div class="nd-info">报告长度: <strong class="text-purple2">${len > 1000 ? (len / 1000).toFixed(1) + 'k' : len}</strong> 字符</div>
      <div class="nd-refl"><dt>视觉选图</dt><dd>${selected}</dd><dt>本地缓存</dt><dd>${cached}</dd><dt>正文配图</dt><dd>${embedded}</dd></div>`;
  }

  // ── Language Editor ──
  if (node === 'language_editor') {
    const count = detail.edits_count || 0;
    const candidates = detail.candidates_count || 0;
    const labels = {
      contrast_template: '模板对比',
      empty_transition: '空泛转折',
      reader_directive: '读者指令',
      meta_summary: '仪式化总结',
      repetitive_opening: '重复开头',
    };
    badge = count ? `${count} 处改写` : candidates ? '候选未改' : '✓ 通过';
    body = count
      ? `<div class="nd-info">已改写 <strong class="text-cyan">${count}</strong> 处模板化表达</div>` +
        (detail.edits_preview || []).map(edit => `
          <div class="nd-sq">
            <div class="nd-sq-copy">
              <div class="nd-sq-kw">${esc(labels[edit.issue_type] || edit.issue_type || '语言问题')} · ${esc(edit.reason || '')}</div>
              <div class="nd-sq-q">原：${esc(edit.before || '')}</div>
              <div class="nd-info">改：${esc(edit.after || '')}</div>
            </div>
          </div>
        `).join('')
      : candidates
        ? `<div class="nd-warn">发现 ${candidates} 处模板表达，但没有通过安全校验的改写</div>`
        : '<div class="nd-ok">✓ 未发现需要处理的模板化表达</div>';
  }

  // ── Article Editor ──
  if (node === 'article_editor') {
    const count = detail.edits_count || 0;
    const types = detail.edit_types || {};
    badge = count ? `${count} 处修改` : detail.changed ? '已去重' : '✓ 通过';
    body = count
      ? `<div class="nd-info">已应用 <strong class="text-green">${count}</strong> 处可验证修改</div>
         <div class="nd-refl"><dt>无支持</dt><dd>${types.unsupported || 0}</dd><dt>过度断言</dt><dd>${types.overstated || 0}</dd><dt>重复</dt><dd>${types.redundant || 0}</dd></div>`
      : detail.changed
        ? '<div class="nd-ok">✓ 已清理正文中的重复表达</div>'
        : '<div class="nd-ok">✓ 未发现需要局部修改的断言</div>';
  }

  // ── Reflector ──
  if (node === 'reflector') {
    const passed = detail.reflection_pass;
    const round = detail.reflection_round || 0;
    // 轮次由卡片头部的 .sc-round 芯片显示，角标只留判定结果
    badge = passed ? '✓ 通过' : '✗ 需修正';
    body = `
      <div class="nd-refl">
        <dt>轮次</dt><dd>${round}</dd>
        <dt>判定</dt><dd class="${passed ? 'text-green' : 'text-amber'}">${passed ? '✓ 通过' : '✗ 需修正'}</dd>
    `;
    if (detail.reflection_feedback) {
      body += `<dt>反馈</dt><dd>${esc(detail.reflection_feedback)}</dd>`;
    }
    if (detail.best_report_restored) {
      body += `<dt>交付版本</dt><dd class="text-cyan">已恢复第 ${detail.best_reflection_round || 1} 轮最佳稿</dd>`;
    }
    if (detail.supplement_queries) {
      body += `<dt>补充</dt><dd class="text-cyan">${detail.supplement_queries.map(esc).join(', ')}</dd>`;
    }
    body += '</div>';
  }

  // ── Knowledge Store ──
  if (node === 'knowledge_store') {
    const rid = detail.saved_report_id || '';
    badge = rid ? rid.slice(0, 8) + '…' : '✓ 已保存';
    body = `<div class="nd-ok">✓ 报告已保存${rid ? ` <span class="text-muted font-mono">(${rid.slice(0, 8)}…)</span>` : ''}</div>`;
  }

  // ── Diff Analyzer ──
  if (node === 'diff_analyzer') {
    const cc = detail.changes_count || 0;
    const sim = detail.similarity || 0;
    badge = `${(sim * 100).toFixed(1)}% · ${cc} 变化`;
    body = `<div class="nd-info">相似度: <strong class="text-cyan">${(sim * 100).toFixed(1)}%</strong> · 变更: <strong class="${cc ? 'text-amber' : 'text-green'}">${cc} 项</strong></div>`;
  }

  // ── Change Notifier ──
  if (node === 'change_notifier') {
    badge = detail.notification_sent ? '✓ 已发送' : '— 无需通知';
    body = `<div class="${detail.notification_sent ? 'nd-ok' : 'text-muted'}" style="font-size:.76rem">${detail.notification_sent ? '✓ 通知已发送' : '— 无需通知'}</div>`;
  }

  return { body, badge };
}

/* ── 节点面板：每个节点一张卡，原地更新，旧轮次收进折叠区 ──────────
   以前是每来一个事件就 append 一张新卡片，补搜/重写循环会让 validator、
   searcher、synthesizer 各堆出四五张几乎一样的卡片（实测 7000px 高）。 */
const nodePanels = new Map();

function resetNodePanels() {
  nodePanels.clear();
  const container = $('stepCards');
  if (container) container.replaceChildren();
}

/** 取到（必要时创建）某节点的面板，并按图的节点顺序插入，而不是按到达顺序 */
function ensureNodePanel(node, meta) {
  const existing = nodePanels.get(node);
  if (existing) return existing;

  const flow = appState.activeSteps.length ? appState.activeSteps : STEPS;
  const order = flow.indexOf(node) >= 0 ? flow.indexOf(node) : flow.length + nodePanels.size;

  const card = document.createElement('section');
  card.className = `step-card sc-${meta.color}`;
  card.dataset.order = String(order);
  card.innerHTML = `
    <div class="sc-header">
      <div class="sc-icon">${meta.icon}</div>
      <span class="sc-title">${esc(meta.title)}</span>
      <span class="sc-round" hidden></span>
      <span class="sc-badge"></span>
    </div>
    <div class="sc-body"></div>
    <details class="sc-history" hidden>
      <summary><span class="sc-history-label">历史轮次</span></summary>
      <div class="sc-history-body"></div>
    </details>
  `;

  const container = $('stepCards');
  const next = [...container.children].find(el => Number(el.dataset.order) > order);
  container.insertBefore(card, next || null);

  const panel = {
    card,
    bodyEl: card.querySelector('.sc-body'),
    badgeEl: card.querySelector('.sc-badge'),
    roundEl: card.querySelector('.sc-round'),
    historyEl: card.querySelector('.sc-history'),
    historyBodyEl: card.querySelector('.sc-history-body'),
    historyLabelEl: card.querySelector('.sc-history-label'),
    rounds: 0,
  };
  nodePanels.set(node, panel);
  return panel;
}

/** 把当前正文挪进历史折叠区，标上它属于第几轮 */
function archiveNodeRound(panel) {
  const round = panel.rounds - 1;
  const entry = document.createElement('div');
  entry.className = 'sc-history-entry';
  const label = document.createElement('div');
  label.className = 'sc-history-round';
  label.textContent = `第 ${round} 轮${panel.badgeEl.textContent ? ' · ' + panel.badgeEl.textContent : ''}`;
  entry.appendChild(label);
  const holder = document.createElement('div');
  holder.innerHTML = panel.bodyEl.innerHTML;
  entry.appendChild(holder);
  panel.historyBodyEl.prepend(entry);

  const count = panel.historyBodyEl.children.length;
  panel.historyLabelEl.textContent = `历史轮次（${count}）`;
  panel.historyEl.hidden = false;
}

function showNodeDetail(node, detail) {
  if (!detail || Object.keys(detail).length === 0) return;

  const meta = NODE_META[node] || { icon: '●', title: node, color: 'purple' };
  const { body, badge } = buildNodeBody(node, detail);
  if (!body) return;

  const panel = ensureNodePanel(node, meta);
  panel.rounds += 1;
  if (panel.rounds > 1) archiveNodeRound(panel);

  panel.badgeEl.textContent = badge || '';
  panel.roundEl.textContent = `第 ${panel.rounds} 轮`;
  panel.roundEl.hidden = panel.rounds < 2;
  panel.bodyEl.innerHTML = body;
  panel.card.classList.remove('is-updated');
  // 强制重排以便重复触发高亮动画
  void panel.card.offsetWidth;
  panel.card.classList.add('is-updated');

  if ($('executionDetails').open) {
    setTimeout(() => panel.card.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50);
  }
}

function showReport(data) {
  const v = $('reportView');
  v.classList.add('visible');
  appState.reportShown = true;

  // 如果之前有实时渲染的原始文本，先保存
  if (appState.streamingReportText) {
    appState.lastReportText = appState.streamingReportText;
    appState.streamingReportText = '';
  } else {
    appState.lastReportText = data.report || '';
  }

  const m = $('reportMeta');
  const tu = data.token_usage || {};
  const items = [
    data.topic_id ? { l: '主题', v: '已关联', c: 'badge-cyan' } : null,
    data.is_incremental ? { l: '模式', v: '增量', c: 'badge-amber' } : null,
    { l: '子问题', v: (data.sub_questions || []).length },
    { l: '来源', v: data.search_results_count || 0 },
    { l: '反思', v: data.reflection_round || 0 },
    { l: '通过', v: data.reflection_pass ? '✓' : '↻' },
    tu.total_tokens ? { l: 'Tokens', v: fmtTokens(tu.total_tokens) } : null,
    { l: '耗时', v: elapsed() },
  ].filter(Boolean);

  m.innerHTML = items.map(i => `
    <div class="meta-chip">
      ${i.c ? `<span class="badge ${i.c}">${i.v}</span>` : `<span class="mv">${i.v}</span>`} ${i.l}
    </div>
  `).join('');

  // 用完整 Markdown 替换实时渲染的原始文本
  const body = $('reportBody');
  try {
    body.innerHTML = renderMarkdown(appState.lastReportText);
  } catch (e) {
    body.innerHTML = `<p class="text-red">报告渲染失败: ${esc(e.message)}</p><pre>${esc(appState.lastReportText)}</pre>`;
  }
  attachImageFallbacks(body);
  renderMermaidBlocks(body);
  body.classList.remove('streaming-mode', 'is-streaming');
  // 从流式草稿平滑过渡到定稿排版，而不是直接跳变
  body.classList.remove('is-settled');
  void body.offsetWidth;
  body.classList.add('is-settled', 'typing');

  buildReportScorecard(data);
  buildReportToc();

  // 显示折叠按钮
  $('collapseBtn').style.display = 'inline-flex';
  $('collapsedInfoText').textContent = `研究报告 · ${appState.lastReportText.length} 字符（点击展开）`;
}

/* ── 报告顶部证据质量指标卡 ─────────────────────────── */

function buildReportScorecard(data) {
  const host = $('reportScorecard');
  if (!host) return;

  const validator = appState.nodeDetails.validator || {};
  const total = validator.branch_total || (data.sub_questions || []).length;
  const sufficient = validator.branch_sufficient || 0;
  const coverage = typeof validator.avg_coverage === 'number' ? validator.avg_coverage : null;
  const score = data.reflection_score || (appState.nodeDetails.reflector || {}).reflection_score || 0;
  // 新报告：置信度由后端代码从证据质量计算（data.confidence）；
  // 旧报告：正文里可能还有「置信度：高/中/低」文本，回退用 extractConfidence
  let confidence;
  const _conf = String(data.confidence || '');
  if (_conf === '高' || _conf === '中' || _conf === '低') {
    const _tones = { '高': 'good', '中': 'warn', '低': 'bad' };
    confidence = { text: _conf, tone: _tones[_conf], note: '证据质量评估' };
  } else {
    confidence = extractConfidence(appState.lastReportText);
  }

  const tiles = [
    {
      label: '整体置信度',
      value: confidence.text,
      tone: confidence.tone,
      note: confidence.note,
    },
    {
      label: '分支证据充足',
      value: total ? `${sufficient}/${total}` : '—',
      tone: !total ? 'muted' : (sufficient === total ? 'good' : (sufficient === 0 ? 'bad' : 'warn')),
      note: coverage === null ? '未评估覆盖度' : `平均覆盖度 ${(coverage * 100).toFixed(0)}%`,
    },
    {
      label: '质量评分',
      value: score ? `${score}/15` : '—',
      // 硬规则（引用一致性、未解决缺口、用户反馈）可以在评分达标时仍然判不通过，
      // 所以色调看的是「是否通过」，不是分数本身，否则会出现 15/15 却未通过的错觉
      tone: !score ? 'muted' : (data.reflection_pass ? 'good' : (score >= 12 ? 'warn' : 'bad')),
      note: data.reflection_pass
        ? `第 ${data.reflection_round || 1} 轮通过`
        : (score >= 12
          ? '评分达标，但引用或缺口未过硬规则'
          : `第 ${data.reflection_round || 1} 轮未通过`),
    },
    {
      label: '检索来源',
      value: String(data.search_results_count || 0),
      tone: (data.search_results_count || 0) > 0 ? 'good' : 'bad',
      note: data.is_incremental ? '增量模式（含历史知识）' : '本次检索去重后',
    },
  ];

  const warnings = [];
  if (data.search_unavailable) warnings.push('搜索层不可用，本次未获得新证据');
  if (data.evidence_assessment_degraded) warnings.push('语义证据评估不可用，仅做了机械校验');
  if (data.stop_reason === 'search_budget_exhausted') warnings.push('已用尽搜索预算，部分内容证据有限');
  if (data.stop_reason === 'reflection_budget_exhausted') warnings.push('已用尽反思轮次，部分章节可能还有完善空间');
  if (data.stop_reason === 'no_new_queries') warnings.push('已无新的补充查询可执行');

  host.replaceChildren();
  const grid = document.createElement('div');
  grid.className = 'rs-grid';
  tiles.forEach(t => {
    const tile = document.createElement('div');
    tile.className = `rs-tile is-${t.tone}`;
    const l = document.createElement('span');
    l.className = 'rs-label';
    l.textContent = t.label;
    const val = document.createElement('strong');
    val.className = 'rs-value';
    val.textContent = t.value;
    const n = document.createElement('span');
    n.className = 'rs-note';
    n.textContent = t.note;
    tile.append(l, val, n);
    grid.appendChild(tile);
  });
  host.appendChild(grid);

  if (warnings.length) {
    const box = document.createElement('div');
    box.className = 'rs-warnings';
    warnings.forEach(w => {
      const row = document.createElement('div');
      row.className = 'rs-warning';
      row.textContent = `⚠ ${w}`;
      box.appendChild(row);
    });
    host.appendChild(box);
  }

  // 研究说明：局限/降级/未解决缺口等评价性信息，存于后端 research_notes，
  // 默认折叠展示，不影响文章正文的可发布性
  const notes = Array.isArray(data.research_notes) ? data.research_notes : [];
  if (notes.length) {
    const det = document.createElement('details');
    det.className = 'rs-notes';
    const sum = document.createElement('summary');
    sum.textContent = `研究说明（${notes.length}）`;
    const list = document.createElement('ul');
    notes.forEach(n => {
      const li = document.createElement('li');
      li.textContent = n;
      list.appendChild(li);
    });
    det.append(sum, list);
    host.appendChild(det);
  }
  host.hidden = false;
}

/** 从报告正文里读「整体置信度：高/中/低」，兼容加粗等写法 */
function extractConfidence(text) {
  const m = /(?:整体)?置信度[：:\s*]*\**\s*([高中低])/.exec(text || '');
  const level = m ? m[1] : '';
  if (level === '高') return { text: '高', tone: 'good', note: '报告自评' };
  if (level === '中') return { text: '中', tone: 'warn', note: '报告自评' };
  if (level === '低') return { text: '低', tone: 'bad', note: '报告自评' };
  return { text: '未标注', tone: 'muted', note: '报告未给出置信度' };
}

/* ── 章节目录 ──────────────────────────────────────── */

function buildReportToc() {
  const toc = $('reportToc');
  const body = $('reportBody');
  if (!toc || !body) return;

  if (appState.tocObserver) {
    appState.tocObserver.disconnect();
    appState.tocObserver = null;
  }

  const heads = [...body.querySelectorAll('h2, h3')];
  if (heads.length < 2) {
    toc.hidden = true;
    toc.replaceChildren();
    return;
  }

  const title = document.createElement('div');
  title.className = 'toc-title';
  title.textContent = '章节目录';
  const list = document.createElement('ul');
  list.className = 'toc-list';

  heads.forEach((h, i) => {
    const id = `report-sec-${i}`;
    h.id = id;
    const li = document.createElement('li');
    li.className = `toc-item toc-${h.tagName.toLowerCase()}`;
    const a = document.createElement('a');
    a.href = `#${id}`;
    a.textContent = (h.textContent || '').trim();
    a.dataset.target = id;
    a.addEventListener('click', ev => {
      ev.preventDefault();
      h.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    li.appendChild(a);
    list.appendChild(li);
  });

  toc.replaceChildren(title, list);
  toc.hidden = false;

  // 阅读位置高亮：取当前视口内最靠上的可见标题
  const links = new Map([...list.querySelectorAll('a')].map(a => [a.dataset.target, a]));
  appState.tocObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      links.forEach(a => a.classList.remove('is-active'));
      const a = links.get(entry.target.id);
      if (a) a.classList.add('is-active');
    });
  }, { rootMargin: '-12% 0px -80% 0px', threshold: 0 });
  heads.forEach(h => appState.tocObserver.observe(h));
}

function resetReportStreaming() {
  const v = $('reportView');
  const body = $('reportBody');
  appState.streamingReportText = '';
  body.replaceChildren();
  body.classList.add('streaming-mode', 'is-streaming', 'typing');
  body.classList.remove('is-settled');
  // 生成中先隐藏定稿后才有意义的模块
  ['reportScorecard', 'reportToc'].forEach(id => {
    const el = $(id);
    if (el) el.hidden = true;
  });
  return { v, body };
}

function handleReportChunk(d) {
  const chunk = d.chunk || '';
  if (!chunk) return;

  const v = $('reportView');
  const body = $('reportBody');

  // 第一次收到 chunk 时初始化
  if (!v.classList.contains('visible')) {
    v.classList.add('visible');
    resetReportStreaming();
  }

  // 累积原始 Markdown 文本
  appState.streamingReportText += chunk;

  // 节流渲染完整 Markdown，避免逐 token 解析和原始 Markdown 排版混乱。
  if (!appState.reportRenderTimer) {
    appState.reportRenderTimer = setTimeout(() => {
      appState.reportRenderTimer = null;
      try {
        body.innerHTML = renderMarkdown(appState.streamingReportText);
      } catch {
        body.textContent = appState.streamingReportText;
      }
      attachImageFallbacks(body);
      // 光标跟在正文末尾，表示还在生成
      const caret = document.createElement('span');
      caret.className = 'report-caret';
      caret.setAttribute('aria-hidden', 'true');
      body.appendChild(caret);
      body.scrollTop = body.scrollHeight;
    }, 140);
  }
}

function toggleReportCollapse() {
  const wrapper = $('reportContentWrapper');
  const collapsedInfo = $('reportCollapsedInfo');
  const btn = $('collapseBtn');
  const btnText = $('collapseBtnText');

  if (wrapper.classList.contains('collapsed')) {
    // 展开
    wrapper.classList.remove('collapsed');
    collapsedInfo.style.display = 'none';
    btn.classList.remove('collapsed');
    btnText.textContent = '折叠';
  } else {
    // 折叠
    wrapper.classList.add('collapsed');
    collapsedInfo.style.display = 'flex';
    btn.classList.add('collapsed');
    btnText.textContent = '展开';
  }
}

function finishResearch() {
  stopPollRun();
  appState.isResearchRunning = false;
  if (researchViz) researchViz.running = false;
  document.body.classList.remove('research-active');
  $('submitBtn').disabled = false;
  $('spinner').style.display = 'none';
  $('btnText').textContent = '开始研究';
  $('cancelBtn').style.display = 'none';
  appState.abortController = null;
  appState.eventSource = null;
  appState.hitlThreadId = null;
  appState.hitlInterruptPoint = null;
  appState.activeSteps = [];
  if (appState.reportRenderTimer) {
    clearTimeout(appState.reportRenderTimer);
    appState.reportRenderTimer = null;
  }
  appState.streamingReportText = '';
  setTimeout(() => $('reportBody')?.classList.remove('typing'), 1500);
}

function copyReport() {
  if (appState.lastReportText) {
    copyToClipboard(appState.lastReportText);
  } else {
    toast('没有可复制的报告', 'warning');
  }
}

/** 导出报告为 Markdown 文件（含问题标题 + 正文 + 参考文献） */
function downloadReport() {
  const text = appState.lastReportText || '';
  if (!text) {
    toast('没有可下载的报告', 'warning');
    return;
  }
  const q = ($('foundryQuestion') && $('foundryQuestion').textContent) || '研究报告';
  const title = q.trim().replace(/[\\/:*?"<>|\n]+/g, ' ').slice(0, 60);
  const md = `# ${q.trim()}\n\n${text}`;
  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (title || '研究报告') + '.md';
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(a.href);
  a.remove();
}

// Keyboard shortcuts
$('question').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    startResearch();
  }
});

document.addEventListener('keydown', e => {
  // Ctrl/Cmd + K — focus research input
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    navigate('research');
    setTimeout(() => $('question').focus(), 100);
  }
  // Escape — close modal or cancel research
  if (e.key === 'Escape') {
    if (appState.isResearchRunning) {
      cancelResearch();
    } else {
      // Close any visible modal
      $$('.modal-overlay.visible').forEach(m => m.classList.remove('visible'));
      if ($('confirmOverlay').classList.contains('visible')) {
        resolveConfirm(false);
      }
    }
  }
});



/* ═══════════════════════════════════════════════
   Research Mode Refresh
   ═══════════════════════════════════════════════ */
async function refreshResearchModes() {
  const sel = $('researchMode');
  const currentVal = sel.value;

  // Remove all options except the first (normal)
  while (sel.options.length > 1) {
    sel.remove(1);
  }

  try {
    const topics = await api('/topics');
    topics.forEach(t => {
      const o = document.createElement('option');
      o.value = t.id;
      o.textContent = '◫ ' + t.name;
      sel.appendChild(o);
    });

    // Restore selection if still valid
    if (currentVal && Array.from(sel.options).some(o => o.value === currentVal)) {
      sel.value = currentVal;
    }
  } catch { /* ignore */ }
}
