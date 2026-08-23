/* ═══════════════════════════════════════════════
   HITL (Human-in-the-Loop)
   ═══════════════════════════════════════════════ */
function startHITLResearch() {
  const question = $('question').value.trim();
  if (!question) { $('question').focus(); return; }
  if (appState.isResearchRunning) return;

  appState.isResearchRunning = true;
  appState.researchStartTime = Date.now();
  appState.activeSteps = CORE_RESEARCH_STEPS;
  appState.pipelineReached = -1;
  appState.nodeDetails = {};
  renderPipelineStrip(appState.activeSteps, '');
  document.body.classList.add('research-active');
  $('foundryQuestion').textContent = question;
  $('foundryStatus').textContent = '正在建立交互研究图';
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
  $('hitlPanel').style.display = 'none';
  $('reportContentWrapper').classList.remove('collapsed');
  $('reportCollapsedInfo').style.display = 'none';
  $('collapseBtn').style.display = 'none';
  $('collapseBtn').classList.remove('collapsed');
  $('collapseBtnText').textContent = '折叠';
  appState.streamingReportText = '';
  appState.reportCollapsed = false;

  addLog(`启动 <strong class="text-purple2">交互研究</strong>: <strong>${esc(question)}</strong>`);

  // Use fetch + ReadableStream for POST-based SSE
  appState.abortController = new AbortController();
  fetch('/research/hitl/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
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
            if (done) { finishResearch(); return; }
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
            if (retryCount < 2) { retryCount++; setTimeout(read, 3000); }
            else { addLog(`<span class="text-red">连接失败: ${esc(err.message)}</span>`); finishResearch(); }
          });
      }
      read();
    })
    .catch(err => {
      if (err.name === 'AbortError') return;
      addLog(`<span class="text-red">连接失败: ${esc(err.message)}</span>`);
      finishResearch();
    });
}

function handleInterrupt(data) {
  appState.hitlThreadId = data.thread_id;
  appState.hitlInterruptPoint = data.interrupt_point;

  const panel = $('hitlPanel');
  const body = $('hitlBody');

  if (data.interrupt_point === 'confirm_sub_questions') {
    // 子问题确认面板
    $('hitlTitle').textContent = '确认研究规划';
    $('hitlSubtitle').textContent = '审核 AI 拆解的子问题，可编辑后继续';
    $('hitlSkipBtn').textContent = '直接继续';
    $('hitlConfirmBtn').textContent = '确认并继续';

    const sqs = data.sub_questions || [];
    body.innerHTML = '<div style="margin-bottom:12px;font-size:.82rem;color:var(--text2)">AI 将研究问题拆解为以下子问题，你可以修改后继续：</div>' +
      sqs.map((sq, i) => `
        <div class="hitl-sq-item">
          <div class="hitl-sq-num">${i + 1}</div>
          <div class="hitl-sq-fields">
            <input class="input hitl-sq-question" value="${esc(sq.question || '')}" placeholder="子问题（中文）">
            <input class="input hitl-sq-search" value="${esc(sq.search_query || '')}" placeholder="主搜索词（语言自适应）" style="font-family:'JetBrains Mono','Fira Code',monospace;font-size:.78rem">
          </div>
        </div>
      `).join('');

    addLog('<span class="text-purple2">👤 等待确认子问题...</span>');

  } else if (data.interrupt_point === 'review_report') {
    // 报告审核面板
    $('hitlTitle').textContent = '审核研究报告';
    $('hitlSubtitle').textContent = '查看报告草稿，可补充要求后继续';
    $('hitlSkipBtn').textContent = '直接通过';
    $('hitlConfirmBtn').textContent = '提交反馈';

    const report = data.report || '';
    let reportHtml = '';
    try { reportHtml = renderMarkdown(report); } catch { reportHtml = esc(report); }

    body.innerHTML = `
      <div class="hitl-report-preview">${reportHtml}</div>
      <div style="font-size:.82rem;color:var(--text2);margin-bottom:8px">补充要求或改进建议（可选）：</div>
      <textarea class="input hitl-feedback-input" id="hitlFeedback" placeholder="例如：请补充更多数据支撑、请增加对比分析..."></textarea>
    `;

    addLog('<span class="text-purple2">👤 等待审核报告...</span>');
  }

  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resumeHITLResearch(skip) {
  const threadId = appState.hitlThreadId;
  const interruptPoint = appState.hitlInterruptPoint;
  if (!threadId) return;

  // 隐藏中断面板
  $('hitlPanel').style.display = 'none';

  let resumeData = { interrupt_point: interruptPoint };

  if (!skip && interruptPoint === 'confirm_sub_questions') {
    // 收集编辑后的子问题
    const questions = document.querySelectorAll('.hitl-sq-question');
    const searches = document.querySelectorAll('.hitl-sq-search');
    const subQuestions = [];
    questions.forEach((q, i) => {
      subQuestions.push({
        question: q.value.trim(),
        search_query: searches[i] ? searches[i].value.trim() : '',
      });
    });
    resumeData.sub_questions = subQuestions;
    addLog('<span class="text-green">✓ 子问题已确认</span>');

  } else if (!skip && interruptPoint === 'review_report') {
    // 收集用户反馈
    const feedback = $('hitlFeedback') ? $('hitlFeedback').value.trim() : '';
    resumeData.user_feedback = feedback;
    if (feedback) {
      addLog(`<span class="text-green">✓ 反馈已提交: ${esc(feedback.slice(0, 50))}</span>`);
    } else {
      addLog('<span class="text-green">✓ 报告已确认</span>');
    }

  } else {
    addLog('<span class="text-amber">跳过交互，直接继续</span>');
  }

  // 调用恢复端点
  appState.abortController = new AbortController();
  fetch('/research/hitl/resume/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, resume_data: resumeData }),
    signal: appState.abortController.signal,
  })
    .then(res => {
      if (!res.ok) throw new Error('Resume failed');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      function read() {
        reader.read()
          .then(({ done, value }) => {
            if (done) { finishResearch(); return; }
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
            addLog(`<span class="text-red">恢复失败: ${esc(err.message)}</span>`);
            finishResearch();
          });
      }
      read();
    })
    .catch(err => {
      addLog(`<span class="text-red">恢复失败: ${esc(err.message)}</span>`);
      finishResearch();
    });
}

