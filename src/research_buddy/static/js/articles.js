/* ═══════════════════════════════════════════════
   Article Archive
   ═══════════════════════════════════════════════ */

const ARTICLE_STATUS = {
  completed: { label: '已完成', badge: 'badge-green' },
  error: { label: '失败', badge: 'badge-red' },
  running: { label: '运行中', badge: 'badge-amber' },
};

const ARTICLE_CURATION = {
  raw: { label: '未整理', badge: 'badge-muted' },
  candidate: { label: '候选', badge: 'badge-cyan' },
  approved: { label: '已批准', badge: 'badge-green' },
  excluded: { label: '已排除', badge: 'badge-red' },
};

const ARTICLE_SOURCE_LABELS = {
  research_sync: '普通研究',
  research_stream: '流式研究',
  research_cli: '命令行研究',
  knowledge_sync: '知识研究',
  knowledge_stream: '流式知识研究',
  knowledge_cli: '命令行知识研究',
  knowledge_legacy: '历史知识报告',
  hitl: '交互研究',
  tracking: '持续追踪',
};

const ARTICLE_STAGE_LABELS = {
  synthesizer: '综合写作',
  language_editor: '语言审校',
  article_editor: '事实审校',
  reflector: '反思评审',
  legacy_final: '历史最终稿',
};

function articleStatusMeta(value) {
  return ARTICLE_STATUS[value] || { label: value || '未知', badge: 'badge-muted' };
}

function articleCurationMeta(value) {
  return ARTICLE_CURATION[value] || ARTICLE_CURATION.raw;
}

function articleSourceLabel(value) {
  return ARTICLE_SOURCE_LABELS[value] || value || '未知来源';
}

function articleStageLabel(value) {
  return ARTICLE_STAGE_LABELS[value] || value || '未知阶段';
}

function renderArchiveIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function renderArchiveMarkdown(markdown) {
  try {
    return renderMarkdown(markdown || '');
  } catch (_) {
    return `<pre>${esc(markdown || '')}</pre>`;
  }
}

function openArticleFromKeyboard(event, articleId) {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  event.preventDefault();
  navigate('article-detail', { articleId });
}

function articleDate(value) {
  if (!value) return '时间未知';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString('zh-CN');
}

async function loadArticleCount() {
  try {
    const articles = await api('/articles?limit=200');
    $('articleCount').textContent = articles.length >= 200 ? '200+' : articles.length;
  } catch (_) {
    $('articleCount').textContent = '—';
  }
}

async function loadArticles() {
  const target = $('articlesList');
  if (!target) return;
  target.innerHTML = '<div class="skeleton" style="height:320px;border-radius:var(--r2)"></div>';

  try {
    const articles = await api('/articles?limit=200');
    appState.articles = articles;
    $('articleCount').textContent = articles.length >= 200 ? '200+' : articles.length;
    populateArticleSourceFilter(articles);
    filterArticles();
  } catch (error) {
    target.innerHTML = `
      <div class="empty-state">
        <i data-lucide="circle-alert" aria-hidden="true"></i>
        <div class="empty-text">素材库加载失败：${esc(error.message)}</div>
        <button class="btn" onclick="loadArticles()"><i data-lucide="refresh-cw"></i>重试</button>
      </div>`;
    renderArchiveIcons();
  }
}

function populateArticleSourceFilter(articles) {
  const select = $('articleSourceFilter');
  const previous = select.value;
  const sourceTypes = [...new Set(articles.map(item => item.source_type).filter(Boolean))].sort();
  select.innerHTML = '<option value="">全部来源</option>' + sourceTypes.map(value =>
    `<option value="${esc(value)}">${esc(articleSourceLabel(value))}</option>`
  ).join('');
  if (sourceTypes.includes(previous)) select.value = previous;
}

function filterArticles() {
  const query = ($('articleSearch').value || '').trim().toLowerCase();
  const status = $('articleStatusFilter').value;
  const curation = $('articleCurationFilter').value;
  const source = $('articleSourceFilter').value;
  const filtered = appState.articles.filter(item => {
    const haystack = `${item.question || ''}\n${item.report_preview || ''}`.toLowerCase();
    return (!query || haystack.includes(query))
      && (!status || item.status === status)
      && (!curation || item.curation_status === curation)
      && (!source || item.source_type === source);
  });
  renderArticleArchiveSummary(appState.articles, filtered.length);
  renderArticleList(filtered);
}

function renderArticleArchiveSummary(allArticles, visibleCount) {
  const completed = allArticles.filter(item => item.status === 'completed').length;
  const approved = allArticles.filter(item => item.curation_status === 'approved').length;
  const degraded = allArticles.filter(item => item.reflection_judge_degraded).length;
  $('articleArchiveSummary').innerHTML = `
    <div><strong>${allArticles.length}</strong><span>全部素材</span></div>
    <div><strong>${completed}</strong><span>已完成</span></div>
    <div><strong>${approved}</strong><span>已批准</span></div>
    <div><strong>${degraded}</strong><span>Judge 降级</span></div>
    <div class="archive-visible-count"><strong>${visibleCount}</strong><span>当前显示</span></div>`;
}

function renderArticleList(articles) {
  const target = $('articlesList');
  if (!articles.length) {
    target.innerHTML = `
      <div class="empty-state">
        <i data-lucide="archive-x" aria-hidden="true"></i>
        <div class="empty-text">没有符合当前条件的文章</div>
      </div>`;
    renderArchiveIcons();
    return;
  }

  target.innerHTML = `
    <table class="archive-table">
      <thead><tr><th>文章</th><th>生成来源</th><th>模型与反思</th><th>策展</th><th>时间</th><th aria-label="操作"></th></tr></thead>
      <tbody>${articles.map(item => {
        const status = articleStatusMeta(item.status);
        const curation = articleCurationMeta(item.curation_status);
        const preview = (item.report_preview || item.error || '暂无正文').replace(/[#*`]/g, '').trim();
        return `
          <tr class="archive-row" onclick="navigate('article-detail',{articleId:'${item.id}'})" onkeydown="openArticleFromKeyboard(event,'${item.id}')" tabindex="0" role="link">
            <td data-label="文章">
              <div class="archive-title-line">
                <div class="archive-question">${esc(item.question || '未命名文章')}</div>
                <span class="badge ${status.badge}">${status.label}</span>
              </div>
              <div class="archive-preview">${esc(preview.slice(0, 130))}${preview.length > 130 ? '…' : ''}</div>
            </td>
            <td data-label="生成来源"><span class="archive-source">${esc(articleSourceLabel(item.source_type))}</span><small>${esc(item.style || '默认风格')}</small></td>
            <td data-label="模型与反思">
              <span class="archive-model">${esc(item.writer_model || '历史配置未知')}</span>
              <small>${item.reflection_score || 0}/15 · ${item.reflection_rounds || 0} 轮${item.reflection_judge_degraded ? ' · 已降级' : ''}</small>
            </td>
            <td data-label="策展"><span class="badge ${curation.badge}">${curation.label}</span></td>
            <td data-label="时间"><time>${esc(articleDate(item.created_at))}</time></td>
            <td class="archive-open" aria-label="查看详情"><i data-lucide="chevron-right"></i></td>
          </tr>`;
      }).join('')}</tbody>
    </table>`;
  renderArchiveIcons();
}

async function loadArticleDetail(articleId) {
  const target = $('articleDetailContent');
  target.innerHTML = '<div class="skeleton" style="height:440px;border-radius:var(--r2)"></div>';

  try {
    const article = await api('/articles/' + articleId);
    appState.currentArticle = article;
    appState.currentArticleId = articleId;
    $('articleCurationSelect').value = article.curation_status || 'raw';
    target.innerHTML = renderArticleDetail(article);
    appState.lastReportText = article.report || '';
    attachImageFallbacks(target);
    renderMermaidBlocks(target);
    renderArchiveIcons();
    updateBreadcrumb();
  } catch (error) {
    target.innerHTML = `
      <div class="empty-state">
        <i data-lucide="circle-alert" aria-hidden="true"></i>
        <div class="empty-text">文章加载失败：${esc(error.message)}</div>
      </div>`;
    renderArchiveIcons();
  }
}

function renderArticleDetail(article) {
  const status = articleStatusMeta(article.status);
  const curation = articleCurationMeta(article.curation_status);
  const usage = article.token_usage || {};
  const versions = article.versions || [];
  const reviews = article.reviews || [];
  return `
    <header class="article-detail-header">
      <div class="article-detail-kicker">
        <span class="badge ${status.badge}">${status.label}</span>
        <span class="badge ${curation.badge}">${curation.label}</span>
        <span>${esc(articleSourceLabel(article.source_type))}</span>
      </div>
      <h2>${esc(article.question || '未命名文章')}</h2>
      <div class="article-detail-date">${esc(articleDate(article.created_at))}</div>
      <div class="article-metrics">
        <div><span>写作模型</span><strong>${esc(article.writer_model || '未知')}</strong></div>
        <div><span>反思模型</span><strong>${esc(article.judge_model || '未知')}</strong></div>
        <div><span>反思评分</span><strong>${article.reflection_score || 0}/15</strong></div>
        <div><span>反思轮次</span><strong>${article.reflection_rounds || 0}</strong></div>
        <div><span>Token</span><strong>${fmtTokens(usage.total_tokens || 0)}</strong></div>
        <div><span>阶段版本</span><strong>${versions.length}</strong></div>
      </div>
      ${article.reflection_judge_degraded ? '<div class="article-warning"><i data-lucide="triangle-alert"></i>本次反思模型不可用，仅执行了确定性规则检查</div>' : ''}
      ${article.error ? `<div class="article-error"><i data-lucide="circle-x"></i>${esc(article.error)}</div>` : ''}
    </header>
    <div class="article-tabs" role="tablist">
      <button class="article-tab active" data-article-tab="final" onclick="switchArticleTab('final',this)" role="tab" aria-selected="true" aria-controls="article-panel-final">最终稿</button>
      <button class="article-tab" data-article-tab="versions" onclick="switchArticleTab('versions',this)" role="tab" aria-selected="false" aria-controls="article-panel-versions">版本历史 <span>${versions.length}</span></button>
      <button class="article-tab" data-article-tab="reviews" onclick="switchArticleTab('reviews',this)" role="tab" aria-selected="false" aria-controls="article-panel-reviews">质量评价 <span>${reviews.length}</span></button>
    </div>
    <section class="article-tab-panel active" id="article-panel-final" role="tabpanel">${renderArticleFinal(article)}</section>
    <section class="article-tab-panel" id="article-panel-versions" role="tabpanel">${renderArticleVersions(article)}</section>
    <section class="article-tab-panel" id="article-panel-reviews" role="tabpanel">${renderArticleReviews(article)}</section>`;
}

function renderArticleFinal(article) {
  const reportHtml = renderArchiveMarkdown(article.report);
  return `
    <div class="article-final-layout">
      <article class="report-body article-archive-report">${reportHtml || '<p class="text-muted">本次生成没有最终正文。</p>'}</article>
      <aside class="article-inspector">
        ${renderArticleSources(article.sources || [])}
        ${renderArticleImages(article.selected_images || [])}
        ${renderArticleNotes(article.research_notes || [])}
        ${renderArticleEdits(article)}
      </aside>
    </div>`;
}

function renderArticleSources(sources) {
  if (!sources.length) return '<section><h3>参考来源</h3><p class="text-muted">未保存来源</p></section>';
  return `<section><h3>参考来源 <span>${sources.length}</span></h3><div class="article-source-list">${sources.map(source => `
    <a href="${esc(safeUrl(source.url))}" target="_blank" rel="noopener noreferrer">
      <i data-lucide="external-link"></i><span>${esc(source.title || source.url)}</span>
    </a>`).join('')}</div></section>`;
}

function renderArticleImages(images) {
  if (!images.length) return '<section><h3>选中图片</h3><p class="text-muted">本次没有选中图片</p></section>';
  return `<section><h3>选中图片 <span>${images.length}</span></h3><div class="article-image-list">${images.map(image => {
    const url = safeUrl(image.cached_url || image.url);
    return `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer"><img src="${esc(url)}" alt="${esc(image.alt || '文章插图')}" loading="lazy"><span>${esc(image.alt || '文章插图')}</span></a>`;
  }).join('')}</div></section>`;
}

function renderArticleNotes(notes) {
  if (!notes.length) return '';
  return `<section><h3>研究说明 <span>${notes.length}</span></h3><ul class="article-note-list">${notes.map(note => `<li>${esc(note)}</li>`).join('')}</ul></section>`;
}

function renderArticleEdits(article) {
  const languageCount = (article.language_edits || []).length;
  const evidenceCount = (article.evidence_edits || []).length;
  if (!languageCount && !evidenceCount && !article.best_report_restored) return '';
  return `<section><h3>编辑记录</h3><dl class="article-edit-summary">
    <div><dt>语言修改</dt><dd>${languageCount}</dd></div>
    <div><dt>事实修改</dt><dd>${evidenceCount}</dd></div>
    <div><dt>恢复历史最佳</dt><dd>${article.best_report_restored ? '是' : '否'}</dd></div>
  </dl></section>`;
}

function renderArticleVersions(article) {
  const versions = article.versions || [];
  if (!versions.length) return '<div class="empty-state"><div class="empty-text">没有保存阶段版本</div></div>';
  return `
    <div class="version-workbench">
      <nav class="version-list" aria-label="阶段版本">
        ${versions.map((version, index) => `
          <button class="version-item${index === versions.length - 1 ? ' active' : ''}" onclick="selectArticleVersion(${index},this)">
            <span class="version-sequence">${String(version.sequence || index + 1).padStart(2, '0')}</span>
            <span><strong>${esc(articleStageLabel(version.stage))}</strong><small>反思轮次 ${version.reflection_round || 0}</small></span>
            ${version.metadata && version.metadata.changed ? '<i data-lucide="pencil-line"></i>' : ''}
          </button>`).join('')}
      </nav>
      <div class="version-viewer" id="articleVersionViewer">${renderVersionViewer(versions, versions.length - 1)}</div>
    </div>`;
}

function selectArticleVersion(index, button) {
  const article = appState.currentArticle;
  if (!article || !article.versions[index]) return;
  document.querySelectorAll('.version-item').forEach(item => item.classList.remove('active'));
  if (button) button.classList.add('active');
  $('articleVersionViewer').innerHTML = renderVersionViewer(article.versions, index);
  attachImageFallbacks($('articleVersionViewer'));
  renderMermaidBlocks($('articleVersionViewer'));
  renderArchiveIcons();
}

function renderVersionViewer(versions, index, compare = false) {
  const version = versions[index];
  const previous = index > 0 ? versions[index - 1] : null;
  const metadata = version.metadata || {};
  const metaItems = [
    metadata.score !== undefined ? `评分 ${metadata.score}/15` : '',
    metadata.edits_count !== undefined ? `修改 ${metadata.edits_count} 处` : '',
    metadata.judge_degraded ? 'Judge 降级' : '',
    metadata.best_report_restored ? '恢复历史最佳稿' : '',
  ].filter(Boolean);
  if (compare && previous) {
    return `
      <div class="version-viewer-header"><div><span>版本对比</span><h3>${esc(articleStageLabel(previous.stage))} → ${esc(articleStageLabel(version.stage))}</h3></div><button class="btn btn-sm" onclick="toggleArticleVersionCompare(${index},false)"><i data-lucide="panel-top-close"></i>退出对比</button></div>
      <div class="version-compare">
        <section><div class="version-compare-label">上一版本</div><article class="report-body">${renderArchiveMarkdown(previous.report)}</article></section>
        <section><div class="version-compare-label">当前版本</div><article class="report-body">${renderArchiveMarkdown(version.report)}</article></section>
      </div>`;
  }
  return `
    <div class="version-viewer-header">
      <div><span>版本 ${version.sequence || index + 1}</span><h3>${esc(articleStageLabel(version.stage))}</h3><p>${metaItems.map(esc).join(' · ') || '未记录额外元数据'}</p></div>
      ${previous ? `<button class="btn btn-sm" onclick="toggleArticleVersionCompare(${index},true)"><i data-lucide="columns-2"></i>与上一版对比</button>` : ''}
    </div>
    ${version.feedback ? `<div class="version-feedback"><strong>评审反馈</strong><p>${esc(version.feedback)}</p></div>` : ''}
    <article class="report-body version-report">${renderArchiveMarkdown(version.report)}</article>`;
}

function toggleArticleVersionCompare(index, enabled) {
  const versions = (appState.currentArticle && appState.currentArticle.versions) || [];
  $('articleVersionViewer').innerHTML = renderVersionViewer(versions, index, enabled);
  attachImageFallbacks($('articleVersionViewer'));
  renderMermaidBlocks($('articleVersionViewer'));
  renderArchiveIcons();
}

function renderArticleReviews(article) {
  const reviews = article.reviews || [];
  if (!reviews.length) {
    return `<div class="empty-state"><i data-lucide="message-square-dashed"></i><div class="empty-text">还没有质量评价</div><button class="btn btn-primary" onclick="openArticleReview()"><i data-lucide="message-square-plus"></i>添加第一条评价</button></div>`;
  }
  return `<div class="review-list">${reviews.slice().reverse().map(review => {
    const dimensions = Object.entries(review.dimension_scores || {});
    return `<article class="review-item">
      <div class="review-score"><strong>${review.overall_score === null ? '—' : Number(review.overall_score).toFixed(1)}</strong><span>/10</span></div>
      <div class="review-content">
        <header><span>${review.reviewer_type === 'human' ? '人工评价' : esc(review.reviewer_type)}</span><time>${esc(articleDate(review.created_at))}</time>${review.include_in_evaluation ? '<span class="badge badge-green">评价集</span>' : ''}</header>
        ${dimensions.length ? `<div class="review-dimensions">${dimensions.map(([name, score]) => `<span>${esc(reviewDimensionLabel(name))}<b>${esc(score)}</b></span>`).join('')}</div>` : ''}
        ${review.issue_tags && review.issue_tags.length ? `<div class="review-tags">${review.issue_tags.map(tag => `<span>${esc(tag)}</span>`).join('')}</div>` : ''}
        ${review.notes ? `<p>${esc(review.notes)}</p>` : '<p class="text-muted">未填写评价说明</p>'}
      </div>
    </article>`;
  }).join('')}</div>`;
}

function reviewDimensionLabel(name) {
  return { accuracy: '准确性', depth: '深度', naturalness: '自然度', images: '配图' }[name] || name;
}

function switchArticleTab(name, button) {
  document.querySelectorAll('.article-tab').forEach(tab => {
    const active = tab.dataset.articleTab === name;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('.article-tab-panel').forEach(panel => panel.classList.remove('active'));
  const panel = $('article-panel-' + name);
  if (panel) panel.classList.add('active');
  if (button) button.focus();
  renderArchiveIcons();
}

async function saveArticleCuration() {
  if (!appState.currentArticleId) return;
  const value = $('articleCurationSelect').value;
  try {
    const article = await api(`/articles/${appState.currentArticleId}/curation`, {
      method: 'PATCH', body: { curation_status: value },
    });
    appState.currentArticle = article;
    toast('素材状态已保存', 'success');
    await loadArticleDetail(appState.currentArticleId);
  } catch (error) {
    toast('保存失败：' + error.message, 'error');
  }
}

function openArticleReview() {
  if (!appState.currentArticleId) return;
  ['reviewOverall', 'reviewAccuracy', 'reviewDepth', 'reviewNaturalness', 'reviewImages', 'reviewTags', 'reviewNotes'].forEach(id => { $(id).value = ''; });
  $('reviewIncludeEval').checked = false;
  showModal('articleReview');
  renderArchiveIcons();
}

function optionalScore(id) {
  const raw = $(id).value.trim();
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0 || value > 10) throw new Error('评分必须在 0-10 之间');
  return value;
}

async function submitArticleReview() {
  if (!appState.currentArticleId) return;
  const button = $('submitArticleReview');
  try {
    const overall = optionalScore('reviewOverall');
    const dimensions = {
      accuracy: optionalScore('reviewAccuracy'),
      depth: optionalScore('reviewDepth'),
      naturalness: optionalScore('reviewNaturalness'),
      images: optionalScore('reviewImages'),
    };
    Object.keys(dimensions).forEach(key => { if (dimensions[key] === null) delete dimensions[key]; });
    const tags = $('reviewTags').value.split(/[,，]/).map(value => value.trim()).filter(Boolean);
    button.disabled = true;
    await api(`/articles/${appState.currentArticleId}/reviews`, {
      method: 'POST',
      body: {
        reviewer_type: 'human', overall_score: overall,
        dimension_scores: dimensions, issue_tags: tags,
        notes: $('reviewNotes').value.trim(),
        include_in_evaluation: $('reviewIncludeEval').checked,
      },
    });
    hideModal('articleReview');
    toast('文章评价已保存', 'success');
    await loadArticleDetail(appState.currentArticleId);
    switchArticleTab('reviews', document.querySelector('[data-article-tab="reviews"]'));
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
  }
}
