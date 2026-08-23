/* ═══════════════════════════════════════════════
   Report Detail
   ═══════════════════════════════════════════════ */
async function loadReportDetail(reportId) {
  const c = $('reportDetailContent');
  c.innerHTML = '<div class="skeleton" style="height:300px;border-radius:var(--r2)"></div>';

  try {
    const r = await api('/reports/' + reportId);
    const d = new Date(r.created_at).toLocaleString('zh-CN');
    const b = [
      r.is_incremental ? '<span class="badge badge-cyan">增量</span>' : '<span class="badge badge-accent">全新</span>',
      r.confidence ? `<span class="badge badge-${r.confidence === '高' ? 'green' : r.confidence === '中' ? 'amber' : 'red'}">${r.confidence}</span>` : '',
      r.total_tokens ? `<span class="badge badge-muted">⚡ ${fmtTokens(r.total_tokens)} tokens</span>` : '',
      `<span class="badge badge-muted">${r.search_results_count || 0} src</span>`,
      `<span class="badge badge-muted">${r.reflection_rounds || 0} rounds</span>`,
    ].filter(Boolean).join('');

    // Store for copy
    appState.lastReportText = r.report || '';

    let reportHtml = '';
    try {
      reportHtml = renderMarkdown(r.report || '');
    } catch (e) {
      reportHtml = `<p class="text-red">报告渲染失败</p><pre>${esc(r.report || '')}</pre>`;
    }

    c.innerHTML = `
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px">${b}</div>
      <h2 style="font-size:1.2rem;font-weight:700;margin-bottom:3px">${esc(r.question)}</h2>
      <div class="text-muted font-mono" style="font-size:.74rem;margin-bottom:20px">${d}</div>
      <div class="report-body">${reportHtml}</div>
      ${r.sources && r.sources.length ? `
        <div style="margin-top:20px">
          <h3 class="section-title">📎 参考来源</h3>
          <div style="display:flex;flex-direction:column;gap:5px">
            ${r.sources.map(s => `
              <a href="${esc(safeUrl(s.url))}" target="_blank" rel="noopener noreferrer" class="source-link">
                <span class="source-icon" aria-hidden="true">◈</span>${esc(s.title || s.url)}
              </a>
            `).join('')}
          </div>
        </div>
      ` : ''}
      ${r.research_notes && r.research_notes.length ? `
        <details class="rs-notes" style="margin-top:16px">
          <summary>研究说明（${r.research_notes.length}）</summary>
          <ul style="margin:8px 0 0;padding-left:20px;color:var(--text-2, #8899aa)">
            ${r.research_notes.map(n => `<li>${esc(n)}</li>`).join('')}
          </ul>
        </details>
      ` : ''}
    `;
    attachImageFallbacks(c);
    renderMermaidBlocks(c);
  } catch (e) {
    c.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">⚠</div>
        <div class="empty-text">加载失败: ${esc(e.message)}</div>
      </div>
    `;
  }
}



/* ═══════════════════════════════════════════════
   Utility
   ═══════════════════════════════════════════════ */
function safeUrl(value) {
  try {
    const url = new URL(String(value || ''), window.location.origin);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : '#';
  } catch {
    return '#';
  }
}

function renderMarkdown(markdown) {
  // 1) 先抽出 LaTeX 公式（$$…$$ 独立公式、$…$ 行内公式），换成占位符，
  //    避免 marked 把 $ 当作普通文本、也避免被下面的净化逻辑干扰。
  const mathStore = [];
  const mathRe = /(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g;
  const stripped = (markdown || '').replace(mathRe, m => {
    const id = '\u0001M' + mathStore.length + '\u0001';
    mathStore.push(m);
    return id;
  });

  const template = document.createElement('template');
  template.innerHTML = marked.parse(stripped);
  template.content.querySelectorAll('script, style, iframe, object, embed, link, meta').forEach(el => el.remove());
  template.content.querySelectorAll('*').forEach(el => {
    for (const attr of [...el.attributes]) {
      const name = attr.name.toLowerCase();
      if (name.startsWith('on') || name === 'style' || name === 'srcdoc') {
        el.removeAttribute(attr.name);
      }
    }
    if (el.hasAttribute('href')) el.setAttribute('href', safeUrl(el.getAttribute('href')));
    if (el.hasAttribute('src')) el.setAttribute('src', safeUrl(el.getAttribute('src')));
    if (el.tagName === 'A') {
      el.setAttribute('target', '_blank');
      el.setAttribute('rel', 'noopener noreferrer');
    }
  });

  // 2) 把占位符替换回 KaTeX 渲染的公式
  if (mathStore.length) {
    const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    textNodes.forEach(node => {
      const text = node.nodeValue;
      if (!text || text.indexOf('\u0001M') === -1) return;
      const frag = document.createDocumentFragment();
      const parts = text.split(/\u0001M(\d+)\u0001/);
      for (let i = 0; i < parts.length; i++) {
        if (i % 2 === 0) {
          if (parts[i]) frag.appendChild(document.createTextNode(parts[i]));
        } else {
          const raw = mathStore[Number(parts[i])];
          frag.appendChild(buildMathElement(raw));
        }
      }
      node.parentNode.replaceChild(frag, node);
    });
  }

  // 3) 插图统一包装成 figure + figcaption（alt 变题注），与正文排版协调。
  // 顺序很重要：先记录原父节点并完成 DOM 替换，再把 img 移进 figure——
  // 若先 appendChild 把 img 移进 figure，img.parentNode 就变成 figure 自己，
  // 之后 replaceChild 会抛 "The new child element contains the parent"。
  template.content.querySelectorAll('img').forEach(img => {
    try {
      img.loading = 'lazy';
      const parent = img.parentNode;
      const onlyChild = !!(parent && parent.tagName === 'P' && parent.childNodes.length === 1);
      const fig = document.createElement('figure');
      fig.className = 'report-figure';
      if (onlyChild && parent && parent.parentNode) {
        parent.parentNode.replaceChild(fig, parent);   // 整个 p 换成 figure
      } else if (parent) {
        parent.replaceChild(fig, img);                 // 仅把 img 换成 figure
      }
      fig.appendChild(img);                            // 把 img 移进 figure（此时已脱离原位置）
      const alt = img.getAttribute('alt') || '';
      if (alt) {
        const cap = document.createElement('figcaption');
        cap.textContent = alt;
        fig.appendChild(cap);
      }
    } catch (e) {
      // 单张图包装失败不影响整篇渲染
      console.warn('插图包装失败，跳过:', e);
    }
  });
  return template.innerHTML;
}

/** 用 KaTeX 渲染一段 LaTeX 公式（$$…$$ 或 $…$），失败/无 KaTeX 时回退为源码 */
function buildMathElement(raw) {
  const display = raw.startsWith('$$');
  const tex = raw.replace(/^\$\$?/, '').replace(/\$\$?$/, '').trim();
  const el = document.createElement('span');
  if (window.katex) {
    try {
      el.innerHTML = window.katex.renderToString(tex, {
        throwOnError: false, displayMode: display, output: 'html',
      });
      el.className = display ? 'report-math report-math-display' : 'report-math';
      return el;
    } catch (e) {
      // fall through
    }
  }
  el.textContent = raw;
  el.className = 'report-math report-math-fallback';
  return el;
}

/** 插图加载失败自动隐藏（热链接的第三方图片可能失效/防盗链）。
 * 必须在 innerHTML 插入真实 DOM 之后调用——模板序列化会丢失监听器。 */
function attachImageFallbacks(root) {
  if (!root) return;
  root.querySelectorAll('img').forEach(img => {
    if (img.dataset.imgFallback) return;
    img.dataset.imgFallback = '1';
    img.addEventListener('error', () => { img.style.display = 'none'; });
  });
}

/* ── Mermaid 图解渲染（LLM 自绘架构图/流程图） ────────── */

let _mermaidReady = false;
let _mermaidSeq = 0;

function initMermaid() {
  if (!window.mermaid || _mermaidReady) return;
  _mermaidReady = true;
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',          // 防止图内脚本注入
    theme: 'default',
    fontFamily: "'DM Sans','PingFang SC',sans-serif",
    flowchart: { curve: 'basis', padding: 12 },
  });
}

/** 把正文里的 ```mermaid 代码块渲染成 SVG（marked 渲染为
 * <pre><code class="language-mermaid">），渲染失败保留原代码块。 */
async function renderMermaidBlocks(root) {
  if (!window.mermaid || !root) return;
  try { initMermaid(); } catch (e) { return; }
  const blocks = root.querySelectorAll('pre > code.language-mermaid');
  for (const code of blocks) {
    const pre = code.parentNode;
    if (!pre || pre.dataset.mermaidDone) continue;
    pre.dataset.mermaidDone = '1';
    try {
      const id = 'mmd' + (++_mermaidSeq);
      const { svg } = await window.mermaid.render(id, code.textContent);
      const holder = document.createElement('div');
      holder.className = 'report-mermaid';
      holder.innerHTML = svg;
      pre.replaceWith(holder);
    } catch (e) {
      console.warn('Mermaid 图解渲染失败，保留源码:', e);
      pre.dataset.mermaidDone = '';
    }
  }
}

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

/** token 数友好显示：12345 → "12.3k" */
function fmtTokens(n) {
  n = Number(n || 0);
  if (n < 1000) return String(n);
  return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
}

