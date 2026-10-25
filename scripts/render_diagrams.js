#!/usr/bin/env node
/**
 * 把 docs/diagrams/html/*.html 渲染成 docs/assets/*.png。
 *
 * 为什么不用 Mermaid 出这两张图：
 * Mermaid 画的是「节点 + 边」的拓扑图，适合精确表达流程走向，但做不出产品文档里
 * 那种分层、带图标、图文并茂的架构图。产品架构图需要排版控制（分栏、对齐、
 * 颜色分区），用 HTML/CSS 写更直接，再截图为 PNG。
 *
 * 为什么是 PNG 而不是 SVG：
 * 这两张图的信息密度靠的是 CSS 排版（flex 分栏、圆角、内阴影）。转 SVG 要么
 * 走 foreignObject（GitHub 会剥离，文字全没），要么把 CSS 盒模型手工翻译成
 * SVG 图元，成本高且易错。用 2x 截图，在 README 里显示宽度约 880px，
 * 实际像素 2960px，任何屏幕都够清晰。
 *
 * 用法：
 *   node scripts/render_diagrams.js            # 渲染全部
 *   node scripts/render_diagrams.js workflow   # 只渲染某一个
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const projectRoot = path.resolve(__dirname, '..');
const htmlDir = path.join(projectRoot, 'docs', 'diagrams', 'html');
const assetsDir = path.join(projectRoot, 'docs', 'assets');

/**
 * 图名 → [html 文件名, 输出 png, 视口宽度]
 *
 * 视口宽度必须贴近 GitHub README 正文区（约 860px）。图比正文区宽时会被
 * 等比压缩，字号跟着变小——这是"图里的字看不清"的根因，不是字号设小了。
 * 所以画布固定 900px，显示时接近 1:1。代价是横向放不下太多列，
 * 只能靠减少信息量来适配：宁可拆成两张图，也不要在一张图里塞满小字。
 */
const DIAGRAMS = {
  architecture: ['architecture.html', 'architecture.png', 900],
  workflow: ['workflow.html', 'workflow.png', 900],
};

function findCachedChromium() {
  const candidates = [];
  const home = os.homedir();
  const roots = [
    path.join(home, '.cache', 'puppeteer', 'chrome'),
    path.join(home, '.cache', 'ms-playwright'),
  ];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    for (const version of fs.readdirSync(root).sort().reverse()) {
      for (const rel of [
        path.join('chrome-linux64', 'chrome'),
        path.join('chrome-linux', 'chrome'),
      ]) {
        candidates.push(path.join(root, version, rel));
      }
    }
  }
  return candidates.find(p => fs.existsSync(p)) || '';
}

async function main() {
  let puppeteer;
  try {
    puppeteer = require('puppeteer-core');
  } catch {
    console.error('缺少 puppeteer-core，请先执行 pnpm install');
    process.exit(1);
  }

  const executablePath = findCachedChromium();
  if (!executablePath) {
    console.error('未找到 Chromium。请先执行：npx puppeteer browsers install chrome');
    process.exit(1);
  }

  const only = process.argv[2];
  const targets = only
    ? Object.entries(DIAGRAMS).filter(([name]) => name === only)
    : Object.entries(DIAGRAMS);

  if (targets.length === 0) {
    console.error(`未知图名：${only}。可选：${Object.keys(DIAGRAMS).join(', ')}`);
    process.exit(1);
  }

  fs.mkdirSync(assetsDir, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--font-render-hinting=none'],
  });

  try {
    for (const [name, [htmlFile, pngFile, width]] of targets) {
      const htmlPath = path.join(htmlDir, htmlFile);
      if (!fs.existsSync(htmlPath)) {
        console.error(`源文件不存在：${htmlPath}`);
        continue;
      }

      const page = await browser.newPage();
      await page.setViewport({ width, height: 1000, deviceScaleFactor: 2 });
      await page.goto(`file://${htmlPath}`, { waitUntil: 'networkidle0' });
      await new Promise(r => setTimeout(r, 400));

      // body.scrollHeight 会被视口撑高，改量最后一个子元素的底部，得到真实内容高度
      const height = await page.evaluate(() => {
        const body = document.body;
        let maxBottom = body.getBoundingClientRect().top;
        for (const child of body.children) {
          maxBottom = Math.max(maxBottom, child.getBoundingClientRect().bottom);
        }
        const padBottom = parseFloat(getComputedStyle(body).paddingBottom) || 0;
        return Math.ceil(maxBottom + padBottom);
      });

      await page.setViewport({ width, height, deviceScaleFactor: 2 });
      await new Promise(r => setTimeout(r, 250));

      const outPath = path.join(assetsDir, pngFile);
      await page.screenshot({ path: outPath, fullPage: true, type: 'png' });

      const kb = Math.round(fs.statSync(outPath).size / 1024);
      console.log(`✓ ${pngFile}  ${width}x${height} (2x)  ${kb} KB`);

      await page.close();
    }
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error.message || error);
  process.exit(1);
});
