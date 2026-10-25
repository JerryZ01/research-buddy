const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const projectRoot = path.resolve(__dirname, '..');
const diagramsDir = path.join(projectRoot, 'docs', 'diagrams');
const assetsDir = path.join(projectRoot, 'docs', 'assets');
const mmdc = path.join(
  projectRoot,
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'mmdc.cmd' : 'mmdc',
);

const diagrams = [
  ['architecture-overview.mmd', 'architecture-overview.svg'],
  ['research-workflow.mmd', 'research-workflow.svg'],
  ['data-boundaries.mmd', 'data-boundaries.svg'],
];

function findCachedChromium() {
  const configured = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
    || process.env.PUPPETEER_EXECUTABLE_PATH;
  if (configured && fs.existsSync(configured)) return configured;

  const roots = [
    path.join(os.homedir(), '.cache', 'puppeteer', 'chrome'),
    path.join(os.homedir(), '.cache', 'ms-playwright'),
  ];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    const versions = fs.readdirSync(root).sort().reverse();
    for (const version of versions) {
      const candidates = [
        path.join(root, version, 'chrome-linux64', 'chrome'),
        path.join(root, version, 'chrome-win64', 'chrome.exe'),
        path.join(root, version, 'chrome-mac', 'Chromium.app', 'Contents', 'MacOS', 'Chromium'),
        path.join(root, version, 'chrome-linux', 'chrome'),
      ];
      const executable = candidates.find(candidate => fs.existsSync(candidate));
      if (executable) return executable;
    }
  }
  return '';
}

if (!fs.existsSync(mmdc)) {
  throw new Error('Mermaid CLI 未安装，请先运行 npm install');
}

fs.mkdirSync(assetsDir, { recursive: true });
const executablePath = findCachedChromium();
const configPath = path.join(os.tmpdir(), `research-buddy-mermaid-${process.pid}.json`);
const argsFor = (input, output) => [
  '-i', path.join(diagramsDir, input),
  '-o', path.join(assetsDir, output),
  '-b', 'transparent',
  '-t', 'neutral',
  '-w', '1800',
  ...(executablePath ? ['-p', configPath] : []),
];

try {
  if (executablePath) {
    fs.writeFileSync(configPath, JSON.stringify({ executablePath }), 'utf8');
  }
  for (const [input, output] of diagrams) {
    const result = spawnSync(mmdc, argsFor(input, output), {
      cwd: projectRoot,
      encoding: 'utf8',
      stdio: 'inherit',
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
      throw new Error(`生成 ${output} 失败，退出码 ${result.status}`);
    }
  }
} finally {
  if (fs.existsSync(configPath)) fs.unlinkSync(configPath);
}
