const fs = require('fs');
const os = require('os');
const path = require('path');
const { defineConfig, devices } = require('playwright/test');

function findCachedChromium() {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }
  const root = path.join(os.homedir(), '.cache', 'puppeteer', 'chrome');
  if (!fs.existsSync(root)) return undefined;
  const versions = fs.readdirSync(root).sort().reverse();
  for (const version of versions) {
    const executable = path.join(root, version, 'chrome-linux64', 'chrome');
    if (fs.existsSync(executable)) return executable;
  }
  return undefined;
}

const executablePath = findCachedChromium();

module.exports = defineConfig({
  testDir: './tests/e2e',
  outputDir: 'test-results',
  timeout: 30_000,
  fullyParallel: true,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: true,
    launchOptions: executablePath ? { executablePath } : {},
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'python -m http.server 4173 --bind 127.0.0.1 --directory src/research_buddy/static',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: true,
    timeout: 30_000,
  },
  projects: [
    { name: 'desktop-chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile-chromium', use: { ...devices['Pixel 7'] } },
  ],
});
