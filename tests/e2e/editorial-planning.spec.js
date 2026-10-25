const { test, expect } = require('playwright/test');

const editorialDetail = {
  evidence_count: 18,
  section_count: 4,
  must_cover_count: 7,
  intent: 'mechanism',
  depth: 'deep',
  audience: '已经了解 Agent 基础概念、希望理解反思反馈如何影响下一轮执行的技术读者',
  thesis: 'Agent 的反思机制只有在评价结果能够转化为下一轮可执行的状态变化时才真正有效，单独增加一次模型自评并不会自然提高最终质量。',
  sections: [
    { heading: '反思发生在执行链路的什么位置', purpose: '明确生成、评价、状态更新和再次执行之间的关系，避免把反思误解为普通的提示词重写。', depth: 'deep', completion_count: 2 },
    { heading: '评价结果如何变成下一轮输入', purpose: '解释结构化反馈、失败原因和证据缺口分别进入状态后会触发哪些不同路径。', depth: 'deep', completion_count: 3 },
    { heading: '多轮反思为什么可能让结果变差', purpose: '说明同模型自评偏差、信息损失与反复重写之间的边界条件。', depth: 'standard', completion_count: 2 },
    { heading: '怎样判断反思已经足够', purpose: '给出质量门槛、预算和历史最佳稿恢复之间的停止条件。', depth: 'standard', completion_count: 2 },
  ],
  degraded: false,
};

test.beforeEach(async ({ page }) => {
  await page.route(/^https?:\/\/(?!127\.0\.0\.1:4173).*/, route => route.abort());
  await page.goto('/index.html');
  await page.evaluate(detail => {
    navigate('research');
    appState.activeSteps = CORE_RESEARCH_STEPS;
    resetNodePanels();
    showNodeDetail('editorial_planner', detail);
    document.getElementById('executionDetails').open = true;
  }, editorialDetail);
});

test('编辑规划长文本保持正常横向排版', async ({ page }, testInfo) => {
  const card = page.locator('.step-card').filter({ hasText: '编辑规划' });
  const thesis = card.locator('.nd-editorial-thesis p');
  const sections = card.locator('.nd-editorial-section');

  await expect(card).toBeVisible();
  await expect(thesis).toContainText('状态变化');
  await expect(sections).toHaveCount(4);

  const geometry = await thesis.evaluate(element => {
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return { width: rect.width, height: rect.height, whiteSpace: style.whiteSpace };
  });
  expect(geometry.width).toBeGreaterThan(180);
  expect(geometry.height).toBeLessThan(130);
  expect(geometry.whiteSpace).toBe('normal');

  for (const section of await sections.all()) {
    const box = await section.boundingBox();
    expect(box && box.width).toBeGreaterThan(180);
  }

  await card.screenshot({ path: testInfo.outputPath('editorial-planning.png') });
});
