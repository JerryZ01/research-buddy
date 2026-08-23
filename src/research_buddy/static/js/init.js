/* ═══════════════════════════════════════════════
   Init
   ═══════════════════════════════════════════════ */
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
}

init();