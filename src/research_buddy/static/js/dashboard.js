/* ═══════════════════════════════════════════════
   Dashboard
   ═══════════════════════════════════════════════ */
async function loadDashboard() {
  try {
    const [topics, jobs] = await Promise.all([
      api('/topics'),
      api('/tracking/jobs'),
    ]);

    // Collect all reports across topics
    let allReports = [];
    for (const t of topics) {
      try {
        const reports = await api(`/topics/${t.id}/reports?limit=10000`);
        reports.forEach(r => { r.topic_name = t.name; r.topic_id = t.id; });
        allReports.push(...reports);
      } catch { /* skip */ }
    }

    // Sort by date, take latest 5
    allReports.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    const recent = allReports.slice(0, 5);

    const activeTracking = topics.filter(t => t.tracking_enabled).length;
    const totalReports = allReports.length;

    // Stats
    $('dashboardStats').innerHTML = `
      <div class="card stat-card tint-purple">
        <div class="stat-accent accent-purple"></div>
        <div class="stat-icon" aria-hidden="true">◫</div>
        <div class="stat-value text-purple2">${topics.length}</div>
        <div class="stat-label">研究主题</div>
      </div>
      <div class="card stat-card tint-cyan">
        <div class="stat-accent accent-cyan"></div>
        <div class="stat-icon" aria-hidden="true">▯</div>
        <div class="stat-value text-cyan">${totalReports}</div>
        <div class="stat-label">研究报告</div>
      </div>
      <div class="card stat-card tint-green">
        <div class="stat-accent accent-green"></div>
        <div class="stat-icon" aria-hidden="true">◉</div>
        <div class="stat-value text-green">${activeTracking}</div>
        <div class="stat-label">活跃追踪</div>
      </div>
      <div class="card stat-card tint-amber">
        <div class="stat-accent accent-amber"></div>
        <div class="stat-icon" aria-hidden="true">⬡</div>
        <div class="stat-value text-amber">${jobs.length}</div>
        <div class="stat-label">调度任务</div>
      </div>
    `;

    // Recent activity
    if (recent.length === 0) {
      $('activityList').innerHTML = `
        <div class="empty-state">
          <div class="empty-icon" aria-hidden="true">⬡</div>
          <div class="empty-text">还没有研究记录，开始你的第一次研究吧</div>
          <button class="btn btn-primary" onclick="navigate('research')">开始研究</button>
        </div>
      `;
    } else {
      $('activityList').innerHTML = recent.map(r => {
        const d = new Date(r.created_at).toLocaleString('zh-CN');
        const dotColor = r.is_incremental ? 'var(--cyan)' : 'var(--purple)';
        return `
          <div class="activity-item" onclick="navigate('report-detail',{reportId:'${r.id}'})">
            <div class="activity-dot" style="background:${dotColor}"></div>
            <div class="activity-info">
              <div class="activity-question">${esc(r.question)}</div>
              <div class="activity-meta">${esc(r.topic_name || '')} · ${d}</div>
            </div>
            <span class="badge ${r.is_incremental ? 'badge-cyan' : 'badge-accent'}">${r.is_incremental ? '增量' : '全新'}</span>
          </div>
        `;
      }).join('');
    }

    // Update sidebar badges
    $('topicCount').textContent = topics.length;
    $('trackingCount').textContent = jobs.length;
  } catch (e) {
    $('dashboardStats').innerHTML = `
      <div class="empty-state" style="grid-column:1/-1;padding:30px">
        <div class="empty-icon" aria-hidden="true">⚠</div>
        <div class="empty-text">加载失败: ${esc(e.message)}</div>
      </div>
    `;
  }
}

