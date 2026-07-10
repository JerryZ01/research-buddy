# 领域知识库 + 持续追踪 — 设计分析报告

## 一、核心问题定义

Research Buddy 目前是一个**一次性研究工具**：输入问题 → 搜索 → 生成报告 → 结束。每次研究都是独立的，没有记忆，无法积累知识。

我们想要演化为：**持久化领域知识助手**，核心能力：

| 能力 | 说明 | 用户价值 |
|------|------|----------|
| **知识持久化** | 研究报告、来源、关键事实存入本地知识库 | 不重复研究，历史可追溯 |
| **增量研究** | 基于已有知识，只搜索新信息 | 省时省力（省 API 调用） |
| **定时追踪** | 按计划自动监控领域变化 | 不遗漏重要更新 |
| **变化检测** | 对比新旧信息，标记变更 | 快速掌握动态 |
| **智能通知** | 重要变化时推送通知 | 被动获取，不用主动查 |

---

## 二、现有成熟方案调研

### 2.1 记忆/持久化方案

| 方案 | Stars | 核心思路 | 适合度 |
|------|-------|----------|--------|
| **Mem0** | 60K+ | 通用 AI Agent 记忆层，支持短期/长期/情景记忆，向量+图存储 | ⭐⭐⭐ — 功能全面但较重，适合需要精细记忆管理的场景 |
| **Letta (MemGPT)** | 23K+ | 有状态 Agent 平台，LLM 自主管理记忆（核心/归档记忆），支持无限上下文 | ⭐⭐ — 架构优秀但引入全新 Agent 框架，与 LangGraph 耦合度低 |
| **LangGraph Store** | 内置 | LangGraph 0.2+ 的 BaseStore，跨线程键值存储 | ⭐⭐⭐⭐⭐ — 原生集成，零额外依赖 |
| **Zep** | 2K+ | 专注对话记忆，知识图谱+时序记忆 | ⭐⭐ — 偏对话场景，与我们的研究场景不太匹配 |

**推荐**：**LangGraph Store（主）+ SQLite（辅）**
- LangGraph Store 做跨研究会的知识共享
- SQLite 做结构化数据持久化（报告、来源、追踪配置）
- 不引入额外框架，保持简洁

### 2.2 向量数据库（知识检索）

| 方案 | 类型 | 特点 | 适合度 |
|------|------|------|--------|
| **ChromaDB** | 嵌入式 | 纯 Python，零配置，自带 embedding，开发体验极好 | ⭐⭐⭐⭐⭐ — 最适合个人项目 |
| **LanceDB** | 嵌入式 | Rust 内核，性能好，支持增量索引 | ⭐⭐⭐⭐ — 性能更好但社区小 |
| **sqlite-vec** | SQLite 扩展 | 最轻量，与 SQLite 一体 | ⭐⭐⭐ — 功能有限但最简单 |
| **Qdrant** | 独立服务 | 功能丰富，支持过滤 | ⭐⭐ — 对个人项目太重 |

**推荐**：**ChromaDB**
- 嵌入式运行（无需服务端），与项目轻量定位一致
- 自带 embedding 函数，不额外配模型
- 持久化到本地目录，增量索引
- 与 LangChain/LlamaIndex 均有良好集成

### 2.3 知识图谱方案

| 方案 | 核心思路 | 适合度 |
|------|----------|--------|
| **Microsoft GraphRAG** | 从文档自动抽取实体和关系，构建社区层次图，支持全局摘要 | ⭐⭐⭐ — 效果好但开销大（LLM 调用量大），适合大规模文档 |
| **LightRAG** | 简化版 GraphRAG，轻量图结构，增量更新 | ⭐⭐⭐⭐ — 更适合我们的增量场景 |
| **Neo4j** | 专业图数据库 | ⭐⭐ — 太重 |

**推荐**：**暂不引入知识图谱**
- 第一阶段用向量检索 + 结构化元数据足够
- 后期如需实体关系追踪，可考虑 LightRAG
- 避免 over-engineering

### 2.4 深度研究产品参考

| 产品 | 特点 | 可借鉴之处 |
|------|------|------------|
| **Perplexity Deep Research** | 多步搜索+综合，支持 Collections（持久化保存研究） | Collections 概念 — 按主题归档研究报告 |
| **OpenAI Deep Research** | 自主搜索数十个网站，综合分析 | 搜索深度 + 来源丰富度 |
| **Gemini Deep Research** | Google 生态，访问 Google Scholar 等 | 学术资源整合 |
| **Stanford STORM** | 开源，Wikipedia 式文章生成，多视角 | 开源实现参考，文章结构化 |

**关键借鉴**：
- Perplexity 的 **Collections** 概念：将研究按主题分组，支持后续追加
- STORM 的 **增量写作** 模式：先搜后写，可追加新章节

### 2.5 变化检测方案

| 方案 | 类型 | 特点 |
|------|------|------|
| **changedetection.io** | 开源 | Docker 自托管，CSS 选择器监控，Web UI，通知集成 |
| **Huginn** | 开源 | IFTTT 开源替代，Agent 模式，可编程 |
| **文本 diff** | 自实现 | Python `difflib`，简单高效 |
| **LLM 语义 diff** | 自实现 | LLM 对比新旧内容，输出人可读的变化摘要 |

**推荐**：**LLM 语义 diff + 文本 diff**
- 不爬网页（避免反爬和法律风险），利用 Tavily API 重新搜索
- 用 LLM 对比新旧搜索结果的语义差异
- `difflib` 做基础文本 diff 作为快速筛选

### 2.6 定时调度方案

| 方案 | 特点 | 适合度 |
|------|------|--------|
| **APScheduler** | 纯 Python，支持 cron/interval/date，可持久化 | ⭐⭐⭐⭐⭐ — 最适合 |
| **Celery Beat** | 分布式，需 Redis/RabbitMQ | ⭐⭐ — 太重 |
| **系统 crontab** | 最简单，但与 Python 集成差 | ⭐⭐⭐ — 备选 |

**推荐**：**APScheduler**
- 与 FastAPI 集成自然（可在 startup 事件中启动 scheduler）
- 支持 cron 表达式（如每天 9 点检查）
- 支持持久化（SQLite job store）
- 代码量少

### 2.7 通知方案

| 方案 | 特点 | 适合度 |
|------|------|--------|
| **企业微信机器人** | Webhook，中国用户友好 | ⭐⭐⭐⭐⭐ |
| **钉钉机器人** | Webhook，中国用户友好 | ⭐⭐⭐⭐⭐ |
| **Telegram Bot** | Webhook，国际用户 | ⭐⭐⭐⭐ |
| **Email** | 通用，smtplib | ⭐⭐⭐ |
| **Apprise** | 统一通知库，支持 100+ 服务 | ⭐⭐⭐⭐ |

**推荐**：**先实现 Webhook 通知 + Apprise 集成**
- Webhook 最通用（企业微信/钉钉/Telegram 都是 webhook）
- Apprise 做统一接口，用户配置 URL 即可
- 后期可加邮件

---

## 三、架构设计

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│                   Research Buddy v2                    │
│                                                        │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Web UI  │───→│ FastAPI  │───→│  LangGraph       │  │
│  │         │←───│ API 层   │←───│  Research Graph  │  │
│  └─────────┘    └────┬─────┘    └───────┬──────────┘  │
│                      │                   │             │
│              ┌───────┴──────┐   ┌────────┴────────┐   │
│              │ APScheduler  │   │ LangGraph Store  │   │
│              │ 定时追踪调度  │   │ 跨线程知识共享    │   │
│              └───────┬──────┘   └────────┬────────┘   │
│                      │                   │             │
│  ┌───────────────────┴───────────────────┴──────────┐  │
│  │              Knowledge Layer (知识层)              │  │
│  │                                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │  │
│  │  │ SQLite   │  │ChromaDB  │  │ File Store     │  │  │
│  │  │ 结构数据 │  │ 向量检索 │  │ 报告 Markdown  │  │  │
│  │  └──────────┘  └──────────┘  └────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Notification Layer (通知层)           │  │
│  │  Webhook / Apprise → 企业微信 / 钉钉 / Telegram   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 数据模型

```sql
-- 研究主题（类似 Perplexity 的 Collection）
CREATE TABLE topics (
    id TEXT PRIMARY KEY,           -- UUID
    name TEXT NOT NULL,            -- 主题名称
    description TEXT,              -- 主题描述
    tracking_keywords TEXT,        -- 追踪关键词（JSON 数组）
    tracking_cron TEXT,            -- 追踪频率（cron 表达式），如 "0 9 * * 1-5"
    tracking_enabled INTEGER DEFAULT 0, -- 是否启用追踪
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 研究报告
CREATE TABLE reports (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,        -- 关联主题
    question TEXT NOT NULL,        -- 原始问题
    report TEXT NOT NULL,          -- Markdown 报告
    confidence TEXT,               -- 置信度（高/中/低）
    sources TEXT,                  -- 来源列表（JSON）
    search_results_count INTEGER, -- 搜索结果数
    reflection_rounds INTEGER,    -- 反思轮次
    is_incremental INTEGER DEFAULT 0, -- 是否增量研究报告
    parent_report_id TEXT,        -- 基于哪份报告的增量
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

-- 追踪记录
CREATE TABLE tracking_logs (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'running',  -- running/completed/failed
    changes_detected INTEGER DEFAULT 0,
    change_summary TEXT,           -- 变更摘要（LLM 生成）
    report_id TEXT,                -- 如有变化，关联的新报告
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

-- 变更条目
CREATE TABLE changes (
    id TEXT PRIMARY KEY,
    tracking_log_id TEXT NOT NULL,
    change_type TEXT,              -- new_info/contradiction/update/removal
    description TEXT,              -- 变更描述
    old_content TEXT,              -- 旧内容
    new_content TEXT,              -- 新内容
    significance TEXT,             -- high/medium/low
    FOREIGN KEY (tracking_log_id) REFERENCES tracking_logs(id)
);
```

### 3.3 LangGraph 工作流扩展

```
现有流程（一次性研究）：
  question → planner → searcher → validator → synthesizer → reflector → END

新增流程（增量研究）：
  question → knowledge_lookup → planner* → searcher* → validator →
             diff_analyzer → synthesizer* → reflector → END
                        ↓
               [knowledge_store]

新增流程（定时追踪）：
  APScheduler trigger → topic_loader → searcher → diff_analyzer →
                         change_notifier → END
                              ↓ (如有变化)
                         [knowledge_store + notification]
```

新增节点：

| 节点 | 职责 |
|------|------|
| **knowledge_lookup** | 查询 ChromaDB + SQLite，找相关历史报告和关键事实 |
| **diff_analyzer** | 对比新旧搜索结果，用 LLM 识别语义变化 |
| **change_notifier** | 检测到重要变化时，通过 Webhook 推送通知 |
| **knowledge_store** | 将新报告、来源、关键事实写入知识库 |

修改节点：

| 节点 | 修改 |
|------|------|
| **planner** | 增量模式：基于已有知识生成补充搜索词，而非完全从零规划 |
| **searcher** | 增量模式：只搜索新信息，跳过已知信息 |
| **synthesizer** | 增量模式：在已有报告基础上补充/修正，而非从头写 |

### 3.4 增量研究核心逻辑

```
用户问 "中国新能源汽车市场最新进展"（第二次问）

1. knowledge_lookup:
   - 找到历史报告（上次研究了截至 2025 Q2 的情况）
   - 提取关键事实：比亚迪销量 300 万、渗透率 35%...
   - 提取已搜索来源

2. planner（增量模式）:
   - 知道已有信息，规划 "2025 Q3 新能源汽车 新政策/销量"
   - 只生成增量搜索词，不重复搜索已知信息

3. searcher（增量模式）:
   - 只搜索增量关键词
   - 可能只需 2-3 次搜索（而非 4-5 次）

4. diff_analyzer:
   - 对比新搜索结果与已知事实
   - 识别变化：渗透率 35% → 40%、新政策出台等
   - 生成变更列表

5. synthesizer（增量模式）:
   - 输出：增量报告（只包含变化和新增内容）
   - 附带完整的更新后报告
```

### 3.5 定时追踪核心逻辑

```
用户创建了主题 "中国新能源汽车" 并开启追踪（每天 9 点）

1. APScheduler 每天 9 点触发追踪任务
2. topic_loader: 加载主题配置、关键词、最近报告
3. searcher: 用追踪关键词搜索最新信息
4. diff_analyzer: 对比搜索结果与已有知识
5. 判断是否有变化：
   - 无变化 → 记录 tracking_log（no change）→ END
   - 有变化 → 生成变更摘要 → 保存 → 发通知 → END

通知内容示例：
  🔔 [Research Buddy] 中国新能源汽车 追踪更新
  检测到 2 项重要变化：
  1. 📈 渗透率从 35% 升至 40%（来源：xxx）
  2. 📋 新政策：以旧换新补贴延续（来源：xxx）
  查看完整报告：http://localhost:8000/reports/xxx
```

---

## 四、技术选型总结

| 组件 | 选型 | 理由 |
|------|------|------|
| 结构化存储 | **SQLite** | 零配置、Python 内置、单文件、够用 |
| 向量检索 | **ChromaDB** | 嵌入式、自带 embedding、增量索引、LangChain 集成 |
| 跨线程知识共享 | **LangGraph Store** | 原生集成、零额外依赖 |
| 定时调度 | **APScheduler** | 纯 Python、cron 支持、与 FastAPI 集成简单 |
| 变化检测 | **LLM 语义 diff + difflib** | 精准检测语义变化、避免爬虫 |
| 通知推送 | **Apprise + Webhook** | 统一接口、支持 100+ 服务 |
| 新依赖 | chromadb, apscheduler, apprise | 总共 3 个新依赖 |

---

## 五、分阶段实施路线

### Phase 6：知识持久化 + 增量研究

**目标**：研究不再是"一次性"的，历史可追溯、可增量

- [ ] SQLite 数据模型（topics, reports）
- [ ] Knowledge Layer：SQLite + ChromaDB 初始化
- [ ] knowledge_lookup 节点：查历史报告、提取关键事实
- [ ] knowledge_store 节点：保存报告到 SQLite、chunk 到 ChromaDB
- [ ] planner 增量模式：基于已有知识规划补充搜索
- [ ] synthesizer 增量模式：在已有报告基础上补充/修正
- [ ] API 新增：`POST /topics`、`GET /topics/{id}/reports`、`POST /research/incremental`
- [ ] Web UI：主题列表页、报告历史页、增量研究按钮

**学到**：ChromaDB、向量检索、增量 RAG、数据建模

### Phase 7：定时追踪 + 变化检测

**目标**：自动监控领域变化，不遗漏重要更新

- [ ] APScheduler 集成：FastAPI startup 时启动 scheduler
- [ ] tracking 配置：主题级 cron 表达式、关键词
- [ ] 追踪工作流：topic_loader → searcher → diff_analyzer
- [ ] diff_analyzer 节点：LLM 对比新旧搜索结果，识别语义变化
- [ ] changes 表 + tracking_logs 表
- [ ] API 新增：`PUT /topics/{id}/tracking`、`GET /topics/{id}/changes`
- [ ] Web UI：追踪配置、变更时间线

**学到**：APScheduler、变化检测算法、LLM 语义对比

### Phase 8：智能通知 + 生产优化

**目标**：变化主动推送，系统稳定可依赖

- [ ] Apprise 通知集成：Webhook 统一接口
- [ ] 企业微信/钉钉机器人通知模板
- [ ] 通知频率控制：不频繁打扰（同一主题每天最多 1 次通知）
- [ ] change_notifier 节点：判断重要性 → 推送通知
- [ ] 通知配置：环境变量或 UI 配置
- [ ] Langfuse 追踪评估：增量研究质量 vs 全新研究质量
- [ ] 性能优化：ChromaDB 增量索引、搜索结果去重

**学到**：通知系统设计、频率控制、生产优化

---

## 六、目录结构扩展

```
src/research_buddy/
├── knowledge/              # 新增：知识层
│   ├── __init__.py
│   ├── db.py               # SQLite 数据模型和操作
│   ├── vector.py            # ChromaDB 向量存储和检索
│   └── store.py             # 统一知识层接口
├── tracking/                # 新增：追踪层
│   ├── __init__.py
│   ├── scheduler.py         # APScheduler 调度器
│   ├── diff.py              # 变化检测（LLM 语义 diff + difflib）
│   └── notifier.py          # 通知推送（Apprise + Webhook）
├── nodes/                   # 扩展节点
│   ├── planner.py           # 修改：支持增量模式
│   ├── searcher.py          # 修改：支持增量搜索
│   ├── synthesizer.py       # 修改：支持增量综合
│   ├── knowledge_lookup.py  # 新增：知识查询节点
│   ├── knowledge_store.py   # 新增：知识存储节点
│   ├── diff_analyzer.py     # 新增：变化分析节点
│   └── change_notifier.py   # 新增：变化通知节点
├── state.py                 # 扩展 State
├── graph.py                 # 扩展工作流
└── ...
```

---

## 七、关键设计决策及理由

| 决策 | 选择 | 理由 |
|------|------|------|
| 为什么不爬网页做 diff？ | 用 Tavily 重新搜索 | 避免反爬风险、法律问题、维护成本；Tavily 返回的摘要已足够做语义对比 |
| 为什么不用 Mem0/Letta？ | LangGraph Store + 自建知识层 | 不引入新框架，保持技术栈一致性；我们的需求更偏"文档级知识"而非"对话记忆" |
| 为什么不用知识图谱？ | 向量检索 + 结构化元数据 | 研究报告是长文档而非短三元组；知识图谱的增量维护成本高；后期需要再加 |
| 为什么 ChromaDB 而非 Qdrant？ | 嵌入式、零配置 | 个人项目不需要分布式向量库；ChromaDB 的开发体验最好 |
| 为什么 APScheduler 而非 Celery？ | 简单、纯 Python | 单机部署不需要分布式队列；APScheduler 与 FastAPI 集成自然 |
| 增量报告 vs 全新报告？ | 两者都生成 | 增量报告快速了解变化；完整报告保证完整性 |

---

## 八、风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| Tavily API 调用量增加（追踪会定期调用） | 成本上升 | 追踪模式用 `search_depth="basic"`（更便宜），频率可配 |
| ChromaDB 数据膨胀 | 磁盘占用 | 定期清理过期 chunk，设置 TTL |
| LLM 变化检测误报 | 通知疲劳 | 设置变化重要性阈值，只通知 high/medium |
| 增量研究质量不如全新研究 | 报告质量下降 | Langfuse 评估增量 vs 全新报告质量，持续优化 prompt |
| APScheduler 与 FastAPI 生命周期冲突 | 进程管理 | 使用 FastAPI lifespan 事件管理 scheduler |

---

## 九、与竞品的核心差异化

| 竞品 | 缺少什么 | 我们的独特价值 |
|------|----------|----------------|
| Perplexity Deep Research | 无持续追踪、无本地知识库 | 本地持久化 + 定时追踪 + 变化检测 |
| OpenAI Deep Research | 无增量研究、无自定义追踪 | 增量研究省 API + 可配置追踪频率 |
| STORM | 无持续追踪、无变化检测 | 追踪 + 通知 + 增量更新 |
| changedetection.io | 无语义理解、纯文本 diff | LLM 语义 diff，理解变化含义 |

**核心定位**：不是一个更好的搜索引擎，而是一个**能持续追踪领域变化的 AI 研究助手**。
