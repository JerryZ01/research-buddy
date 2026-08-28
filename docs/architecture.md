# Research Buddy 架构全解

> 基于 LangGraph + Langfuse 的深度研究 Agent：输入一个问题 → 自动拆解子问题 → 并行搜索 → 证据验证 → 流式综合报告 → LLM 自评循环修正。
>
> 本文描述**当前代码真实行为**（含自适应研究系统）。旧版纯规则 validator / 任意判 pass 的 reflector 已被替换为证据驱动 + fail-closed 的混合评估。

---

## 一、项目概览

### 1.1 核心能力

| 能力 | 说明 |
|------|------|
| 深度研究 | 输入问题 → 拆解 1-6 个子问题 → 并行搜索 → 证据验证 → 流式生成**可发布文章**（正文无编号引用，文末核心参考文献由 LLM 筛选 + 代码重编号） |
| 自适应研究 | 稳定 `sub_question_id` 追溯、自适应搜索语言、URL+内容指纹双去重、混合证据充足性判断 |
| 循环修正 | LLM 自评报告质量 + 引用一致性校验 → 不足则补搜或重写 → 搜索/反思双预算闸防无限循环 |
| 人机交互 | 规划后暂停让用户编辑子问题，综合后暂停让用户补充要求 |
| 增量研究 | 基于历史知识只搜索新信息，复用已有知识，`parent_report_id` 形成增量链 |
| 定时追踪 | 按 cron 定期重新研究，对比新旧报告检测变化，按策略推送通知 |
| 可观测性 | Langfuse 全链路 Trace/Span/Generation，LLM-as-Judge 自动评分，Prompt 版本管理 |

### 1.2 技术栈

| 层 | 技术 | 作用 |
|----|------|------|
| 工作流编排 | LangGraph ≥ 0.2 | StateGraph + Node + Conditional Edge + Loop + interrupt |
| LLM | langchain-openai ≥ 0.3 | ChatOpenAI（通过中转站 API，temperature=0） |
| 搜索 | tavily-python ≥ 0.5 | Tavily 搜索 API（basic/advanced 两种深度） |
| 可观测性 | langfuse ≥ 2.0 | Trace/Span/Generation、Score、Prompt 管理、Dataset |
| 向量检索 | chromadb ≥ 1.5 | 报告分块 + 关键事实的语义检索（Embedding 后端可配置） |
| 持久化 | SQLite | 主题、报告、追踪记录、变更条目（WAL + 线程本地连接） |
| 定时调度 | APScheduler ≥ 3.11 | AsyncIOScheduler + CronTrigger |
| API 层 | FastAPI + sse-starlette | HTTP + SSE 流式 + 静态文件 |
| 包管理 | uv + hatchling | 依赖管理 + 虚拟环境 |

### 1.3 目录结构

```
research-buddy/
├── CLAUDE.md                    # 项目说明 + 路线图
├── pyproject.toml               # 依赖管理（uv + hatchling）
├── .env.example                 # 环境变量模板
├── Dockerfile                   # Docker 部署
├── src/
│   └── research_buddy/
│       ├── __init__.py          # 包入口 + 日志配置 + CLI main()
│       ├── config.py            # 环境变量配置（load_dotenv）
│       ├── state.py             # ResearchState TypedDict + 子类型定义
│       ├── utils.py             # 共享工具（parse_llm_json, create_llm, ...）
│       ├── graph.py             # LangGraph 工作流（4 种图 + 3 路由函数 + 4 运行函数）
│       ├── api.py               # FastAPI 应用（HTTP + SSE + HITL + 静态文件）
│       ├── nodes/               # LangGraph 节点实现（9 个）
│       │   ├── planner.py       # 规划节点（稳定 ID + 自适应语言）
│       │   ├── searcher.py      # 搜索节点（并行 + 双去重 + 补搜）
│       │   ├── validator.py     # 验证节点（确定性 + LLM 混合评估）
│       │   ├── synthesizer.py   # 综合节点（流式 + 三模式）
│       │   ├── reflector.py     # 反思节点（自评 + 引用校验 + fail-closed）
│       │   ├── knowledge_lookup.py  # 知识查询节点
│       │   ├── knowledge_store.py   # 知识存储节点（双写 + 关键事实提取）
│       │   ├── diff_analyzer.py     # 变化分析节点
│       │   └── change_notifier.py   # 变化通知节点
│       ├── tools/
│       │   └── search.py        # Tavily API 搜索（懒初始化单例）
│       ├── knowledge/           # 知识层（SQLite + ChromaDB）
│       │   ├── db.py            # SQLite 数据库（线程本地连接）
│       │   ├── store.py         # KnowledgeStore 统一门面
│       │   └── vector.py        # ChromaDB 向量存储
│       ├── tracking/            # 追踪层
│       │   ├── scheduler.py     # APScheduler 定时调度器
│       │   ├── notifier.py      # 多平台 Webhook 通知
│       │   └── diff.py          # 文本差异分析（difflib + LLM）
│       ├── eval/                # 评估层
│       │   ├── dataset.py       # Langfuse 测试数据集（8 题）
│       │   ├── judge.py         # LLM-as-Judge 评分
│       │   └── prompts.py       # Prompt 版本管理（9 个 prompt）
│       └── static/
│           ├── index.html       # Web UI（单页应用）
│           └── app.css         # 视觉系统（浅色主体 + 深色控制台）
├── tests/                       # 13 个测试文件
├── scripts/                     # 启动/评估/分阶段测试脚本
└── docs/
    ├── architecture.md          # 本文件
    ├── project-analysis-memory.md  # 项目分析记忆
    └── learning-notes.md        # 学习笔记
```

---

## 二、核心架构：LangGraph 工作流

### 2.1 状态定义（state.py）

`ResearchState`（state.py:53）是所有图共享的 TypedDict，节点返回 dict 由 LangGraph 通道 reducer 自动 merge。

```python
class ResearchState(TypedDict):
    # 输入
    question: str

    # 知识层
    topic_id: str
    knowledge_context: str
    has_knowledge: bool
    is_incremental: bool
    known_source_urls: list[str]          # 增量去重用
    key_facts: Annotated[list[str], operator.add]
    saved_report_id: str

    # 规划阶段
    sub_questions: list[SubQuestion]      # 覆盖语义（HITL 可编辑替换）

    # 搜索阶段
    search_results: Annotated[list[SearchResult], operator.add]
    validation_gaps: list[ValidationGap]   # 覆盖语义（搜索后可清空）
    evidence_assessments: list[EvidenceAssessment]  # 覆盖语义
    evidence_assessment_degraded: bool     # 语义评估不可用，报告需披露
    search_history: Annotated[list[dict], operator.add]
    search_round: int
    total_queries: int
    stop_reason: str                       # 跨节点状态机信号
    research_complete: bool
    search_unavailable: bool               # 搜索层不可用（无 key 或全部失败）

    # 综合阶段
    report: str                     # 最终报告（可发布文章正文：无评价性内容、无内嵌 URL）
    confidence: str                 # 置信度（高/中/低），代码从证据质量计算，不进正文
    research_notes: list[str]       # 研究说明（局限/降级/未解决缺口），覆盖语义，不进正文
    source_table: list[dict]        # 编号引用表 [{index,title,url,source}]，覆盖语义，synthesizer 构建

    # 反思阶段
    reflection_pass: bool
    reflection_feedback: str
    reflection_round: int

    # Human-in-the-loop
    user_feedback: str

    # 追踪层
    detected_changes: Annotated[list[dict], operator.add]
    similarity: float
    tracking_log_id: str
    notification_sent: bool

    # 进度消息（供 API 层推送）
    messages: Annotated[list[str], operator.add]
```

**两类累积语义（核心设计点）**：

- **追加语义** (`Annotated[list, operator.add]`)：`key_facts` / `search_results` / `search_history` / `detected_changes` / `messages` —— 多节点产出自动 extend，累积证据链。
- **覆盖语义**（无 operator.add）：`sub_questions` / `validation_gaps` / `evidence_assessments` / `research_notes` / `source_table` —— 便于 HITL 编辑替换 planner 输出、searcher 清空已处理缺口、validator/reflector 重写评估、synthesizer 每轮全量重建研究说明与编号引用表。

> ⚠️ 这套覆盖/追加语义在代码中**镜像了 4 处**：state.py 的 `operator.add` 注解、`utils.stream_and_accumulate`（utils.py:135）、api.py 三个 SSE 生成器各内联一份。修改语义时需四处同步（见第十四章已知技术债）。

**子类型**（state.py:7-51）：

- `SubQuestion`：`id` / `question` / `search_query`（主查询）/ `search_queries`（多语言查询列表）/ `language` / `region` / `source_preference`
- `SearchResult`：`sub_question_id` / `sub_question` / `query` / `language` / `region` / `title` / `url` / `content` / `score`
- `ValidationGap`：`sub_question_id` / `question` / `search_query`（补搜词）/ `reason` / `priority` / `language` / `region`
- `EvidenceAssessment`：`sub_question_id` / `status` / `coverage` / `valid_results` / `distinct_domains` / `missing_evidence` / `contradictions`

### 2.2 四种工作流图

#### 图 1：核心研究图 `create_research_graph()` (graph.py:122)

无知识层，全自动。由 `_add_core_nodes_and_edges()` (graph.py:91) 装配。

```
 START → planner → searcher → validator
                              │
                     route_after_validation
                    ┌─────────┴──────────┐
                  有缺口              无缺口 / 预算耗尽
                    │                      │
                    ▼                      ▼
                 searcher ←(回环补搜)   editorial_planner → synthesizer
                                                                      │
                                                                      ▼
                                                     language_editor → article_editor → reflector
                                                                                  │
                                            should_continue
                              ┌──────────────┬──────────────┐
                       通过/预算/轮次      有缺口        无缺口
                          达上限          (search_again)   (revise_report)
                            │                │                │
                            ▼                ▼                ▼
                           END          searcher ←────  synthesizer ←(重写报告)
```

**两个循环回路**：
- **补搜回路**（validator 驱动）：validator 发现证据缺口 → 回 searcher 补搜。
- **修正回路**（reflector 驱动）：reflector 未通过且有缺口 → 回 searcher；未通过但无缺口 → 回 synthesizer 重写（`revise_report`）。

#### 图 2：知识研究图 `create_knowledge_research_graph()` (graph.py:129)

带知识层，支持增量研究。头加 `knowledge_lookup`，尾加 `knowledge_store`。

```
 START → knowledge_lookup → planner → searcher → validator
                                      (同上的补搜/修正循环)
                                        │
                                  reflector
                                        │ should_continue_to_store
                              ┌─────────┬──────────┐
                          通过/预算     有缺口       无缺口
                          /轮次达上限  (search_again) (revise_report)
                              │           │            │
                              ▼           ▼            ▼
                       knowledge_store ←────  searcher ←── synthesizer
                              │
                              ▼
                             END
```

与图 1 唯一结构差异：reflector 用 `should_continue_to_store` 替代 `should_continue`，**通过时流向 `knowledge_store` 而非 END**。

#### 图 3：HITL 研究图 `create_research_graph_with_hitl()` (graph.py:174)

拓扑同图 1，但编译时注入：

```python
graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["searcher", "reflector"],
)
```

```
 START → planner ⏸(暂停让用户确认/编辑子问题) → searcher → validator → synthesizer
                                                                      │
                                                          ⏸(暂停让用户审阅报告/补充要求)
                                                                      │
                                                                reflector → END
```

- 在 `searcher` 前暂停：展示子问题，用户可编辑（覆盖语义替换 `sub_questions`）。
- 在 `reflector` 前暂停：展示报告，用户可输入 `user_feedback`（reflector 会强制不通过以优先处理）。
- 用 `MemorySaver` 保存中断态，`Command(resume=...)` 恢复执行。

#### 图 4：追踪图 `create_tracking_graph()` (graph.py:245)

在图 2 基础上，`knowledge_store` 之后追加追踪链。

```
 START → knowledge_lookup → planner → searcher → validator
                                      (同上的补搜/修正循环)
                                        │
                                  reflector
                                        │ should_continue_to_store
                                  ┌─────┬──────────┐
                              通过/预算  有缺口       无缺口
                              /轮次上限  (search_again) (revise_report)
                                  │       │            │
                                  ▼       ▼            ▼
                          knowledge_store ←── searcher ← synthesizer
                                  │
                                  ▼
                          diff_analyzer  (对比新旧报告，两层变化检测)
                                  │
                                  ▼
                          change_notifier  (按策略发 Webhook 通知)
                                  │
                                  ▼
                                 END
```

### 2.3 三个路由函数

| 路由函数 | 位置 | 触发节点 | 分支 |
|---------|------|---------|------|
| `route_after_validation` | graph.py:81 | validator | `synthesize` / `search_again` |
| `should_continue` | graph.py:45 | reflector（图 1） | `end` / `search_again` / `revise_report` |
| `should_continue_to_store` | graph.py:63 | reflector（图 2/4） | `knowledge_store` / `search_again` / `revise_report` |

**`route_after_validation` 逻辑**（graph.py:83-88）：

```
有缺口 且 stop_reason ∉ {search_budget_exhausted, no_new_queries, search_unavailable} → search_again（补搜）
否则 → synthesize（无缺口，或补搜已无路可走则直接综合）
```

**`should_continue` / `should_continue_to_store` 逻辑**（终止条件优先级）：

```
1. reflection_pass == True              → end / knowledge_store（通过）
2. search_round >= MAX_SEARCH_ROUNDS(4) → end / knowledge_store（搜索轮次预算耗尽）
   或 total_queries >= MAX_TOTAL_QUERIES(30)（查询总数预算耗尽）
3. reflection_round >= MAX_REFLECTION_ROUNDS(2) → end / knowledge_store（反思轮次达上限）
4. search_unavailable == True            → revise_report（补搜必然再失败，只能改写）
5. validation_gaps 为空                  → revise_report（回 synthesizer 重写）
6. 否则（有缺口）                         → search_again（回 searcher 补搜）
```

### 2.4 节点详解

#### planner（planner.py:72）—— 规划

**职责**：拆解 1-6 个子问题，每个带 `search_queries`（可含中英双语查询 + 地区偏好），由代码生成稳定 `sub_question_id`（`sq_01`…）。

**两种模式**：
- **全新模式**：`PLANNER_PROMPT`，正常拆解。
- **增量模式**：`INCREMENTAL_PLANNER_PROMPT`，基于已有 `knowledge_context` 只规划缺失的 2-3 个子问题。

**输入**：`question` / `is_incremental` / `has_knowledge` / `knowledge_context`
**输出**：`sub_questions`（覆盖）/ `messages`

**关键设计**：`sub_question_id` 由代码生成而非依赖中文文本——后续搜索任务/结果/证据评估/缺口全部挂在该稳定 ID 上，补搜结果绝不靠可变显示文本归属（planner.py:112-155）。

#### searcher（searcher.py:48）—— 搜索

**职责**：并行搜索子问题，支持补搜（来自 validator/reflector 的 `validation_gaps`）；增量模式下去重已知来源 URL；同轮及历史轮次不重复执行相同查询。**不调用 LLM。**

**输入**：`sub_questions` / `validation_gaps` / `search_results` / `search_history` / `search_round` / `is_incremental` / `known_source_urls` / `has_knowledge` / `knowledge_context` / `total_queries`
**输出**：`search_results`（追加）/ `validation_gaps`（清空）/ `search_history`（追加）/ `search_round`+1 / `total_queries` 累加（仅补搜查询）/ `stop_reason` / `search_unavailable` / `messages`

**流程**：

1. **任务合并**：`validation_gaps`（补搜）+ 未搜索的原始子问题（按 `search_queries` 展开为多查询）。
2. **查询去重**：对 `search_query` 归一化，跳过已在 `search_history` 或本轮重复的查询；第 2 轮起 high 优先级用 Tavily `advanced` 深度。
3. **预算计数**：`total_queries` **只累计补搜查询**（`reason != "initial"` 的任务）。初始轮基础搜索不占预算——planner 每个子问题会展开中英双语查询，一轮 8~12 个，若全部计入 `MAX_TOTAL_QUERIES=30`，补搜只能补 1~2 轮就触发 `search_budget_exhausted`，快变话题的覆盖度永远补不满。
4. **并行执行**：`ThreadPoolExecutor(max_workers=min(total, 4))`，`as_completed` 收集，逐个统计失败数。
5. **结果双去重**：`normalize_url()`（去 www/utm/fragment）+ 内容 sha256 指纹双重去重；增量模式合并 `known_source_urls`。

**搜索层失败处理**（`tools/search.search` 会抛 `SearchUnavailableError`，不再静默返回 `[]`）：

| 情况 | 行为 |
|------|------|
| 部分查询失败 | 继续，`messages` 里报 `N/M 个查询失败`，`search_unavailable=False` |
| 本轮全失败，但已有历史检索结果 | 继续，置 `search_unavailable=True` + `stop_reason=search_unavailable`，路由不再把预算耗在必然失败的补搜上 |
| 本轮全失败 + 零累计证据 + 有历史知识 | 降级：仅基于 `knowledge_context` 作答，置 `search_unavailable=True`，报告必须披露 |
| 本轮全失败 + 零累计证据 + 无历史知识 | **抛 `SearchUnavailableError` 中止**，API 发 SSE `error` 事件 |

> 最后一行是关键：早期实现下 Tavily 无 key 时 `search()` 返回 `[]`，全流程照跑，最后由 LLM 凭训练数据编出一份零来源却写着「整体置信度：高」的报告。

#### validator（validator.py:115）—— 证据验证

**职责**：按子问题聚合结果，用"确定性指标 + LLM 语义评估"双层判断证据是否充足，产出 `validation_gaps` 驱动补搜。

**输入**：`sub_questions` / `search_results` / `search_round` / `total_queries` / `question` / `stop_reason`
**输出**：`validation_gaps`（覆盖）/ `evidence_assessments`（覆盖）/ `stop_reason` / `research_complete` / `messages`

**第一层 — 确定性指标**（validator.py:134-179）：

- 按 `sub_question_id` 聚合，过滤短内容（`< MIN_SEARCH_CONTENT_LENGTH=80`）与重复 URL。
- 计算覆盖率：

```
relevance   = 正分搜索结果得分均值
count_score = min(1.0, valid_results / MIN_RESULTS_PER_SUB_QUESTION)
domain_score= min(1.0, distinct_domains / MIN_DISTINCT_DOMAINS)

有相关度分数：coverage = 0.5×count_score + 0.3×domain_score + 0.2×min(1.0, relevance)
无相关度分数：coverage = 0.5×count_score + 0.3×domain_score          # 上限 0.8
```

> 全部结果都没有 score 时，相关度是「未知」而不是「还行」。早期实现默认 `relevance=0.7`，白送 0.14 覆盖度，让无 score 的分支也能凑到 0.94。现在既不给默认值，也不把 0.2 的权重重新分摊（那等于按 `relevance=1` 算），直接丢掉该项：上限 0.8 仍高于默认阈值 0.75，数量与域名都达标的分支照样能过，但拿不到任何相关度加成。

- **硬底线** `hard_floor` = 三项**同时**满足：`valid_results ≥ 2` 且 `distinct_domains ≥ 2` 且 `coverage ≥ 0.75`。多独立域名计数即"交叉验证"的客观信号。

**第二层 — LLM 语义评估** `_llm_assess`：

- 把每个子问题的证据包（metrics + 最多 6 条证据内容）批量发给 `research-buddy-evidence-evaluator` prompt，返回 `status`/`coverage`/`missing_evidence`/`contradictions`/`next_queries`。
- 校验：`status` 必须 ∈ {sufficient, partial, insufficient}；`coverage` 夹紧到 [0,1]；每项截断。
- **返回值区分两种失败**：
  - `None` = 评估器整体不可用（没配 key / 请求失败 / 输出一条都没通过校验）→ 只保留确定性下限，并置 `evidence_assessment_degraded=True`，由 synthesizer 在「研究局限」里披露「未做语义充分性判断」。
  - `dict` 中缺失某分支 = 评估器答了别的分支却跳过了它 → **fail-closed**，该分支不算充足，gap 的 `reason=semantic_assessment_missing`。

> 早期实现两种情况都返回空映射，且 `semantic_sufficient = not llm_item or (...)`，于是 LLM 故障会让证据门槛**变宽**而不是变严，报告还会声称结论已交叉验证。

**综合判定**：

```
评估器不可用        → semantic_sufficient = True（仅确定性下限 + 报告披露降级）
评估器跳过该分支    → semantic_sufficient = False（fail-closed）
评估器给出结论      → semantic_sufficient = (status==sufficient 且 coverage>=0.75)

sufficient     = hard_floor 且 semantic_sufficient 且 无 contradictions
final_coverage = min(确定性 coverage, LLM coverage)  # LLM 存在时取更严
```

- 有矛盾即 `sufficient=False`。
- 未通过则生成 `ValidationGap`：补搜词优先用 LLM 的 `next_queries`，否则 `_fallback_query`（按语言加"官方报告 数据"/"independent sources"等后缀，validator.py:60-73）。
- `priority`：有矛盾或 `final_coverage < 0.5` → `high`，否则 `medium`。

**预算信号**：

```
budget_exhausted = search_round >= MAX_SEARCH_ROUNDS(4) 或 total_queries >= MAX_TOTAL_QUERIES(30)
stop_reason:
  - 有缺口 且 上次 stop_reason ∈ {no_new_queries, search_unavailable} → 沿用该原因
  - 有缺口 且 budget_exhausted                         → search_budget_exhausted
  - 无缺口                                             → evidence_sufficient
```

> 路由 `route_after_validation` 会在 `stop_reason ∈ {search_budget_exhausted, no_new_queries, search_unavailable}` 时即使有缺口也流向 synthesizer——即补搜已无路可走时停止补搜，用现有证据综合（synthesizer 会把局限写入 `research_notes`，不进文章正文）。

#### synthesizer（synthesizer.py:73）—— 综合（可发布文章模式）

**职责**：流式生成**可直接发布的研究文章**。正文不含评价性内容、不含内嵌 URL；来源用编号 `[n]` 引用，文末参考文献由代码生成。

**三种模式**：
- **全新模式**：`SYNTHESIZER_PROMPT`，正常生成。
- **增量模式**：`SYNTHESIZER_INCREMENTAL_PROMPT`，基于已有知识补充更新，客观陈述新旧信息关系，**不使用 🆕/⚠️ 标记符号**。
- **改进模式**：`SYNTHESIZER_REFINE_PROMPT`，根据 `reflection_feedback` 重写文章（对应路由的 `revise_report`）。

**输入**：`question` / `search_results` / `report` / `reflection_feedback` / `is_incremental` / `has_knowledge` / `knowledge_context` / `evidence_assessments` / `validation_gaps` / `stop_reason` / `evidence_assessment_degraded` / `search_unavailable`
**输出**：`report`（文章正文 + 文末参考文献）/ `confidence` / `research_notes` / `source_table` / `messages`

**无正文引用 + 核心参考文献（核心设计）**：
- `build_source_table()`：从 `search_results` + `known_source_urls` 按 `normalize_url` 去重构建来源表 `[{index, title, url, source}]`，本次检索在前、历史知识在后。
- **正文无引用标注**：三个 prompt 均明确禁止正文出现 `[n]` 编号引用或内嵌 URL（来源统一放文末）。
- 核心文献筛选 `curate_core_references()`：一次性非流式 LLM 调用（`research-buddy-core-refs` prompt），从全部来源中选出最核心的 `MAX_REFERENCES=8` 个（优先权威一手来源）；输出编号越界/重复被丢弃；LLM 失败时降级取来源表前 8 个（Tavily 相关度序）。
- 文末参考文献 `render_references()`：由**代码**渲染筛选后的子集并**重新编号 1..k**（`## 参考文献` + `k. [标题](URL)`），与正文一样走 `report_chunk` 流。
- 三个 prompt 均明确禁止：正文内嵌 URL、`[编号]` 引用、自行添加参考文献/置信度章节、写「存在矛盾/证据不足/研究局限」等元评论。

**置信度由代码确定性计算**（`compute_confidence()`，不进正文）：

| 条件 | 置信度 |
|------|--------|
| `search_unavailable`（无新证据） | 低 |
| `evidence_assessment_degraded` / 预算耗尽 / 无新查询 / 有未解决缺口 | 中 |
| 其余（无缺口、无降级、无预算问题） | 高 |

**研究说明移出正文**（`_build_research_notes()`，写入 state 的 `research_notes`，由代码确定性生成，不依赖模型是否照做 prompt 要求）：

| 触发条件 | 说明内容 |
|---------|---------|
| `search_unavailable` | 本次检索未获得任何新证据，结论缺少来源支撑 |
| `evidence_assessment_degraded` | 语义证据评估不可用，仅做了机械校验 |
| `stop_reason ∈ {search_budget_exhausted, no_new_queries, reflection_budget_exhausted}` | 停止原因 + 逐条未解决缺口 |

**流式输出**：`create_llm(streaming=True).stream(prompt)`，通过 `writer` 回调把 chunk 推到 SSE 层。

> ⚠️ `writer` 必须标注为 `langgraph.types.StreamWriter`。LangGraph 只在参数注解命中 `(StreamWriter, "StreamWriter", inspect.Parameter.empty)` 白名单时才注入（见 `langgraph/_internal/_runnable.py` 的 `KWARGS_CONFIG_KEYS`）。曾经写成 `Callable | None = None`，注入被静默跳过，`writer` 恒为 `None`，`report_chunk` 事件全部消失，前端整套打字机 UI 形同虚设，而所有测试照样通过。现在该参数没有默认值——缺少注入直接 `TypeError`，而不是无声失效。`tests/test_synthesizer.py` 与 `tests/test_sse_stream.py` 锁住这个不变量。

#### reflector（reflector.py:60）—— 反思

**职责**：LLM 自评报告质量，叠加引用一致性检查与多个硬规则覆盖，决定是否循环修正。**fail-closed**。

**输入**：`question` / `sub_questions` / `search_results` / `report` / `reflection_round` / `user_feedback` / `evidence_assessments` / `validation_gaps`
**输出**：`reflection_pass` / `reflection_feedback` / `reflection_round`+1 / `validation_gaps`（覆盖）/ `stop_reason` / `research_complete` / `messages`

**评分维度**：完整性(1-5) + 准确性(1-5) + 清晰度(1-5)，满分 15。

**通过判定**（reflector.py:147）：

```
passed = total_score >= 12  且  min(三个维度) >= 3
```

不只看总分——任一维度低于 3 即不通过。

**硬规则覆盖 LLM 结论**（即使 LLM 判通过也强制不通过）：

1. **引用一致性校验**（可发布文章风格，编号引用）：
   - 从报告正文提取 `[n]` 编号引用（`re.findall(r"\[(\d+)\]", report)`），映射到 `source_table`（synthesizer 构建的编号表）。
   - 证据集 = `search_results` 的 url **+ `known_source_urls`**（历史知识的来源）。增量/追踪模式下 synthesizer 被要求引用 `knowledge_context` 里的历史来源，只用 `search_results` 会把每一条历史引用都判成「不在证据集」，导致增量研究每轮必然不通过直到耗尽预算。
   - 编号表非空但正文无任何 `[n]` → 不通过（"报告没有引用任何已检索来源 URL"）。
   - 正文引用编号不在编号表内 → 不通过（"报告包含 N 个不在来源编号表中的引用编号"）。
   - 引用编号对应的 URL 不在证据集 → 不通过（"报告引用了 N 个不在证据集中的来源"）。
   - 正文内嵌裸 URL 且不在证据集 → 不通过（"报告包含 N 个不在证据集中的 URL"，防 LLM 回归旧式内嵌链接）。
   - citation_issues 前置到 feedback。
2. **未解决证据缺口**：上游 `validation_gaps` 非空 → 强制不通过。
3. **用户反馈优先**：有 `user_feedback` 时，即使 LLM 判通过也强制不通过。
4. **解析失败**：LLM 返回无法解析 → 按未通过处理 + 生成兜底 gap（`reason=reflection_parse_error`，`priority=high`）。

**未通过时的缺口生成**：

- `supplement_queries` 转成 `validation_gaps`，`reason=report_quality_gap`、`priority=high`。
- **归属到具体分支**：按 `evidence_assessments` 的 coverage 升序（最弱的分支优先）轮转分配 `sub_question_id`，并继承该分支的 `language`/`region`。早期实现填空串，而 validator 只统计 `sub_question_id` 非空的结果，于是补搜回来的证据不计入任何分支覆盖率，validator 下一轮又产出同样的缺口，白烧预算。
- **合并上游未解决的缺口**：`validation_gaps` 是覆盖语义，直接返回自己的列表会把 validator 标出的缺口整段擦掉 —— 一旦 LLM 没给 `supplement_queries`，缺口就消失，路由改走 `revise_report`，用完全相同的证据再写一遍报告。现在按 `search_query` 去重后合并保留。

**循环决策**（驱动 `should_continue*` 路由）：

- `reflection_pass=True` → END / knowledge_store
- `search_round >= MAX_SEARCH_ROUNDS` 或 `total_queries >= MAX_TOTAL_QUERIES` → END / store（`stop_reason=reflection_budget_exhausted`）
- `reflection_round >= MAX_REFLECTION_ROUNDS` → END / store
- `search_unavailable` → `revise_report`
- `validation_gaps` 为空 → `revise_report`（回 synthesizer 重写）
- 否则 → `search_again`（回 searcher 补搜）

#### knowledge_lookup（knowledge_lookup.py:11）—— 知识查询

**职责**：检索历史知识为增量研究提供上下文。**不调用 LLM。**

**输入**：`question` / `topic_id`
**输出**：`knowledge_context` / `has_knowledge` / `known_source_urls` / `messages`

**流程**：

1. `store.lookup(question, topic_id)`：并行 ChromaDB 向量检索（report_chunks + key_facts）+ SQLite 取最新报告。
2. `_build_context`：拼装主题摘要 + 增量报告链（递归追溯 `parent_report_id` 3 层）+ 关键事实 + 相关报告片段（截 500 字）。
3. `_extract_source_urls`：从最新报告 sources 抽 URL 供 searcher 增量去重。

#### knowledge_store（knowledge_store.py:12）—— 知识存储

**职责**：将报告保存到知识库（SQLite + ChromaDB 双写），关联 `parent_report_id` 形成增量链；提取关键事实、来源、置信度。

**输入**：`question` / `report` / `topic_id` / `is_incremental` / `key_facts` / `search_results` / `reflection_round`
**输出**：`saved_report_id` / `messages`

**流程**：

1. 增量模式先 `store.get_latest_report(topic_id)` 取 `parent_report_id`（knowledge_store.py:38-42）。
2. `store.save_report()` 双写：SQLite 写元数据/sources/key_facts/research_notes，ChromaDB 写报告分块 + 事实向量。
3. **关键事实提取** `_extract_key_facts`（knowledge_store.py:67-113）：优先用 state 已有 `key_facts`，否则用 **内联 LLM prompt** 从报告提取（⚠️ 未走 Langfuse Prompt 管理，与其他节点不一致），失败兜底取 search_results 首句。
4. 来源提取用 `normalize_url` 去重；置信度直接取 `state["confidence"]`（synthesizer 代码计算，正文已不含置信度文本），旧 state 无该字段时回退正文文本匹配，最终默认「中」。研究说明存 `research_notes` 列。

#### diff_analyzer（diff_analyzer.py:13）—— 变化分析

**职责**：对比新旧报告识别语义变化，把变更写入 SQLite。

**输入**：`topic_id` / `report` / `saved_report_id` / `tracking_log_id`
**输出**：`detected_changes` / `similarity` / `tracking_log_id` / `messages`

**流程**：

1. 取旧报告：`store.list_reports(topic_id, limit=5)` 跳过刚保存的 `saved_report_id`，否则 `get_latest_report`。无旧报告则跳过（首次研究）。
2. 委托 `DiffAnalyzer.analyze()`（见 4.1）两层检测。
3. 持久化：复用或新建 `tracking_log`，逐条 `store.create_change`，更新 `changes_detected` 与 `change_summary`。

#### change_notifier（change_notifier.py:11）—— 变化通知

**职责**：按通知策略通过 Webhook 推送变化通知。**不调用 LLM。**

**输入**：`detected_changes` / `topic_id`
**输出**：`notification_sent` / `messages`

**通知策略** `_should_notify`（change_notifier.py:59-79）：

```
有 high 级别变化               → 通知
有 ≥ 2 条 medium 变化          → 通知
仅 low 变化 / 无变化 / 无 topic → 跳过
```

取主题名后调 `get_notifier().send_change_notification`（见 4.2）。

---

## 三、知识层架构

### 3.1 三层结构

```
 ┌─────────────────────────────────────────────────────┐
 │                    上层代码                          │
 │  nodes/*.py      api.py      scheduler.py           │
 └─────────────┬───────────────────────────────────────┘
               │ 只调用 KnowledgeStore
               ▼
 ┌─────────────────────────────────────────────────────┐
 │              KnowledgeStore (store.py)               │
 │              统一 API 入口（门面模式）                  │
 └──────────┬──────────────────────┬───────────────────┘
            │ 委托                 │ 委托
            ▼                      ▼
 ┌──────────────────────┐  ┌──────────────────────────┐
 │  Database (SQLite)    │  │  VectorStore (ChromaDB)  │
 │  4 张表:             │  │  2 集合:                 │
 │  · topics            │  │  · report_chunks         │
 │  · reports           │  │  · key_facts             │
 │  · tracking_logs     │  │  Embedding 可配置:       │
 │  · changes           │  │  EMBEDDING_BACKEND       │
 └──────────────────────┘  └──────────────────────────┘
```

**设计原则**：上层只通过 `KnowledgeStore` 访问知识层，不直接用 `Database` 或 `VectorStore`。`knowledge/__init__.py` 只导出 `KnowledgeStore` 和 `get_knowledge_store`。

### 3.2 Database（db.py）

SQLite 数据层，文件 `{DATA_DIR}/research_buddy.db`。`threading.local()` 线程本地连接，`journal_mode=WAL` + `foreign_keys=ON`。建表幂等。4 张表 + 4 个索引：

| 表 | 用途 | 关键字段 |
|----|------|---------|
| topics | 研究主题 | id, name, description, tracking_keywords(JSON), tracking_cron, tracking_enabled, created_at, updated_at |
| reports | 研究报告 | id, topic_id(FK 级联), question, report, confidence, sources(JSON), research_notes(JSON), input_tokens/output_tokens/total_tokens, search_results_count, reflection_rounds, is_incremental, parent_report_id, key_facts(JSON), created_at |
| tracking_logs | 追踪记录 | id, topic_id(FK 级联), triggered_at, status(默认 running), changes_detected, change_summary, report_id |
| changes | 变更条目 | id, tracking_log_id(FK 级联), change_type(默认 new_info), description, old_content, new_content, significance(默认 medium) |

**索引**：`idx_reports_topic`、`idx_reports_created`、`idx_tracking_topic`、`idx_changes_log`。行转 dict 时反序列化 JSON 并把 0/1 转 bool。全局单例 `get_db()`。

### 3.3 VectorStore（vector.py）

ChromaDB 向量存储，持久化目录 `{DATA_DIR}/chroma_db`。两个集合均 `metadata={"hnsw:space": "cosine"}`（余弦距离）：

| 集合 | 用途 | Embedding 模型 |
|------|------|---------------|
| report_chunks | 报告文本分块 | 由 `EMBEDDING_BACKEND` 决定，默认 `all-MiniLM-L6-v2` |
| key_facts | 关键事实 | 同上 |

**关键设计**：

- **Embedding 后端可配置**：`EMBEDDING_BACKEND` 三选一。
  - `default`（默认）：ChromaDB 内置 `all-MiniLM-L6-v2`，零额外依赖，英文为主，中文召回偏低。
  - `sentence-transformers`：本地 `paraphrase-multilingual-MiniLM-L12-v2`，中文检索质量好；需要可选依赖 `uv sync --extra multilingual`（会拉入 torch）。
  - `openai`：走 `OPENAI_API_BASE` 的 `/embeddings`；中转站不一定提供该接口。
  - `EMBEDDING_MODEL` 可覆盖各后端的默认模型。
  - 请求的后端不可用时打 **WARNING** 并降级到 `default`，不会静默切换（早期实现是静默 `except`，导致「多语言中文检索」在没装 sentence-transformers 的环境里从未真正生效）。
  - 解析结果模块级记忆化，避免两个集合各加载一份模型。
- **禁止混用不同模型**：生效模型写入 collection metadata（`embedding_backend` / `embedding_model`）。后续启动若配置与已有向量不一致，抛 `EmbeddingBackendMismatch` 而不是继续查询 —— 两个 MiniLM 都是 384 维，混用不会报维度错，只会让结果悄悄失去意义。旧库没有标记时打 WARNING 并按当前配置补标。
- **幂等写入**：`add_report`/`add_facts` 写入前先 `delete_report`/`delete_facts` 清理旧数据。
- **分块**：`chunk_size=500`、`chunk_overlap=100`，按段落聚合、超长段落强制切分。
- **距离阈值**：`max_distance=0.5`（cosine distance），过滤不相关结果。`n_results` 夹紧到集合大小。
- **懒初始化**：client/collection 通过 @property 延迟创建。全局单例 `get_vector_store()`。

### 3.4 KnowledgeStore（store.py）

统一门面，整合 SQLite + ChromaDB：

| 方法 | 说明 |
|------|------|
| `save_report()` | 双写 SQLite + ChromaDB（报告分块 + 关键事实向量） |
| `delete_report()` | 先删 SQLite 再删向量（防止部分删除） |
| `delete_topic()` | 先删 SQLite（级联）成功后再清理向量 |
| `lookup()` | 向量检索 chunks + facts + SQLite 取最新报告，返回 `{chunks, facts, latest_report, has_knowledge}` |
| `get_knowledge_summary()` | 格式化主题知识摘要（元数据 + 关键事实 + 前 5 来源 + 报告前 500 字） |
| `get_report()` / `get_latest_report()` / `list_reports()` | 通过门面访问 db |
| 主题/追踪 CRUD | create/get/list/update topic、tracking_log、change 全部委托 SQLite |

**协调原则**：SQLite 是主数据源（存元数据/结构化数据），ChromaDB 是辅助（存向量分块）。删除时 SQLite 优先、成功后才清向量，避免脏数据。全局单例 `get_knowledge_store()`。

---

## 四、追踪层架构

### 4.1 DiffAnalyzer（diff.py）—— 两层变化检测

```
 旧报告 + 新报告
       │
       ▼
 ┌─────────────────────────┐
 │ 第一层：difflib          │
 │ SequenceMatcher          │
 │ 计算文本相似度            │
 └───────────┬─────────────┘
             │
     ┌───────┴────────┐
     │                 │
 similarity ≥ 0.85   similarity < 0.85
     │                 │
     ▼                 ▼
 ┌─────────┐   ┌─────────────────────┐
 │无显著变化│   │ 第二层：LLM 语义分析  │
 │返回      │   │ 识别有意义的信息变更   │
 │has_      │   └──────────┬──────────┘
 │changes=  │              │
 │false     │     ┌────────┴────────┐
 └─────────┘     │                 │
             LLM 成功           LLM 失败
                 │                 │
                 ▼                 ▼
         ┌──────────────┐  ┌──────────────────┐
         │ 结构化变更列表│  │ Fallback:        │
         │ type         │  │ difflib 行级差异  │
         │ description  │  │ 简单变更列表      │
         │ significance │  └──────────────────┘
         └──────────────┘
```

- **第一层**（diff.py:94-104）：`difflib.SequenceMatcher` 按行算相似度，`similarity_threshold=0.85`，≥ 阈值直接返回无变化（跳过 LLM 省钱）。
- **第二层**（diff.py:106-147）：报告超长截断到 3000 字，用 `research-buddy-diff-analyzer` prompt（要求只识别事实性变化，忽略文字表述差异），`create_llm().invoke` → `parse_llm_json` 解析 JSON 数组；校验每条为 dict 且含 `description`，归一为 `{type, description, old_content, new_content, significance}`。
- **fallback**（diff.py:149-179）：LLM 失败时按行 set 差集，新增行取前 5（`>10` 字记为 `new_info`/`medium`），移除行取前 3（`update`/`low`）。

**输出结构**：

```json
{
  "has_changes": true,
  "similarity": 0.42,
  "changes": [
    {"type": "new_info|update|contradiction",
     "description": "...",
     "old_content": "...",
     "new_content": "...",
     "significance": "high|medium|low"}
  ]
}
```

**语义**：`new_info`（新增）、`update`（更新）、`contradiction`（矛盾）；`high`（重大政策/数据）、`medium`（一般）、`low`（细节）。

### 4.2 Notifier（notifier.py）—— 多平台 Webhook

| 平台 | URL 特征 | 格式 |
|------|---------|------|
| 企业微信 | `qyapi.weixin.qq.com` | Markdown（`## 🔔` 标题 + `🔴 **[新增]** 描述` 加粗行） |
| 钉钉 | `oapi.dingtalk.com` | Markdown（`🔔` title + `🟡 [更新] 描述` 不加粗行） |
| 通用 | 其他（含 Telegram） | JSON（`{title, topic_id, changes, timestamp}` 透传） |

> ⚠️ 文档/注释声称支持 Telegram，但**实际只有 3 个 payload builder**，Telegram 与未知平台统一走 generic JSON 透传。`_build_change_lines`（notifier.py:119-128）是未完成的死代码，实际格式化由 `_format_change_line` 完成。

**配置**：单一环境变量 `NOTIFICATION_WEBHOOK_URL`（默认空串）。

**频率控制**：内存级 `_last_sent[topic_id]`，**12 小时/主题冷却期**（重启失效）。`httpx.Client(timeout=10)` POST JSON，HTTP 200 视为成功并更新 last_sent。提供 `send_test_notification` 测试通知。全局单例 `get_notifier()`。

### 4.3 TrackingScheduler（scheduler.py）—— APScheduler 定时

```
 FastAPI lifespan 启动
         │
         ▼
 TrackingScheduler.start()
         │
         ▼
 加载所有 tracking_enabled 且有 tracking_cron 的主题
         │
         ▼
 为每个主题添加 CronTrigger 定时任务
         │
         │   ... 等待 cron 触发 ...
         ▼
 APScheduler cron 触发 (e.g. "0 9 * * 1-5" 工作日 9 点)
         │
         ▼
 _run_tracking(topic_id) ← async 函数
         │
         ▼
 asyncio.to_thread(_do_tracking) ← 避免阻塞事件循环
         │
         ▼
 create_tracking_graph().stream()
 （追踪图，内置 diff_analyzer + change_notifier）
         │
         ▼
 结果自动保存到知识库 + 变化检测 + 通知
```

**关键设计**：

- `AsyncIOScheduler` + `MemoryJobStore`（**任务存内存、不持久化**，重启后从 SQLite 重新加载）+ `AsyncExecutor`。时区固定 `Asia/Shanghai`。
- **仅支持 `CronTrigger`**：`add_tracking_job` 解析 5 段标准 cron 表达式，非 5 段拒绝。job_id = `tracking_{topic_id}`，`replace_existing=True` + 先删同 ID 任务实现幂等更新，`misfire_grace_time=300`。
- **配置变更必须同步调度器**：`sync_tracking_job(topic)` 按主题当前配置新增/更新/移除任务，返回 `{"scheduled": bool, "reason": str}`（`reason=invalid_cron` 表示配置已写库但 cron 无法解析）。`PUT /topics/{id}` 与 `POST /topics` 写库后必须调它，`DELETE /topics/{id}` 必须调 `remove_tracking_job`。
  > 早期实现里 `add_tracking_job` 的唯一调用点是 `start()`：保存追踪配置只改了 SQLite，UI 提示「已保存」，`GET /tracking/jobs` 仍是空，要重启进程才生效；关闭追踪则旧任务继续触发到进程退出。
- `list_jobs()` 用 `getattr(job, "next_run_time", None)` 读下次触发时间：调度器未 `start()` 时任务处于 pending，APScheduler 3.x 的 `Job` 声明了该 slot 但不赋值，直接取属性会 `AttributeError`。
- `_run_tracking` 是 async 函数，但 `graph.stream()` 同步阻塞，用 `asyncio.to_thread()` 包裹。
- 执行前先建 `tracking_log`（status=running），用追踪关键词（缺省回退主题名）构造问题，调 `create_tracking_graph()`（**硬编码 `is_incremental=True`**）经 `stream_and_accumulate` 流式执行；完成后更新 tracking_log（`completed`/`failed`、`changes_detected`、`change_summary`、`report_id`）。
- 全局单例 `get_scheduler()`。

---

## 五、评估层架构

### 5.1 Dataset（dataset.py）

Langfuse 测试数据集 `research-buddy-eval`，共 **8 个研究问题**（LangGraph vs LangChain、Python GIL、Docker vs K8s、RAG、FastAPI vs Flask、Git rebase vs merge、Transformer 注意力、微服务 vs 单体）。

每条用例：`{"input": "研究问题", "expected_output": [要点列表]}`，每题 3-4 个中文要点。

**幂等性**：`create_dataset()` 先读取已有 items 的 input 集合，跳过已存在者，仅新增缺失项。

### 5.2 Judge（judge.py）

LLM-as-Judge 自动评分，三维度 1-5 分：

| 维度 | 评分范围 | 说明 |
|------|---------|------|
| relevance | 1-5 | 报告是否紧扣研究问题 |
| completeness | 1-5 | 预期要点是否被覆盖 |
| accuracy | 1-5 | 论点是否有来源支撑 |

`judge_report()`：`create_llm()` → `get_prompt_from_langfuse("research-buddy-judge", ...)` → `llm.invoke` → `parse_llm_json`。**容错**：解析失败、或解析成功但不是 JSON 对象（模型返回数组/标量）时，给三维度默认 3 分并标 `parse_failed=True`；分数超范围、非数值或布尔值修正为 3。`run_eval.py` 汇总时排除 `parse_failed` 的条目，避免占位分数粉饰结论。

**评分写入 Langfuse**：`run_research()` 用 `_langfuse_run()` 为整次运行开一个根 span，图内所有节点的 span 通过 OTEL 上下文挂到同一 trace 下，并把 `langfuse_trace_id` 放进返回值；`run_eval.py` 直接用它调 `score_trace()`。

> 早期实现是事后 `langfuse.get_traces(limit=1)` 猜「最近一条 trace」：既有竞态，而且该方法在 Langfuse v3 重写时已删除，装的 4.x 上直接 `AttributeError` 被 `except` 吞掉 —— 整条打分链路是 no-op，只打一行「⚠️ 评分写入失败」。

### 5.2b Dataset 幂等（dataset.py）

`create_dataset()` 用固定 id（`research-buddy-eval-NN`）调 `create_dataset_item`，天然 upsert；幂等检查读 `get_dataset(name).items`。

> 注意 `create_dataset()` 返回的是 API 的 `Dataset` 模型，**没有 `items` 属性**，只有 `get_dataset()` 返回的 `DatasetClient` 才有。早期实现对前者取 `.items`，`AttributeError` 被 `except Exception: pass` 吞掉，幂等检查形同虚设，每跑一次重复插入一整份用例。

### 5.3 Prompts（prompts.py）

Prompt 版本管理：本地 prompt 用 Python `.format()` 语法（`{var}`），注册到 Langfuse 时转 mustache（`{{var}}`），拉取时用 `prompt.compile(**kwargs)`，Langfuse 不可用时 fallback 到本地 `.format()`。

- `convert_format_to_mustache()`：用 `string.Formatter.parse` 正确区分变量占位符与 `{{ }}` 转义字面花括号，避免破坏 JSON 示例。
- `get_prompt(name, fallback, **kwargs)`：从 Langfuse 拉取并 `compile`；任何异常退回 `local_fallback.format(**kwargs)`。
- `register_prompts()`：显式运行 `python -m research_buddy.eval.prompts` 时同步 Prompt 到 Langfuse；应用启动默认不发布，只有 `LANGFUSE_AUTO_REGISTER_PROMPTS=true` 才在 lifespan 调用。发布标签由 `LANGFUSE_PROMPT_REGISTER_LABEL` 控制，便于先发 `staging`、评测通过后再发 `production`。
- **版本管理策略**：先取已有版本，内容相同跳过；变化则创建新版本（`production` 标签）；不存在则新建。

---

## 六、共享工具层（utils.py）

**Token 用量统计**：节点里的 LLM 调用不传 config（Langfuse 的 CallbackHandler 因而收不到节点内 Generation），所以 `create_llm()` 在每个 LLM 上挂 `_UsageRecorder` 回调，把每次调用的 token_usage 累计进 `track_run_tokens()` 开启的 contextvar 计数器（`_normalize_usage` 兼容新旧两种 usage 字段命名）。视觉选图（裸 httpx）在 `tools/images.py` 里手动 `add_tokens` + 用 `start_as_current_observation` 记录 Langfuse generation。运行函数（graph.py / api.py SSE）结束把 `token_usage` 写进结果、`report`/`done` 事件和 reports 表。

消除各模块重复代码的统一工具集：

| 函数/常量 | 作用 |
|-----------|------|
| `parse_llm_json(content)` | 剥离 code-fence + `json.loads` + 错误处理 |
| `create_llm(streaming=False)` | ChatOpenAI 工厂（统一 model/api_key/base_url/temperature=0） |
| `get_prompt_from_langfuse(name, fallback, **kwargs)` | Langfuse Prompt 拉取 + fallback（委托 eval/prompts） |
| `summarize_changes(changes)` | 统一变更摘要文本 |
| `normalize_url(url)` | URL 规范化（去 `www.`、去 fragment、去 utm/ref/fbclid/gclid 追踪参数） |
| `stream_and_accumulate(graph, input_data, config)` | 流式执行图 + 累积最终状态（覆盖/追加镜像语义） |
| `SIGNIFICANCE_EMOJI` | `{high: 🔴, medium: 🟡, low: 🟢}` |
| `CHANGE_TYPE_LABEL` | `{new_info: 新增, update: 更新, contradiction: ⚠️ 矛盾}` |

**Langfuse 接入两层**（均可选，失败 fallback 本地）：

1. **Prompt 管理**：`get_prompt_from_langfuse` 远端拉取 + 本地 fallback。
2. **链路追踪**：`get_langfuse_handler()`（graph.py:33）当 `LANGFUSE_PUBLIC_KEY` 与 `LANGFUSE_SECRET_KEY` 均非空时返回 `langfuse.langchain.CallbackHandler`，通过 `config["callbacks"]` 注入 graph 流，自动 trace 所有 LLM 调用；流式结束后 `_langfuse_client.flush()` 刷出。**未使用 `@observe` 装饰器。**

---

## 七、API 层架构（api.py）

### 7.1 FastAPI 应用

```
 应用启动 ─────► [按配置同步 Prompt] ─► TrackingScheduler.start() ─► FastAPI 运行
 (lifespan)    同步 9 个 prompt 到     加载定时任务                   处理 HTTP/SSE
               Langfuse（失败仅警告）
 应用停止 ─────► TrackingScheduler.stop()  (lifespan)
```

### 7.2 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/research` | 同步研究（无知识层，用核心图） |
| POST / GET | `/research/stream` | SSE 流式研究（GET 供浏览器 EventSource） |
| POST | `/research/hitl/stream` | HITL 流式研究（开始） |
| POST | `/research/hitl/resume/stream` | HITL 恢复执行 |
| GET | `/research/hitl/state` | 查询 HITL 中断态 |
| POST | `/research/knowledge` | 知识研究（增量） |
| POST | `/research/knowledge/stream` | 知识研究 SSE |
| POST / GET | `/topics` | 创建 / 列出主题 |
| GET / PUT / DELETE | `/topics/{id}` | 获取 / 更新（清洗 cron 只取前 5 段）/ 删除主题 |
| GET | `/topics/{id}/reports` | 列出报告 |
| GET / DELETE | `/reports/{id}` | 获取 / 删除报告 |
| POST | `/tracking/run` | 手动触发追踪（`asyncio.to_thread` 包裹） |
| GET | `/tracking/jobs` | 列出追踪任务 |
| GET | `/topics/{id}/tracking-logs` | 追踪记录 |
| GET | `/tracking-logs/{id}/changes` | 变更条目 |
| POST | `/tracking/test-notification` | 测试通知 |
| GET | `/`（静态挂载） | Web UI |

### 7.3 SSE 流式输出

**架构**：`asyncio.Queue` + `asyncio.to_thread` 桥接同步 LangGraph 到异步生成器。

```
 客户端 (EventSource / fetch ReadableStream)
     │
     │ SSE 连接
     ▼
 EventSourceResponse (_event_generator)
     │
     │ async inner()
     ▼
 asyncio.Queue  ◄──────────────────────────────┐
     │                                          │
     │ queue.get()                              │ queue.put_nowait()
     ▼                                          │
 asyncio.to_thread(_run_graph)                  │
     │                                          │
     │ 在线程中执行                               │
     ▼                                          │
 graph.stream(stream_mode=['updates','custom']) ┤
 (同步)         ──── progress ──────────────────┤
                ──── message ───────────────────┤
                ──── report_chunk (custom) ─────┤
                ──── report ────────────────────┤
                ──── done ──────────────────────┤
                ──── error ────────────────────┘
```

**事件类型**：`progress`（节点更新，含 summary + 结构化 detail）、`message`（节点内消息）、`report_chunk`（custom event，流式报告片段）、`report`（最终报告）、`done`、`error`。

> `stream_mode` 必须传 **list**：LangGraph 只在 `isinstance(stream_mode, list)` 时把事件包成 `(mode, payload)`，传 tuple 会退化成裸 payload，只能靠 `'type'` 键猜事件类型。三个 SSE 生成器共用 `_emit_stream_event(queue, mode, payload, result)` 做事件转换与状态累积。前端 `EventSource` 的 `error` 监听器同时收具名 `error` 事件和传输错误，带 `data` 的必须交给 `handleSSEEvent` 显示，否则服务端错误会被当成断线，用户只看到「SSE 重连中」。`_extract_detail`（api.py:884）和 `_summarize_update`（api.py:947）为每个节点产出结构化详情和可读摘要。

**HITL 流式**（`_hitl_event_generator` / `_hitl_resume_event_generator`）：用 `MemorySaver` + `thread_id`，流结束后 `graph.get_state` 检查 `snapshot.next` 判断中断点——`searcher` → `confirm_sub_questions`，`reflector` → `review_report`，推送 `interrupt` 事件；恢复时用 `update_state` 覆盖 `sub_questions`/`user_feedback` + `Command(resume=...)`。会话存内存字典 `_hitl_sessions`。

**两种 SSE 客户端**：常规研究用 GET `EventSource`；知识研究/HITL 用 POST `fetch()` + `ReadableStream`（因为 EventSource 不能发请求体）。

---

### 7.4 前端执行视图与报告视图（static/）

`index.html` 承载标记 + 全部 JS + 早期内嵌样式表，`app.css` 是后加载的视觉系统（同名变量时它赢）。

**执行视图三块**：

| 区域 | 元素 | 行为 |
|------|------|------|
| 证据图舞台 | `.observatory-stage` + `#researchCanvas` | 高度固定 424px（`.observatory-viewport` 用 `grid-template-rows: 424px` 锁住行高，否则右侧叙事栏内容一多就把整个视口撑到 700px+）。画布几何中心按 `ResearchEvidenceMap.BOTTOM_INSET = 34` 上移，避开底部状态条 |
| 底部状态条 | `.observatory-metrics` | 分支/信号/缺口计数 + 右侧节点流水线 `renderPipelineStrip()`：当前节点高亮、走过的节点变亮。补搜会让节点重复经过，所以「已完成」只按首次到达的最远位置算 |
| 叙事栏 | `.execution-narrative` | 三行栅格 `auto minmax(0,1fr) auto`：当前阶段 / 分支列表（占剩余高度，独立滚动）/ 最近变化（限高 124px 滚动） |

**节点面板（`#stepCards`）**：每个节点一张固定面板，`nodePanels` 注册表按 `appState.activeSteps` 的顺序插入（不是到达顺序），同节点重复到达即「下一轮」——旧正文进 `.sc-history` 折叠区，头部 `.sc-round` 标注第几轮，卡片闪一次 `is-updated` 高亮。

> 早期实现是每个事件 append 一张新卡片，补搜/重写循环让 validator、searcher、synthesizer 各堆出四五张几乎一样的卡片，实测 `#stepCards` 高 7162px。改成原地更新后同一条件下是 943px。

**报告视图**：`.report-scorecard`（置信度 / 分支证据充足 / 质量评分 / 检索来源 四块指标 + 降级与预算告警）→ `.report-layout` 两栏（正文 + 右侧 sticky `.report-toc`，≤1000px 收起）→ `.report-sources`（正文引用的 URL 按域名分组，标注「本次检索 / 历史知识 / 报告引用」）。流式阶段 `.report-body.is-streaming` + `.report-caret` 光标，定稿时切到 `.is-settled` 做一次过渡。

指标卡的色调看的是「是否通过」而不是分数：硬规则（引用一致性、未解决缺口、用户反馈）可以在 15/15 时仍判不通过。

**CJK 排版铁律**：中文的 min-content 宽度只有一个字（字与字之间都是换行机会）。因此弹性行 / 栅格轨道里的中文一旦被挤压，会直接退化成一字一行的竖排，而 `min-width: 0` **救不了**——它只会让塌缩更彻底。规则是：

- 单行文本：`min-width: 0` + `overflow: hidden` + `text-overflow: ellipsis` + `white-space: nowrap`
- 不可压缩的图标 / 角标 / 计数：`flex-shrink: 0`
- 栅格轨道一律写 `minmax(0, 1fr)`，不写裸 `1fr`
- 可滚动的纵向 flex 列表，列表项必须 `flex-shrink: 0`，否则会被压到比内容矮而互相重叠

> 真实事故：`app.css` 让 `.nd-gap` 复用了 `.nd-sq` 的 `24px 1fr` 栅格，但 `.nd-gap` 的两个子元素是「问题」和「搜索词」，问题落进了 24px 轨道 —— 验证卡片里每个证据缺口的问题都渲染成 24px 宽、28 行高的一列竖字，单张卡片 1642px。修法是给 `.nd-gap` 补上编号 chip，让 DOM 结构与 `.nd-sq` 一致。

## 八、配置层（config.py）

`load_dotenv()` 在 import 时调用。`PROJECT_ROOT` 为 `src/research_buddy` 上两级。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATA_DIR` | `{PROJECT_ROOT}/data` | SQLite + ChromaDB 持久化目录 |
| `OPENAI_API_KEY` | `""` | LLM API Key |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | API 中转站地址 |
| `OPENAI_MODEL` | `gpt-4o` | 模型名称 |
| `LANGFUSE_PUBLIC_KEY` | `""` | Langfuse 公钥（不配则无 Trace） |
| `LANGFUSE_SECRET_KEY` | `""` | Langfuse 私钥 |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse 地址 |
| `MAX_SEARCH_RESULTS` | `5` | 每次搜索最大结果数 |
| `MAX_SEARCH_ROUNDS` | `4` | 最大搜索轮次（补搜循环闸） |
| `MAX_TOTAL_QUERIES` | `30` | 补搜查询预算（初始轮基础搜索不计入） |
| `MAX_REFERENCES` | `8` | 文末参考文献最多条数（LLM 筛选核心子集） |
| `MAX_REFLECTION_ROUNDS` | `2` | 最大反思轮次（修正循环闸） |
| `MIN_EVIDENCE_COVERAGE` | `0.75` | 证据覆盖率硬底线 |
| `MIN_DISTINCT_DOMAINS` | `2` | 最少独立域名数（交叉验证信号） |
| `MIN_RESULTS_PER_SUB_QUESTION` | `2` | 每子问题最少有效结果数 |
| `MIN_SEARCH_CONTENT_LENGTH` | `80` | 有效结果最小内容长度 |
| `TAVILY_API_KEY` | `""` | Tavily 搜索 API Key（**必填**，缺失时研究直接中止） |
| `EMBEDDING_BACKEND` | `default` | 向量后端：`default` / `sentence-transformers` / `openai` |
| `EMBEDDING_MODEL` | `""` | 覆盖所选后端的默认模型 |
| `NOTIFICATION_WEBHOOK_URL` | `""` | 通知 Webhook URL |

---

## 九、完整数据流

### 9.1 全新研究流程

```
 用户: "Python GIL 是什么？"
  │
  ▼
 ┌──────────┐    LLM 拆解为 3 个子问题（稳定 ID sq_01/02/03）
 │ planner  ├────────────────────────────┐
 └──────────┘                            │
  │                                      ▼
  │                               sub_questions（含 search_queries，
  │                               可能中英双语 + 地区偏好）
  │
  ▼
 ┌──────────┐    ThreadPoolExecutor 并行搜索（第 1 轮 basic）
 │ searcher ├────────────────────────────┐
 └──────────┘                            │
  │                                      ▼
  │                               Tavily API × N（并行）
  │                               → 结果双去重（URL + 内容指纹）
  │
  ▼
 ┌──────────┐    确定性指标 + LLM 语义评估
 │validator │ ──→ 全部充足（hard_floor ∩ semantic ∩ 无矛盾）✅
 └──────────┘    stop_reason=evidence_sufficient
  │
  ▼
 ┌───────────┐   LLM 流式生成结构化报告
 │synthesizer│ ──→ "# Python GIL 研究\n## 概述\n..."
 └───────────┘
  │
  ▼
 ┌──────────┐    LLM 自评 + 引用 URL 交叉校验
 │reflector │ ──→ completeness=4, accuracy=4, clarity=4, total=12/15
 └──────────┘    min(dim)=4≥3，引用 URL 全在证据集 → pass: true ✅
  │              stop_reason=completed
  ▼
 输出: 报告 + 来源引用 + 置信度
```

### 9.2 循环修正流程

```
 ┌──────────┐ 评分 9/15 或 min(dim)=2<3 或引用了未知 URL
 │reflector │ pass: false ⚠️
 └────┬─────┘
      │ validation_gaps（supplement_queries 转 gap，priority=high）
      ▼
 ┌──────────┐
 │ searcher │ 补搜（第 2 轮起 high 优先级用 advanced 深度）
 └────┬─────┘
      │
      ▼
 ┌──────────┐
 │validator │ 重新评估
 └────┬─────┘
      │ 无缺口 → synthesizer
      ▼
 ┌───────────┐ 根据反馈用 refine prompt 重写报告
 │synthesizer│
 └────┬──────┘
      │
      ▼
 ┌──────────┐ 评分 13/15，引用一致 → pass: true ✅
 │reflector │
 └────┬─────┘
      ▼
   输出改进后的报告

 ┊ 最多循环 MAX_SEARCH_ROUNDS=4 轮搜索 / 补搜 MAX_TOTAL_QUERIES=30 次查询
 ┊   / MAX_REFLECTION_ROUNDS=2 轮反思，三道预算闸防无限循环
 ┊ 预算耗尽时 stop_reason=search_budget_exhausted，停止补搜直接综合
```

### 9.3 增量研究流程

```
 用户: "Python GIL 最新进展" (topic_id=xxx, is_incremental=true)
  │
  ▼
 ┌─────────────────┐
 │knowledge_lookup │ ──→ 查询知识库
 └─────────────────┘    │
      │                 ├── 向量检索: report_chunks + key_facts（max_distance=0.5）
      │                 └── 结构化查表: 最新报告 + parent_report 链（3 层）
      │
      ▼ knowledge_context + known_source_urls
 ┌──────────┐
 │ planner  │ 增量模式: 只规划缺失的 2-3 个子问题
 └────┬─────┘
      │
      ▼
 ┌──────────┐ 搜索 + 增量去重（过滤 known_source_urls 中的 URL）
 │ searcher │
 └────┬─────┘
      │
      ▼
 ┌───────────┐ 增量综合: 在已有知识上补充更新，标注 🆕 新增 / ⚠️ 矛盾
 │synthesizer│
 └────┬──────┘
      │
      ▼
 ┌──────────┐ 评估通过 ✅
 │reflector │
 └────┬─────┘
      │
      ▼
 ┌────────────────┐
 │knowledge_store │ 保存到知识库
 └────────────────┘
      │                ├── SQLite: 元数据 + key_facts + parent_report_id 挂增量链
      │                └── ChromaDB: 报告分块 + 事实向量
      ▼
 输出: 更新后的报告
```

### 9.4 定时追踪流程

```
 APScheduler cron 触发 (e.g. 工作日 9 点)
      │
      ▼
 _run_tracking(topic_id) ← async 函数
      │
      ▼
 asyncio.to_thread(_do_tracking) ← 避免阻塞事件循环
      │
      ▼
 create_tracking_graph().stream()  (is_incremental=True)
      │
      ▼
 knowledge_lookup → planner → searcher → validator → editorial_planner
      → synthesizer → language_editor → article_editor → reflector
      │ (通过后)
      ▼
 knowledge_store → diff_analyzer → change_notifier
      │                │                │
      │                ▼                ▼
      │           对比新旧报告     通知策略判断
      │           两层变化检测     high>0 或 medium≥2 → 通知
      │           (difflib 阈值        │
      │            0.85 筛选 +         ▼
      │            LLM 语义)      检测 Webhook 类型
      │                          │
      │              ┌───────────┼───────────┐
      │              ▼           ▼           ▼
      │           企业微信      钉钉      通用 JSON
      │          (Markdown)  (Markdown)   (透传)
      ▼
 更新 tracking_log（completed，changes_detected，change_summary）
```

### 9.5 人机交互流程

```
 用户 ──► 输入研究问题
              │
              ▼
         planner 执行
              │
              ▼
         ⏸ 暂停 (interrupt_before=searcher)
         MemorySaver 保存状态
              │
              ▼
         展示子问题，等待用户确认
              │
         用户编辑子问题 (或直接确认)
              │
              ▼
         Command(resume={"sub_questions": [...]})  ← 覆盖语义替换
              │
              ▼
         searcher → validator → synthesizer
              │
              ▼
         ⏸ 暂停 (interrupt_before=reflector)
         MemorySaver 保存状态
              │
              ▼
         展示报告，等待用户反馈
              │
         用户补充要求 (或直接确认)
              │
              ▼
         Command(resume={"user_feedback": "补充..."})
              │
              ▼
         reflector（有 user_feedback → 强制不通过 → 补搜/重写）
              │
              ▼
         输出最终报告
```

### 9.6 通知策略决策流程

```
 检测到变更列表
      │
      ▼
 ┌──────────────────┐
 │ 有 high 级别变化？ │
 └────────┬─────────┘
     是 ──┤── 否
      │         │
      ▼         ▼
 发送通知   ┌────────────────────┐
      │    │ 有 ≥ 2 条 medium 变化？│
      │    └─────────┬──────────┘
      │          是 ──┤── 否
      │           │         │
      │           ▼         ▼
      │       发送通知    跳过通知（仅 low / 无变化）
      │           │
      ▼           ▼
 ┌──────────────────┐
 │ 主题在冷却期内？   │  (12 小时/主题，内存级)
 └────────┬─────────┘
     否 ──┤── 是
      │         │
      ▼         ▼
 检测 Webhook   跳过通知
 类型           (冷却中)
      │
      ├─► qyapi.weixin.qq.com  ──► 企业微信 (Markdown + **加粗**)
      ├─► oapi.dingtalk.com    ──► 钉钉 (Markdown 无加粗)
      └─► 其他（含 Telegram）  ──► 通用 JSON
```

### 9.7 Langfuse 可观测性流程

```
 graph.stream()
      │
      │ CallbackHandler 自动记录
      ▼
 Langfuse
  │
  ├── Trace（每次研究执行）
  │     │
  │     ├── Span: planner
  │     │     └── Generation: LLM 调用详情 (model, tokens, latency)
  │     │
  │     ├── Span: searcher
  │     │
  │     ├── Span: validator
  │     │     └── Generation: evidence-evaluator 调用
  │     │
  │     ├── Span: synthesizer
  │     │     └── Generation: LLM 调用详情
  │     │
  │     └── Span: reflector
  │           └── Generation: LLM 调用详情
  │
  ├── Score: LLM-as-Judge 评分（judge.py 写入）
  │     ├── relevance: 1-5
  │     ├── completeness: 1-5
  │     ├── accuracy: 1-5
  │     └── total: 3-15
  │
  └── Prompt 管理（版本控制，prompts.py 注册 9 个）
        ├── research-buddy-planner
        ├── research-buddy-planner-incremental
        ├── research-buddy-synthesizer
        ├── research-buddy-synthesizer-incremental
        ├── research-buddy-synthesizer-refine
        ├── research-buddy-reflector
        ├── research-buddy-evidence-evaluator
        ├── research-buddy-diff-analyzer
        └── research-buddy-judge
```

> 注：knowledge_store 的关键事实提取用**内联 prompt**，未纳入 Langfuse Prompt 管理，是唯一未接入的 LLM 调用点。

---

## 十、涉及的核心知识点

### 10.1 LangGraph 核心概念

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **StateGraph** | 定义工作流图，节点读写 ResearchState | graph.py |
| **Node** | 每个研究步骤是一个节点函数 | nodes/*.py |
| **State** | TypedDict + Annotated[list, operator.add] 追加语义；部分字段覆盖语义 | state.py |
| **Conditional Edge** | validator→editorial_planner/searcher；reflector→end/store/searcher/synthesizer | graph.py |
| **Loop** | 补搜回路（validator） + 修正回路（reflector），三道预算闸 | graph.py |
| **interrupt_before** | HITL 图在 searcher 和 reflector 前暂停 | graph.py |
| **MemorySaver** | HITL 图的 Checkpointer，保存中断状态 | graph.py |
| **Command(resume=)** | 恢复中断的图执行 | graph.py |
| **graph.stream()** | 流式执行，stream_mode=['updates','custom']（必须是 list 才会 yield (mode, payload)） | graph.py, api.py |

### 10.2 Langfuse 核心概念

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **CallbackHandler** | 接入 LangGraph，自动记录 Trace/Span/Generation | graph.py |
| **Trace** | 每次研究执行是一个 Trace | 自动 |
| **Span** | 每个节点是一个 Span | 自动 |
| **Generation** | 每次 LLM 调用是一个 Generation | 自动 |
| **Score** | LLM-as-Judge 评分写入 Trace | eval/judge.py |
| **Dataset** | 测试数据集（8 个研究问题 + 预期要点） | eval/dataset.py |
| **Prompt Management** | 9 个 prompt 在 Langfuse 中版本管理 | eval/prompts.py |

### 10.3 Python 并发与异步

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **ThreadPoolExecutor** | 并行搜索多个子问题（I/O 密集型，max_workers=min(total,4)） | nodes/searcher.py |
| **asyncio.to_thread()** | 在 async 函数中运行同步阻塞的 graph.stream() | tracking/scheduler.py, api.py |
| **asyncio.Queue** | SSE 事件推送（线程 → 异步生成器） | api.py |
| **threading.local()** | SQLite 线程本地连接 | knowledge/db.py |
| **AsyncIOScheduler** | APScheduler 异步调度器 | tracking/scheduler.py |

### 10.4 向量检索

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **ChromaDB** | PersistentClient 持久化向量存储 + 语义检索 | knowledge/vector.py |
| **Sentence Transformer** | 可选后端（EMBEDDING_BACKEND=sentence-transformers）：paraphrase-multilingual-MiniLM-L12-v2 | knowledge/vector.py |
| **Cosine Distance** | hnsw:space=cosine + max_distance=0.5 过滤 | knowledge/vector.py |
| **Text Chunking** | 按段落分割 + chunk_size=500 + chunk_overlap=100 | knowledge/vector.py |

### 10.5 FastAPI + SSE

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **lifespan** | 按配置同步 Prompt + 启动调度器，停止关调度器 | api.py |
| **EventSourceResponse** | SSE 流式响应（ping=15） | api.py |
| **StaticFiles** | 挂载 Web UI（必须放在路由之后） | api.py |
| **Pydantic BaseModel** | 请求/响应模型 | api.py |

### 10.6 设计模式

| 模式 | 本项目用法 | 文件 |
|------|-----------|------|
| **Facade** | KnowledgeStore 统一门面，隐藏 db + vector | knowledge/store.py |
| **Singleton (Lazy)** | get_db/get_vector_store/get_knowledge_store/get_notifier/get_scheduler/get_tavily | 各模块 |
| **Factory** | create_llm() 工厂函数，统一 LLM 实例化 | utils.py |
| **Strategy** | Notifier 根据 URL 自动选择 payload 格式 | tracking/notifier.py |
| **Observer** | Langfuse CallbackHandler 自动观察 LLM 调用 | graph.py |

---

## 十一、部署架构

### 11.1 整体系统架构

```
 ┌──────────────────────────── 客户端 ────────────────────────────┐
 │   Web UI (index.html + app.css)    API 客户端 (curl / SDK)      │
 └──────┬──────────────────────────────┬──────────────────────────┘
        │ SSE                          │ HTTP
        ▼                              ▼
 ┌──────────────────────── FastAPI 服务 ──────────────────────────┐
 │  HTTP 端点 │ SSE 流式 │ HITL 会话 │ 静态文件挂载                │
 └─────────┬─────────────────┬──────────────────┬────────────────┘
           │                 │                  │
           ▼                 ▼                  ▼
 ┌───────────────────── LangGraph 工作流 ───────────────────────┐
 │  核心研究图 │ 知识研究图 │ 追踪图 │ HITL 图                    │
 └──┬──────────────────┬──────────────────┬────────────────────┘
    │ 节点层            │ 知识层            │ 追踪层
    └──────────────────┴──────────────────┴────────────────────┘
        │                │                    │
 LLM API(中转站)    SQLite+ChromaDB      APScheduler+Webhook
        └──────── Tavily 搜索 │ Langfuse 可观测性 ────────┘
```

### 11.2 本地开发

```bash
# 方式 1: dev.sh 脚本
./scripts/dev.sh start    # 启动（自动杀旧进程）
./scripts/dev.sh stop     # 停止
./scripts/dev.sh status   # 查看状态

# 方式 2: 直接运行
uv run python scripts/run_api.py

# 方式 3: CLI
uv run research-buddy
```

### 11.3 Docker 部署

```bash
docker build -t research-buddy .
docker run -p 8000:8000 --env-file .env research-buddy
```

> ⚠️ Docker 设置未完全修复，可能需要项目安装或 `PYTHONPATH=src`（见第十四章）。

### 11.4 运行时目录

```
data/                        # 运行时数据（.gitignore 已排除）
├── research_buddy.db        # SQLite 数据库（WAL 模式）
└── chroma_db/               # ChromaDB 持久化
```

---

## 十二、测试体系

| 测试文件 | 覆盖内容 | 真实/mock |
|---------|---------|---------|
| test_routing.py | 路由纯函数 `should_continue`/`should_continue_to_store`/`route_after_validation` | 真实纯函数 |
| test_state.py | State TypedDict 字段 + operator.add 追加语义 vs 覆盖语义 | 真实类型检查 |
| test_graph.py | 4 种图编译 + 节点拓扑 + 条件边验证 | 真实图构建（不执行） |
| test_diff.py | DiffAnalyzer 公共 `analyze` + 私有方法 + difflib fallback | 真实（difflib 算法） |
| test_hitl.py | HITL 图编译 + checkpointer + interrupt_before + API 端点存在 + update_state 覆盖语义 | 图真实；API 用 TestClient（不执行真实研究） |
| test_prompt_management.py | `convert_format_to_mustache` + `get_prompt` fallback + 真实 prompt 转换 | 真实纯函数（不连 Langfuse，用不存在的 prompt 名触发 fallback） |
| test_reflector.py | reflector 失败策略 + 代码侧通过条件（解析失败/维度全 5 但 pass:False/未知引用 URL/编号引用校验：[n] 有效通过、编号越界失败、编号表非空但无引用失败、历史来源可被编号引用） | mock create_llm + get_prompt_from_langfuse |
| test_validator.py | validator + searcher 证据评估与搜索归属（充足/不足/矛盾/预算耗尽/双语/语义评估降级/fail-closed/无 score 不加成） | mock _llm_assess + searcher.search |
| test_synthesizer.py | writer 注入是否真的发生（跑单节点图看 custom 流）+ 研究说明移入 research_notes + 置信度代码计算 + 编号引用表/文末参考文献 | 真实图 + mock LLM |
| test_report_notes_db.py | reports 表 research_notes 列持久化 + 旧库无列自动迁移 | 真实 SQLite（临时库） |
| test_sse_stream.py | **端到端**：跑真实 create_research_graph()，验证 SSE 协议（report_chunk / progress 顺序 / message / report / done）+ 搜索层失败发 error 而非报告 | 真实图 + TestClient；LLM 与搜索 mock |
| test_search_failure.py | `tools/search` 无 key 抛错 / 退避重试 / 零命中不算失败；searcher 全失败中止 vs 有历史知识降级 vs 部分失败继续 | mock Tavily client |
| test_tracking_config.py | `sync_tracking_job` 新增/移除/无效 cron/替换 + `PUT /topics` 与 `DELETE /topics` 同步调度器 | 真实 APScheduler；store 与 scheduler 用假实现 |
| test_eval_pipeline.py | judge 容错（合法 JSON 数组 / 不可解析 / 越界与布尔分数）+ Dataset 稳定 id 幂等 | mock LLM 与 Langfuse |
| test_vector_backend.py | embedding 后端解析（默认/未知/不可用告警/记忆化/模型覆盖）+ collection 模型标记与混用拒绝 | mock embedding builder 与 Chroma client |

**端到端测试脚本**：`scripts/test_phase1.py`（线性图）、`test_phase2.py`（条件分支+循环修正）、`test_phase3.py`（人机交互）。

**未覆盖的路径**：知识库 `db.py`/`vector.py` 写入路径与 `store.py` 无测试；追踪 `notifier.py` 四种平台 payload 无测试；`planner`/`knowledge_lookup`/`knowledge_store`/`change_notifier` 节点无单测；HITL 的 resume SSE 生成器未深度测试；前端 JS 无测试。

---

## 十三、环境变量一览

| 变量 | 必需 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ | LLM API Key |
| `OPENAI_API_BASE` | ✅ | API 中转站地址 |
| `OPENAI_MODEL` | 可选 | 模型名称（默认 gpt-4o） |
| `TAVILY_API_KEY` | ✅ | Tavily 搜索 API Key |
| `LANGFUSE_PUBLIC_KEY` | 可选 | Langfuse 公钥（不配则无 Trace） |
| `LANGFUSE_SECRET_KEY` | 可选 | Langfuse 私钥 |
| `LANGFUSE_HOST` | 可选 | Langfuse 地址 |
| `DATA_DIR` | 可选 | 数据目录（默认 `{PROJECT_ROOT}/data`） |
| `MAX_SEARCH_RESULTS` | 可选 | 每次搜索最大结果数（默认 5） |
| `MAX_SEARCH_ROUNDS` | 可选 | 最大搜索轮次（默认 4） |
| `MAX_TOTAL_QUERIES` | 可选 | 补搜查询预算（默认 30，初始轮不计入） |
| `MAX_REFERENCES` | 可选 | 文末参考文献条数上限（默认 8） |
| `MAX_REFLECTION_ROUNDS` | 可选 | 最大反思轮次（默认 2） |
| `MIN_EVIDENCE_COVERAGE` | 可选 | 证据覆盖率硬底线（默认 0.75） |
| `MIN_DISTINCT_DOMAINS` | 可选 | 最少独立域名数（默认 2） |
| `MIN_RESULTS_PER_SUB_QUESTION` | 可选 | 每子问题最少结果数（默认 2） |
| `MIN_SEARCH_CONTENT_LENGTH` | 可选 | 有效结果最小内容长度（默认 80） |
| `EMBEDDING_BACKEND` | 可选 | 向量后端：default / sentence-transformers / openai（默认 default） |
| `EMBEDDING_MODEL` | 可选 | 覆盖所选后端的默认模型 |
| `NOTIFICATION_WEBHOOK_URL` | 可选 | 通知 Webhook URL |

---

## 十四、已知设计观察 / 技术债

供后续重构参考，非功能缺陷清单：

1. **列表累积逻辑重复 4 处**：`utils.stream_and_accumulate`（utils.py:135）+ api.py 三个 SSE 生成器（api.py:509/639/804）各内联一份覆盖/追加镜像逻辑，且需与 state.py 的 `operator.add` 注解手动保持同步。是重构消除重复的明显切入点。
2. **knowledge_store prompt 未接入 Langfuse**：关键事实提取用内联 LLM prompt（knowledge_store.py:67-113），是唯一未纳入 Langfuse Prompt 管理的 LLM 调用点，与其他 9 个 prompt 不一致。
3. **notifier 死代码 + 平台声明不符**：`_build_change_lines`（notifier.py:119-128）未完成未使用；注释声称支持 Telegram 但实际只有 3 个 builder，Telegram 走 generic 透传。
4. **调度状态不持久化**：APScheduler `MemoryJobStore`，任务重启从 SQLite 重建；通知冷却 `_last_sent` 仅内存级，重启失效。
5. **Docker 未完全修复**：可能需要项目安装或 `PYTHONPATH=src`。
6. **测试覆盖不全**：无集成测试；知识层 store、追踪层 scheduler/notifier、工具层 search、eval 层 dataset/judge 均无测试；HITL SSE 仅测端点存在性，会触发真实图执行（需重写）。
7. **前端**：Ctrl+K 提示存在但未实际绑定；无深浅主题切换；app.css 的浅色 `:root` 覆盖了 index.html 内嵌深色变量，最终是"浅色主体 + 深色控制台装饰"混合设计。
