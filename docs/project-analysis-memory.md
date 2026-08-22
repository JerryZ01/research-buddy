# 项目分析记忆

> 最后更新：2026-08-22
>
> 这份文件记录**审计结论和未完成事项**，不重复 `architecture.md` 里已有的架构描述。
> 架构、数据流、节点职责看 `docs/architecture.md`；这里只记「查过什么、修了什么、还剩什么」。

## 一、项目形态

LangGraph + Langfuse 深度研究 Agent，四层：工作流（4 张图 / 9 节点）、知识层（SQLite + ChromaDB）、
追踪层（APScheduler + Diff + Webhook）、评估层（Langfuse Dataset + LLM-as-Judge + Prompt 版本管理）。
前端是 FastAPI 静态托管的单页应用（`static/index.html` + `static/app.css`）。

**当前基线**：`UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` → **143 passed**（约 90s），
`python -m compileall -q src/research_buddy tests scripts` 干净。

---

## 二、2026-08-21/22 已修复

### 2.1 四个静默失效（读代码看不出来，测试也测不出来）

| 问题 | 位置 | 修法 |
|------|------|------|
| 流式报告完全没生效 | `nodes/synthesizer.py` 的 `writer` 标成 `Callable \| None`，不在 LangGraph 的注入白名单（`langgraph/_internal/_runnable.py` 的 `KWARGS_CONFIG_KEYS`）内 | 改成 `writer: StreamWriter` 且**不给默认值**，缺注入直接 `TypeError`；`api.py` 的 `stream_mode` 从 tuple 改 list 并正确解包 `(mode, payload)`，三个 SSE 生成器合并到 `_emit_stream_event()` |
| 保存追踪配置不注册定时任务 | `add_tracking_job` 唯一调用点是 `scheduler.start()` | 新增 `TrackingScheduler.sync_tracking_job(topic)`，接到 POST/PUT/DELETE `/topics`；响应带 `tracking_scheduled` / `tracking_warning`，前端 toast 据此提示 |
| Langfuse 评估打分整条链路 no-op | `run_eval.py` 调 v3 已删除的 `langfuse.get_traces()`；`dataset.py` 对 `create_dataset()` 的返回值取 `.items`（该对象没有 items） | `run_research()` 用 `_langfuse_run()` 开根 span 并回传 `langfuse_trace_id`；Dataset 改用 `get_dataset().items` + 固定 id 真 upsert |
| 中文向量检索实际用英文模型 | `vector.py` 静默 `except` 降级，且 `sentence-transformers` 从未进依赖 | `EMBEDDING_BACKEND`（default / sentence-transformers / openai）+ `EMBEDDING_MODEL`，不可用打 WARNING，生效模型写入 collection metadata，不一致抛 `EmbeddingBackendMismatch` |

### 2.2 证据闭环的五个逻辑缺陷

- **缺口被擦除**：`reflector` 用 `_merge_gaps()` 按 `search_query` 去重合并自己的补充缺口与上游未解决缺口，
  不再无条件覆盖 `validation_gaps`（覆盖语义下返回空列表等于把 validator 的发现删掉，路由改走 `revise_report` 用同样证据重写）。
- **增量引文审计必然失败**：证据集加入 `known_source_urls`，历史来源不再被判「不在证据集」。
- **补搜结果不计入覆盖率**：`_supplement_targets()` 按 coverage 升序把补充查询分配给最弱分支，
  继承其 language/region。之前 `sub_question_id=""`，而 validator 只统计非空 id 的结果。
- **门槛在故障时变宽**：`_llm_assess` 返回 `None`（评估器整体不可用 → 仅确定性下限 + `evidence_assessment_degraded=True` 由报告披露）
  还是 `dict`（缺某分支 = 评估器跳过它 → fail-closed）。无 score 时不再默认 `relevance=0.7`，
  直接丢掉该项（上限 0.8，仍高于阈值 0.75）——**不能**重新归一化剩余权重，那等于假设 `relevance=1.0`。
- **零证据仍产出自信报告**：`tools/search.search` 无 key 直接抛 `SearchUnavailableError`，瞬时错误退避重试 3 次后抛；
  `searcher` 区分四种情况（部分失败继续 / 全失败但有历史结果则标记停止补搜 / 全失败零证据但有历史知识则降级披露 / 全失败零证据无知识则中止）。

### 2.3 前端排版与执行视图

- **中文竖排**（用户实际报告的问题）：`app.css` 让 `.nd-gap` 复用 `.nd-sq` 的 `24px 1fr` 栅格，
  但 `.nd-gap` 的两个子元素是「问题 + 搜索词」，问题落进 24px 轨道 → 24px 宽、28 行高的一列竖字，单卡 1642px。
  补编号 chip 使结构与 `.nd-sq` 一致 → 342px。**与窗口宽度无关，1440/900/820 全复发。**
- 同类隐患全部加固：`.sc-title`/`.sc-badge`、`.nd-kl-item`、`.observatory-mode`、`.execution-section-title`、
  `summary::before`、`.terminal-title`/`.terminal-dot`、`.report-collapsed-info`、`.report-collapse-btn`；
  所有裸 `1fr` → `minmax(0, 1fr)`。
- 修改过程中撞出并修掉的新问题：可滚动纵向 flex 列表的列表项缺 `flex-shrink: 0`，
  被压到比内容矮导致相邻行文字重叠。
- **节点面板**：`#stepCards` 从「每事件 append 一张卡」改为每节点一张固定面板、原地更新、
  轮次进 `.sc-history` 折叠区，按 `activeSteps` 顺序插入。同条件下 7162px → 943px。
- **证据图**：舞台 703px → 472px（视口锁 `grid-template-rows: 424px`），
  指标从右上浮层改成底部状态条 + 节点流水线 `renderPipelineStrip()`。
- **报告区**：证据质量指标卡（置信度 / 分支充足 / 质量评分 / 检索来源 + 降级与预算告警）、
  sticky 章节目录（≤1000px 收起）、来源汇总（按域名分组，标注本次检索 / 历史知识 / 报告引用）、
  流式光标与定稿过渡。指标卡色调看「是否通过」而非分数——硬规则可以在 15/15 时判不通过。
- 前端 `EventSource` 的 `error` 监听器现在把服务端具名 error 的 `data` 交给 `handleSSEEvent`，
  不再当断线显示「SSE 重连中」。

### 2.4 顺手修的

- `pyproject.toml` 依赖下限本来是错的：`langfuse>=2.0` 但代码 import `langfuse.langchain` / `get_client` /
  `start_as_current_observation`（都是 v4 路径）→ 改 `>=4.0,<5`；`langgraph>=0.2` → `>=1.0`；
  version `0.1.0` 对齐到 api 报的 `0.3.0`；新增 `multilingual` 可选 extra。
- `scheduler.list_jobs()` 的 `AttributeError`：APScheduler 3.x 的 `Job` 声明了 `next_run_time` 这个 slot
  但对 pending 任务不赋值，改用 `getattr(job, "next_run_time", None)`。
- `judge_report` 对「合法 JSON 但不是对象」（数组/标量）的兜底；`except (ValueError, Exception)` 这个无效元组；
  `run_eval.py` 汇总排除 `parse_failed` 的占位分数。
- `reflector` 对非 dict evaluation、以及 `supplement_queries` 被写成字符串（会逐字符展开）的兜底。
- 后端补字段供前端展示：`reflection_score`（reflector → state → `_extract_detail`）、
  validator detail 的 `branch_total` / `branch_sufficient` / `avg_coverage` / `assessment_degraded`、
  `report` 事件的 `stop_reason` / `evidence_assessment_degraded` / `search_unavailable`。
- **Langfuse 跨境超时**：SDK 的请求超时默认只有 5 秒
  （`langfuse/_client/client.py:336` 的 `timeout or int(os.environ.get(LANGFUSE_TIMEOUT, 5))`），
  跨境访问 `cloud.langfuse.com` 会反复报 `Failed to export span batch ... Read timed out`，
  **trace 被静默丢弃**。新增 `LANGFUSE_TIMEOUT`（默认 20）和 `LANGFUSE_PROMPT_CACHE_TTL`（默认 600，
  SDK 默认 60 意味着每分钟一次冷取会阻塞节点），并在 `config.py` 里 `os.environ.setdefault` 写回环境变量
  —— SDK 客户端是「按 public key 单例、首次构造生效」，而构造点分散在 `register_prompts` /
  `get_prompt` / `CallbackHandler` / `get_client`，只有环境变量能保证所有路径一致。
  `eval/prompts.py` 同时缓存客户端实例，并给 `get_prompt` 传 `cache_ttl_seconds` / `max_retries=1` /
  `fetch_timeout_seconds`，避免可观测性组件把研究节点堵住。

---

## 三、仍然开放的问题

以下都是 2026-08-22 复查过、**确认还在**的。按「上生产前必须处理」到「可以慢慢来」排。

### 3.1 安全与暴露（上生产前必须处理）

1. **22 个路由零鉴权**，默认绑 `0.0.0.0`（`run_api.py`、`dev.sh`、`Dockerfile` 三处）。
   `DELETE /topics/{id}`、`DELETE /reports/{id}` 能删知识库；`POST /tracking/test-notification` 能刷 webhook；
   `POST /research` 每次匿名调用都花真钱。无鉴权、无限流、无问题长度上限。
2. 无 CORS 中间件、无任何 `Depends`。
3. `renderMarkdown` 是手写 sanitizer，**漏 SVG `xlink:href`**：`el.hasAttribute('href')` 对 `xlink:href` 为 false，
   属性名也不匹配任何删除条件，`<svg><use xlink:href="data:...">` 能存活。它处理的是 LLM 基于抓取网页产出的内容。
4. `marked` 从 CDN 加载且**不锁版本、无 SRI**（`index.html:7`）；`lucide` 锁了版本但同样无 SRI；
   `app.css` 顶部 `@import` Google Fonts。离线部署会直接坏掉，CDN 被投毒则 sanitizer 也一起失效。
5. `.env.example` 把第三方中转站 `https://api.world-cup.asia/v1` 设成默认 `OPENAI_API_BASE`，
   照 README 复制即把所有 prompt 和抓取到的研究内容过一遍第三方（README 里写的是 `api.openai.com`，两处不一致）。
6. DOM id 被未转义地插进 `onclick` 属性当 JS 字符串字面量（多处）。目前都是服务端 uuid4，暂不可达。

### 3.2 并发与生命周期

7. **`POST /research` 和 `POST /research/knowledge` 在 async 路由里同步跑整个 graph**
   （`api.py:161`、`api.py:256`），一个请求能卡死整个 server —— SSE ping、健康检查、其他请求全停。
   `/tracking/run` 用了 `asyncio.to_thread`，是对的，这两个没有。
8. `POST /tracking/test-notification` 在事件循环里做阻塞 `httpx` 请求（timeout=10），最多卡 10 秒。
9. `_hitl_sessions` 是**无 TTL、无上限**的内存字典，只在正常结束/异常时清理。中断后被放弃的会话会
   一直持有编译好的 graph 和整个 `MemorySaver` 状态。404 文案写着「已过期」，但没有任何东西会过期。
10. **多 worker 直接坏**：`_hitl_sessions` 和 `MemorySaver` 都是进程内的，resume 有 (N-1)/N 概率 404；
    `MemoryJobStore` 让 N 个 worker 各自加载同一批 cron 任务 → 每次定时追踪跑 N 遍、写 N 份报告、推 N 次通知。
11. **客户端取消不会停掉服务端工作**：`finally` 里的 `thread_task.cancel()` 只取消 `to_thread` 的包装 future，
    OS 线程会跑到底。点「取消研究」或关掉标签页，LLM/搜索账单照走。
12. `create_llm` 无 timeout、无 retry 上限（`utils.py`）；`search()` 也没给 Tavily 传 timeout。
    上游挂住会无限占用一个线程；配合 11 可能耗尽默认 `ThreadPoolExecutor`。
13. `asyncio.Queue()` 无 `maxsize`，`report_chunk` 用 `put_nowait` 从不 await —— 消费者慢或消失时整份报告堆在内存里。
14. Notifier 的 12 小时冷却是**进程内字典 + check-then-set**：无锁（TOCTOU），重启即重置，多副本各算一份。
15. `scheduler.stop()` 用 `shutdown(wait=False)`，进程中途重启会把 `tracking_logs` 里的行永久留在 `status='running'`，
    启动时没有任何对账逻辑。

### 3.3 数据正确性

16. `store.delete_topic()` 调 `list_reports(topic_id)` 用了默认 `limit=20` —— 超过 20 份报告的主题，
    第 21 份起的 chunks 和 facts **永久留在 ChromaDB**，`report_id` 已无法解析却仍会被检索命中。
17. `db.list_changes()` 的 `ORDER BY significance` 是按字母排：`high, low, medium` —— `low` 排在 `medium` 前面。
18. `created_at` / `triggered_at` 用 `datetime('now')`，只有秒级精度且无 tiebreaker。
    追踪流程里报告刚存就取「上一份」，同秒写入时排序不确定，diff 可能比错对象。
19. **SQLite 没有任何迁移机制**：只有 `CREATE TABLE IF NOT EXISTS`，没设 `PRAGMA user_version`，没有 `ALTER TABLE` 路径。
    以后加列会在用户数据上以 `no such column` 失败，且无法检测或修复。
20. `db.py` 每个线程一个连接且**永不关闭**（`close()` 只关调用线程自己那条）。
    `asyncio.to_thread` 和 FastAPI 的同步端点线程池（默认 40）会轮换线程，
    每条都累积一个 WAL 连接 + 一次冗余的建表 `executescript`。
21. `json.loads(None)`：`dict.get(k, "[]")` 在键存在但值为 NULL 时返回 `None`。
    `store.update_topic(id, tracking_keywords=None)` 能把行写坏，之后每次 `get_topic` 都抛 `TypeError`。
    API 层有 `is not None` 守卫，但守卫在错误的层。
22. `knowledge/store.save_report()` 是**非原子双写**：SQLite 先 commit，再写 Chroma；
    Chroma 抛错时调用方以为保存失败，但报告行已经落库且没有向量。`delete_report` 反向同理。
23. 主键是 `uuid4().hex[:12]`（48 bit），碰撞会以未捕获的 `IntegrityError` 出现。
24. Notifier **HTTP 200 带错误体算成功**：企微/钉钉 token 失效、限流、关键词拦截都返回 200 + `errcode`。
    代码只看 status_code，于是配错的 webhook 会「静默成功」并烧掉 12 小时冷却。
25. `diff_analyzer` 在跑 diff **之前**就把 tracking log 标成 `completed`；`log_id` 失效时还会再建一条，
    导致 changes 挂在副本上而 scheduler 事后更新的是原始 id。
26. `utils.normalize_url` 的 `tracking_prefixes = ("utm_", "ref", "source", ...)` 用 `startswith`，
    会连 `reference` / `refId` / `sourceId` 一起剥掉 → 不同 URL 可能归一成同一个 key 而被当重复丢弃。
27. `utils.merge_state_update` 与 `state.py` 的 reducer 是**两份独立定义**：
    前者对覆盖集之外的任意 list 都 extend，后者只给 5 个字段 `operator.add`，
    所以 `known_source_urls` 在累加器里累积、在真实 graph state 里覆盖。

### 3.4 健壮性与性能

28. `synthesizer` 里 `r['sub_question']` / `r['title']` 等是全流水线唯一不用 `.get` 的地方，
    HITL 编辑或 checkpoint 回放导致字段缺失会在节点中途抛 `KeyError`。
29. `parse_llm_json` 假定入参是 `str`；content-blocks 形式的响应会抛 `AttributeError`，
    而 `planner` 只捕获 `(JSONDecodeError, ValueError)`（这两个还是冗余的，前者是后者子类）。
30. `validator` 里 `float(r.get("score", 0))` 未捕获，非数值 score 会让节点崩。
31. `planner` 不限制子问题数量和每个子问题的查询数 —— 话多的模型会在第一轮扇出无上限的 Tavily 调用，
    预算检查在 validator 里、事后才发生。planner 也不校验 LLM 给的 `id` 是否重复（重复会静默合并分支）。
32. `synthesizer` 的 `formatted_results` 无长度上限：最多 4 轮搜索的**全文** + 上一版报告在每次改写时重发，
    没有任何 token / 字符天花板。
33. `GET /topics/{id}/reports` 的 `limit` 无上限，前端 dashboard 正在利用这一点：
    对每个主题发 `?limit=10000`（N+1 扇出，拉回全库报告正文），只为显示 5 行。
34. `eval/prompts.get_prompt` 的兜底分支里 `local_fallback.format(**kwargs)` 本身会抛 `KeyError`，
    而 `utils.get_prompt_from_langfuse` 只捕获 `ImportError` —— 降级路径比正常路径更脆。
35. `register_prompts()` 把网络故障误读成「prompt 不存在」，Langfuse 每次抖动都会给 10 个 prompt
    各推一个新的 `production` 版本。
36. `knowledge_store._extract_confidence` 是字面子串匹配，`**高**` 这种加粗写法会落到硬编码的 `中`。
37. `diff.py` 对两份报告各截断 3000 字符，但 `similarity` 是在**全文**上算的 ——
    可能判定「有变化」然后报告零变化。
38. `vector.py` 的 `metadata={"hnsw:space": "cosine"}` 对已存在的 collection 会被忽略，
    且在 chromadb 1.5.9 里是已废弃写法（现在是 `configuration=`）。若历史 collection 落在 L2，
    `max_distance=0.5` 就形同虚设，`lookup()` 永远返回空、`has_knowledge` 恒为 False。

### 3.5 打包与部署

39. **Docker 无 `VOLUME`、无 `.dockerignore`**：`DATA_DIR` 落在 `/app/data`（容器可写层），
    每次替换容器整个知识库就没了；构建上下文还会把 `.venv/`、`.git/`、`data/`、`logs/` 一起上传。
40. `CMD` 用 `uv run uvicorn`（不带 `--frozen --no-dev`），容器启动时会重新 sync、把项目装进 `/app/.venv`，
    需要可写的 `/app`，只读 rootfs 或非 root 用户会失败。
41. 容器以 **root** 运行，无 `USER`、无 `HEALTHCHECK`（尽管 `/health` 存在），`COPY` 不含 `README.md`。
42. `run_api.py` 和 `dev.sh` 都硬编码 `--reload`，唯一被文档化的启动方式是开发模式的 reloader。
43. `__init__.py` 的 `main()` 构造了 `graph` 却不用它，直接调 `run_research(question)`（死调用 + 未使用局部变量）。
44. 私有 API 依赖：`langfuse_handler._langfuse_client.flush()` 出现在 5 处；
    `scheduler._run_tracking` 也用了同一个私有属性。

### 3.6 死代码与不一致

45. `.node-detail` / `.nd-header` / `.nd-body` / `.nd-icon` / `.nd-title` / `.nd-badge` / `.step-conn` / `.step-node`
    以及 `app.css` 里整块 `.foundry-*` 规则**没有任何对应标记或 JS**，让样式表看起来覆盖得比实际好。
46. `state["key_facts"]` 从来没有节点写入过，`knowledge_store` 三级事实提取的第一级不可达。
47. `nodes/planner.py` import 了 `OPENAI_API_KEY` / `OPENAI_API_BASE` / `OPENAI_MODEL` 却不用（`create_llm` 才用）。
48. `validator` 校验了最多 3 条 `next_queries`，实际只用 `[0]`。
49. `tools/search.py` 完全没用到 `language` / `region` / `source_preference`
    （没传 Tavily 的 `country` / `topic` / `include_domains`），planner 的地区元数据除了拼进查询词以外不影响任何事。
50. `api.py` 里 `Langfuse`、`create_tracking_graph`、`ResearchState` 是未使用的 import；
    `import asyncio` 在四个函数体里各写一遍。
51. `GET /research/hitl/state` 声明了 `response_model` 但 404/500 直接返回 `JSONResponse`，
    OpenAPI 描述与实际不符；且前端从未调用它（死接口）。
52. `graph.py` 的 `should_continue` 与 `should_continue_to_store` 除终止标签外完全相同。
53. `validator._domain()` 重新实现了 `utils.normalize_url` 已有的 host 归一化；
    `diff_analyzer` 重新拼了 `utils.summarize_changes` 负责的 emoji 行；
    `searcher` 和 `validator` 各做一遍 URL 去重。

---

## 四、测试覆盖缺口

`knowledge/db.py`、`vector.py` 的写入路径、`store.py` 完全没有行为测试；
`tracking/notifier.py` 四种平台 payload 没测；
`planner` / `knowledge_lookup` / `knowledge_store` / `change_notifier` 节点函数没有单测；
HITL 的 resume SSE 生成器没有深度测试；前端 JS 没有自动化测试。

三组测试**没有 mock 出网路径**，用的是真实 `.env`，因此是网络/延迟敏感而非确定性的：
`test_diff.py::TestDiffAnalyzerAnalyze`（2 个）、`test_prompt_management.py::TestGetPromptFallback`（6 个）、
`test_hitl.py::TestHITLAPIEndpoints::test_hitl_stream_endpoint_exists`。
它们目前能过只是因为每条路径都降级到 fallback 且断言足够松。没装 `pytest-timeout`。

`tests/test_sse_stream.py` 是最有价值的一个：跑真实 `create_research_graph()` 走 `TestClient`，
只 mock LLM 和 Tavily，能离线拦住整条流水线的回归。改图或改 SSE 层先跑它。

---

## 五、验证链路

**Python**：`UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`（`UV_CACHE_DIR` 必须给，默认缓存路径不可写）。

**UI**：可以完全离线在真实浏览器里跑。`/tmp/rb_ui_server.py` 把节点模块的 `create_llm` /
`get_prompt_from_langfuse` / `search` / `validator._llm_assess` 换成假实现后启动真实 FastAPI 应用
（端口 8765，启动约 25 秒，一次假研究约 12 秒）；用 `npx playwright@1.49.0` 的 `playwright-core`
驱动 `~/.cache/ms-playwright/chromium-1208/` 里已有的 Chromium。

两个坑：
- `nodes/__init__.py` 把节点函数导出到包命名空间，`from research_buddy.nodes import planner`
  拿到的是**函数**不是模块，给它打 patch 静默无效。必须用 `importlib.import_module`。
- headless Chromium 原本一个中文字体都没有，中文全渲染成豆腐块，截图无法判断排版。
  已从 `/mnt/c/Windows/Fonts/` 复制雅黑/黑体到 `~/.local/share/fonts/` 并 `fc-cache -f`。

---

## 六、不要改回去的设计约束

- **证据门槛只能在故障时变严，不能变宽。** 每个降级分支都要问它把门槛推向哪一边；
  若答案是「更宽松」，就必须在报告里显式披露，否则是错的。
- **`validation_gaps` / `sub_questions` / `evidence_assessments` 是覆盖语义。**
  写这三个字段的节点必须考虑「我会不会把上游的发现擦掉」。
- **`synthesizer` 的 `writer` 必须标注 `StreamWriter` 且不给默认值。** 标错会静默失效而非报错。
- **`graph.stream` 的 `stream_mode` 必须是 list。** 传 tuple 会退化成裸 payload。
- **中文的 min-content 宽度是一个字。** 弹性行里的中文要 `nowrap` + 省略号，图标/角标要 `flex-shrink: 0`，
  栅格轨道写 `minmax(0, 1fr)`，可滚动纵向 flex 列表的列表项要 `flex-shrink: 0`。
  `min-width: 0` 在这里**不是**解药，它只会让塌缩更彻底。
- **报告的局限性由代码确定性追加**，不依赖模型是否照做 prompt 里的披露要求。

