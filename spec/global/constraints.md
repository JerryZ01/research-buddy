# 项目架构约束

![技术栈概览](./images/06-tech-stack.png)

## 技术栈
- **语言:** Python 3.11+
- **前端框架:** 原生 JavaScript（无构建步骤 SPA，深色主题）
- **后端框架:** FastAPI + Uvicorn + sse-starlette
- **工作流编排:** LangGraph 1.x（StateGraph、条件边、循环、HITL interrupt）
- **LLM 访问:** langchain-openai（OpenAI 兼容 API）
- **数据库:** SQLite（WAL 模式、线程本地连接、外键约束）
- **向量检索:** ChromaDB（可选 default / sentence-transformers / openai 三种 embedding 后端）
- **搜索:** Tavily Search API
- **可观测性:** Langfuse 4.x（Trace、Generation、Score、Prompt 管理）
- **定时任务:** APScheduler（AsyncIOScheduler + CronTrigger）
- **构建工具:** hatchling（wheel 构建）
- **包管理器:** uv（Python）+ npm（前端测试工具 Playwright / mermaid-cli）

## 架构决策
- **目录结构:** `src/` 布局，包名 `research_buddy`，测试与脚本分离（`tests/`、`scripts/`）。
- **分层架构:** 展示层 → API 层 → 编排层 → 节点层（研究/编辑/质量）→ 工具层 + 知识/追踪/档案/评估，职责分离。
- **状态管理:** `ResearchState` TypedDict 作为所有图共享状态，字段分「追加」（`operator.add`）与「覆盖」两种 reducer 语义。
- **通信模式:** HTTP + SSE 流式；LangGraph 后台线程运行，浏览器断开不取消研究，`runs` 表保存状态供恢复。
- **证据与模型分离:** 搜索证据与模型判断分离，零证据不允许模型凭内部知识出稿（fail closed）。

## API 风格
- **风格:** RESTful（JSON）+ SSE 流式端点。
- **认证方式:** 无内置用户认证；外部服务通过各自 API Key（`OPENAI_API_KEY`、`TAVILY_API_KEY`、`LANGFUSE_*`）认证。
- **错误处理:** 关键失败显式报错（如 Tavily 缺失中止研究）；可选能力失败降级（图片、Langfuse、编辑节点失败保留原稿）并在响应中给出明确错误/降级标记。

## 编码规范
- **命名约定:** Python 模块/函数/变量 snake_case；环境变量全大写下划线；状态字段 snake_case。
- **文件组织:** 每个 LangGraph 节点一个模块（`nodes/`）；领域能力按 `knowledge/`、`tracking/`、`eval/` 分组；公共能力下沉到 `tools/` 与 `utils.py`。
- **模型调用纪律:** 评估类节点恒温 0，写作节点单独温度；LLM 输出必须经 JSON 形状、范围、引用、原文存在性校验；可由代码确定的内容不交给模型决定。
- **预算约束:** 搜索、反思、续写、图片数量均设显式预算（`MAX_SEARCH_ROUNDS`、`MAX_TOTAL_QUERIES`、`MAX_REFLECTION_ROUNDS`、`MAX_ARTICLE_CONTINUATIONS`、`MAX_IMAGES_IN_ARTICLE`），避免无界循环与费用失控。

## 部署方式
- **环境:** 本地开发（`./scripts/dev.sh`）与生产均可用 Docker（`python:3.11-slim` + uv）单容器部署。
- **CI/CD:** （未检测到）——未发现 `.github/workflows` 等 CI 配置。
- **运行时:** 单机单进程；FastAPI lifespan 启动时初始化 embedding 后端、启动 APScheduler、幂等回填旧报告。

## 安全约束
- **密钥管理:** 所有密钥经 `.env` 读取，`.env.example` 仅含占位符，`.env` 不入库（gitignore）。
- **输入校验:** API 层做请求校验；图片下载限制单图 ≤3MB、最小尺寸 400×300，过滤极端宽高比。
- **图片内容安全:** 视觉模型用「先抄写叠加内容、再判定 watermark_status」的证据优先问法，入选图再逐张独立复核；两处都 fail-closed，拒绝水印、署名、账号 ID、二维码、版权叠印。
- **内容可信:** 事实审校要求「原文精确片段 + 证据支持 + 独立 grounding 复核」，不引入未经支持的新事实；素材批准不自动污染知识库（防污染保守设计）。
- **SQL 安全:** SQLite 使用参数化查询、外键约束、线程本地连接。

---
*最后更新: 2026-09-03 — 初始化生成*
