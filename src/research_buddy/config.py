"""配置管理 - 从 .env 读取环境变量"""

import os
from pathlib import Path
from dotenv import load_dotenv

# override=True：.env 永远是配置的事实来源。
# 否则终端里残留的 export TAVILY_API_KEY 会覆盖 .env 的新值，
# 导致「改了 .env 重启后仍用旧 key」这类难以排查的问题。
load_dotenv(override=True)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据目录（SQLite + ChromaDB 持久化）
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))

# LLM 配置（通过中转站 API）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# 少数 OpenAI 兼容中转站的 WAF 会误拦截 SDK 的 User-Agent / x-stainless-* 头。
# 默认关闭，只有原始 HTTP 可用但 OpenAI SDK 返回 403 时才启用。
OPENAI_STRIP_SDK_HEADERS = os.getenv(
    "OPENAI_STRIP_SDK_HEADERS", "false"
).strip().lower() in {"1", "true", "yes", "on"}
# Article Eval 可使用独立评审模型，降低写作模型评判自身文风的相关偏差。
# 留空或未设置时保持兼容，继续使用 OPENAI_MODEL。
ARTICLE_EVAL_JUDGE_MODEL = os.getenv("ARTICLE_EVAL_JUDGE_MODEL", "").strip() or OPENAI_MODEL
# 生产反思可使用不同于写作模型的独立 Judge。留空时保持原有行为，
# 继续使用 OPENAI_MODEL；与离线回归 Judge 分开配置，便于分别固定版本。
REFLECTOR_MODEL = os.getenv("REFLECTOR_MODEL", "").strip() or OPENAI_MODEL

# 视觉模型选图（可选）：默认与文本模型共用 OPENAI_API_KEY / OPENAI_API_BASE，
# 需要独立的中转站/密钥时单独指定（比如视觉走 DeepSeek、文本走别的站）。
# 留空 VISION_MODEL 则整个图片功能关闭，行为与无图版本一致。
VISION_MODEL = os.getenv("VISION_MODEL", "").strip()
VISION_API_KEY = os.getenv("VISION_API_KEY", "").strip()
VISION_API_BASE = os.getenv("VISION_API_BASE", "").strip()

# 生图模型（可选）：用于「搜图找不到 / 水印不通过 / 需要解释性示意图」的
# 补位。走 OpenAI 兼容的 /images/generations 接口，与文本模型共用中转站时
# 留空 IMAGE_GEN_API_KEY / IMAGE_GEN_API_BASE 即可复用 OPENAI_* 配置。
# 留空 IMAGE_GEN_MODEL 则关闭生图，配图行为退回纯搜图。
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "").strip()
IMAGE_GEN_API_KEY = os.getenv("IMAGE_GEN_API_KEY", "").strip()
IMAGE_GEN_API_BASE = os.getenv("IMAGE_GEN_API_BASE", "").strip()
# 生图尺寸（宽x高）。中转站/模型对尺寸的支持差异较大，常见值：
# 1024x1024（方图）、1536x1024（横图，适合文章插图）、1024x1536（竖图）。
# 留空则不传该字段，由服务端默认决定。
IMAGE_GEN_SIZE = os.getenv("IMAGE_GEN_SIZE", "1536x1024").strip()
# 单篇文章最多允许生图几张（含搜图失败后的兜底）。生图比搜图贵且慢，
# 独立设上限，避免一次研究被生图拖到无法接受。0 = 关闭生图。
MAX_GENERATED_IMAGES = max(0, min(8, int(os.getenv("MAX_GENERATED_IMAGES", "4"))))
# 生图单次请求超时（秒）。图像生成通常 10~60s，给足余量。
IMAGE_GEN_TIMEOUT = float(os.getenv("IMAGE_GEN_TIMEOUT", "120"))

# Langfuse 可观测性
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Langfuse SDK 的请求超时默认只有 5 秒（langfuse/_client/client.py 里
# `timeout or int(os.environ.get(LANGFUSE_TIMEOUT, 5))`），跨境访问
# cloud.langfuse.com 很容易触发 "Failed to export span batch ... Read timed out"，
# 结果是 trace 被静默丢弃。放宽到 20 秒。
LANGFUSE_TIMEOUT = int(os.getenv("LANGFUSE_TIMEOUT", "20"))

# Prompt 缓存 TTL。SDK 默认 60 秒，意味着每分钟就有一次冷取会阻塞节点；
# prompt 内容变动频率远低于此，放到 10 分钟。
LANGFUSE_PROMPT_CACHE_TTL = int(os.getenv("LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS", "600"))
# 应用启动时自动把本地 prompt 常量同步到 Langfuse（幂等：内容未变则跳过，
# 内容有变则创建新版本并打 production 标签）。默认开启，保证本地 prompt 改动
# 立即在生产流程生效，无需手动跑 python -m research_buddy.eval.prompts。
LANGFUSE_AUTO_REGISTER_PROMPTS = os.getenv(
    "LANGFUSE_AUTO_REGISTER_PROMPTS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
LANGFUSE_PROMPT_REGISTER_LABEL = os.getenv(
    "LANGFUSE_PROMPT_REGISTER_LABEL", "production"
).strip() or "production"

# Langfuse 的客户端是「按 public key 的单例，首次构造的配置生效」，
# 而构造点分散在 register_prompts / get_prompt / CallbackHandler / get_client 等处。
# 想让所有路径都拿到同一份超时配置，唯一可靠的办法是写回环境变量。
os.environ.setdefault("LANGFUSE_TIMEOUT", str(LANGFUSE_TIMEOUT))
os.environ.setdefault("LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS", str(LANGFUSE_PROMPT_CACHE_TTL))

# 工作流参数
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_REFLECTION_ROUNDS = int(os.getenv("MAX_REFLECTION_ROUNDS", "2"))
MAX_SEARCH_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "4"))
MAX_TOTAL_QUERIES = int(os.getenv("MAX_TOTAL_QUERIES", "30"))
# 文末参考文献最多展示条数（LLM 从全部来源中筛选核心子集）
MAX_REFERENCES = int(os.getenv("MAX_REFERENCES", "8"))
# 单篇文章插图数量上限（视觉选图供图 + synthesizer 插入共用）
MAX_IMAGES_IN_ARTICLE = max(0, min(12, int(os.getenv("MAX_IMAGES_IN_ARTICLE", "10"))))
# 文章正文最大生成长度（token）。0 = 不限制（默认，由模型/提供商自然决定）；
# 设具体值（如 4000）会限制 synthesizer 流式生成的长度，参考文献/图解
# 由代码追加不受影响。注意别设太小导致文章被截断。
MAX_ARTICLE_TOKENS = int(os.getenv("MAX_ARTICLE_TOKENS", "0"))
# 服务商因单次输出上限截断正文时，最多自动续写几次。续写仍使用同一写作模型，
# 每次都携带已完成正文与编辑简报，避免从头重写。0 = 关闭自动续写。
MAX_ARTICLE_CONTINUATIONS = max(0, min(4, int(os.getenv("MAX_ARTICLE_CONTINUATIONS", "2"))))
# 文章正文生成长度/温度。MAX_ARTICLE_TOKENS 见上文；WRITER_TEMPERATURE 只影响
# synthesizer 出稿（写作调用），planner/validator/reflector 等评估类调用仍保持
# 温度 0 以保证判断稳定。默认 0.5：保留措辞变化，同时降低长文在高温度下
# 出现术语漂移、格式损坏和重写轮质量波动的概率。
WRITER_TEMPERATURE = max(0.0, min(1.0, float(os.getenv("WRITER_TEMPERATURE", "0.5"))))
# 初稿后的事实审校。默认只扫描 1 轮；存在非空替换时会再用
# 1 次独立调用复核证据支持，因此每次出稿最多新增 2 次 LLM 调用。
ENABLE_ARTICLE_EDITOR = os.getenv(
    "ENABLE_ARTICLE_EDITOR", "true"
).strip().lower() in {"1", "true", "yes", "on"}
# 生产只允许关闭或扫描一轮，确保审校 + 独立复核最多增加 2 次调用。
# 离线评测可通过函数参数显式使用更多轮次。
ARTICLE_EDITOR_ROUNDS = min(1, max(0, int(os.getenv("ARTICLE_EDITOR_ROUNDS", "1"))))
# 只在确定性扫描发现高置信模板句时调用一次语言审校模型。
ENABLE_LANGUAGE_EDITOR = os.getenv(
    "ENABLE_LANGUAGE_EDITOR", "true"
).strip().lower() in {"1", "true", "yes", "on"}
MIN_EVIDENCE_COVERAGE = float(os.getenv("MIN_EVIDENCE_COVERAGE", "0.75"))
# LLM 语义评估的覆盖度门槛（软闸）。评估器把 coverage 理解为「核心结论可信度」
# 而非「所有细节完备度」，复杂话题给 0.6 以上即可视为语义充足，避免永远补搜
# 耗尽预算。确定性硬闸（MIN_EVIDENCE_COVERAGE）不受影响，仍要求数量/域名/覆盖度达标。
MIN_SEMANTIC_COVERAGE = float(os.getenv("MIN_SEMANTIC_COVERAGE", "0.6"))
MIN_DISTINCT_DOMAINS = int(os.getenv("MIN_DISTINCT_DOMAINS", "2"))
MIN_RESULTS_PER_SUB_QUESTION = int(os.getenv("MIN_RESULTS_PER_SUB_QUESTION", "2"))
MIN_SEARCH_CONTENT_LENGTH = int(os.getenv("MIN_SEARCH_CONTENT_LENGTH", "80"))

# 搜索 API
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# 向量检索 embedding 后端
#   default              —— ChromaDB 内置 all-MiniLM-L6-v2（英文为主，中文检索质量有限）
#   sentence-transformers —— 本地多语言模型，需要可选依赖：uv sync --extra multilingual
#   openai               —— 走 OPENAI_API_BASE 的 /embeddings 接口（中转站可能不支持）
# 后端不可用时会打 WARNING 并降级到 default，不会静默切换。
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "default").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()

# 通知 Webhook（可选，支持企业微信/钉钉/Telegram）
NOTIFICATION_WEBHOOK_URL = os.getenv("NOTIFICATION_WEBHOOK_URL", "")

# 本地代码/文件检索：让研究能读取本地代码库作为证据来源（与 Tavily web 搜索平行）。
# LOCAL_REPO_ROOT 为空或目录不存在时，本地检索整体关闭，行为与旧版一致。
LOCAL_REPO_ROOT = os.getenv("LOCAL_REPO_ROOT", "").strip()
# 是否自动把「当前工作目录」作为本地检索根目录（LOCAL_REPO_ROOT 为空时生效）。
LOCAL_AUTO_ROOT = os.getenv("LOCAL_AUTO_ROOT", "true").strip().lower() in {"1", "true", "yes", "on"}
if not LOCAL_REPO_ROOT and LOCAL_AUTO_ROOT:
    # 未显式指定目标仓库时，默认检索启动服务时所在的工作目录，开箱可用。
    LOCAL_REPO_ROOT = os.getcwd()
# grep 单次最多返回的命中数（避免把海量匹配灌进上下文）。
LOCAL_GREP_MAX_RESULTS = int(os.getenv("LOCAL_GREP_MAX_RESULTS", "40"))
# 单文件读取大小上限（字节），超过则跳过。
LOCAL_FILE_MAX_BYTES = int(os.getenv("LOCAL_FILE_MAX_BYTES", str(200 * 1024)))
# 本地证据单文件最多保留的字符数（进入证据账本前的截断）。
# 之前 4000 字会截断中等规模源码文件（如 300+ 行的模块），导致 agent 读不到
# 完整函数体；放宽到 8000 字，配合 read_local_file 的分段续读与按 path 合并。
LOCAL_EXCERPT_CHARS = int(os.getenv("LOCAL_EXCERPT_CHARS", "8000"))

# 本地 agentic loop：LLM 自主决定读哪个文件、grep 什么、执行哪些只读命令，并据结果
# 迭代探索，直到收集够写文档的证据（ReAct 模式）。关闭则退化为单轮启发式检索。
LOCAL_AGENTIC_ENABLED = os.getenv("LOCAL_AGENTIC_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
# agentic loop 单次研究最多执行的探索步数（每步 = 一次 LLM 工具调用决策）。
# 深度研究需要更多步数才能充分探索 + 交叉验证，默认 20。
LOCAL_AGENTIC_MAX_STEPS = int(os.getenv("LOCAL_AGENTIC_MAX_STEPS", "20"))
# 是否允许 agentic loop 使用只读 shell（git log/show/diff、find、ls、tree 等）。
# 默认关闭——shell 执行有安全风险，需显式开启。
LOCAL_AGENTIC_SHELL_ENABLED = os.getenv("LOCAL_AGENTIC_SHELL_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

# 是否允许 agentic loop 使用「执行验证」能力：跑代码片段（python -c/node -e）、
# 查真实版本（pip show、--version）、收集测试结构（pytest --collect-only）。
# 这是写文档的关键验证手段（确认示例能跑、版本准确），但会执行任意代码片段，
# 以当前用户权限运行，默认关闭，需显式开启。仍不开放写文件/改源码/任意 shell。
LOCAL_AGENTIC_EXEC_ENABLED = os.getenv("LOCAL_AGENTIC_EXEC_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
# 执行验证单条命令的超时（秒）。
LOCAL_EXEC_TIMEOUT = int(os.getenv("LOCAL_EXEC_TIMEOUT", "30"))

# 网页全文抓取（agentic 研究循环的 web 证据来源，弥补 Tavily 只有摘要的短板）。
# fetch_webpage 抓取 URL 全文并提取正文；失败返回 None 不阻断研究。
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "15"))
FETCH_MAX_BYTES = int(os.getenv("FETCH_MAX_BYTES", str(512 * 1024)))
FETCH_MAX_TEXT_CHARS = int(os.getenv("FETCH_MAX_TEXT_CHARS", "12000"))
# 单次 agentic 研究中，最多抓取多少个网页全文（控制成本与上下文膨胀）。
# 深度研究需要多抓几个权威来源交叉验证，默认 10。
FETCH_MAX_PAGES = int(os.getenv("FETCH_MAX_PAGES", "10"))

# 在线自动打分：研究流程结束后，对拆解/证据/编辑简报/最终报告各做一次 LLM-as-Judge，
# 并把分数写回 Langfuse trace 与本地素材库（前端「流程打分」面板展示）。
# 默认开启——每轮研究额外增加约 3~4 次 LLM 调用；如需关闭设 AUTO_JUDGE_PROCESS=false。
AUTO_JUDGE_PROCESS = os.getenv(
    "AUTO_JUDGE_PROCESS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
