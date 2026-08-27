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

# 视觉模型选图（可选）：默认与文本模型共用 OPENAI_API_KEY / OPENAI_API_BASE，
# 需要独立的中转站/密钥时单独指定（比如视觉走 DeepSeek、文本走别的站）。
# 留空 VISION_MODEL 则整个图片功能关闭，行为与无图版本一致。
VISION_MODEL = os.getenv("VISION_MODEL", "").strip()
VISION_API_KEY = os.getenv("VISION_API_KEY", "").strip()
VISION_API_BASE = os.getenv("VISION_API_BASE", "").strip()

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
# Prompt 发布必须经过文章回归和人工确认。应用启动默认不再把本地常量自动
# 标成 production；需要显式同步时运行 python -m research_buddy.eval.prompts。
LANGFUSE_AUTO_REGISTER_PROMPTS = os.getenv(
    "LANGFUSE_AUTO_REGISTER_PROMPTS", "false"
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
MAX_IMAGES_IN_ARTICLE = int(os.getenv("MAX_IMAGES_IN_ARTICLE", "8"))
# 文章正文最大生成长度（token）。0 = 不限制（默认，由模型/提供商自然决定）；
# 设具体值（如 4000）会限制 synthesizer 流式生成的长度，参考文献/图解
# 由代码追加不受影响。注意别设太小导致文章被截断。
MAX_ARTICLE_TOKENS = int(os.getenv("MAX_ARTICLE_TOKENS", "0"))
# 文章正文生成长度/温度。MAX_ARTICLE_TOKENS 见上文；WRITER_TEMPERATURE 只影响
# synthesizer 出稿（写作调用），planner/validator/reflector 等评估类调用仍保持
# 温度 0 以保证判断稳定。默认 0.9：写作任务常见的采样温度，让多次生成在结构、
# 措辞上自然有差异，而不是每次都收敛到同一篇"标准文"。
WRITER_TEMPERATURE = float(os.getenv("WRITER_TEMPERATURE", "0.9"))
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
