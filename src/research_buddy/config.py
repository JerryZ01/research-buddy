"""配置管理 - 从 .env 读取环境变量"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据目录（SQLite + ChromaDB 持久化）
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))

# LLM 配置（通过中转站 API）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# 视觉模型选图（可选）：配置后文章会用视觉模型从搜索结果中挑选相关插图。
# 留空则整个图片功能关闭，行为与无图版本一致（推荐中转站的 vision 模型，
# 如 v4-flash-vision-exp）。
VISION_MODEL = os.getenv("VISION_MODEL", "").strip()

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
MIN_EVIDENCE_COVERAGE = float(os.getenv("MIN_EVIDENCE_COVERAGE", "0.75"))
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
