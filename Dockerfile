FROM python:3.11-slim

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存
COPY pyproject.toml uv.lock* ./

# 安装依赖
RUN uv sync --frozen --no-dev --no-install-project

# 复制源码
COPY src/ src/
COPY .env.example .env.example

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uv", "run", "uvicorn", "research_buddy.api:app", "--host", "0.0.0.0", "--port", "8000"]
