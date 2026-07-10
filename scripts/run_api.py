"""本地启动 API 服务

用法：
    uv run python scripts/run_api.py

访问：
    http://localhost:8000/docs — Swagger UI
"""

import uvicorn


def main():
    uvicorn.run(
        "research_buddy.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
