#!/usr/bin/env bash
# Research Buddy 启动/停止/重启脚本
#
# 用法：
#   ./scripts/dev.sh          启动（自动杀旧进程）
#   ./scripts/dev.sh start    启动
#   ./scripts/dev.sh stop     停止
#   ./scripts/dev.sh restart  重启
#   ./scripts/dev.sh status   查看状态

set -e

PORT=8000
HOST="0.0.0.0"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/api.log"

mkdir -p "$LOG_DIR"

# 获取占用端口的 PID
get_pid() {
    lsof -ti :$PORT 2>/dev/null || true
}

# 杀掉占用端口的进程
kill_port() {
    local pid
    pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo "🛑 发现端口 $PORT 被进程 $pid 占用，正在终止..."
        kill $pid 2>/dev/null || true
        # 等待最多 5 秒让进程退出
        for i in $(seq 1 10); do
            if ! get_pid >/dev/null 2>&1; then
                break
            fi
            sleep 0.5
        done
        # 如果还没退出，强杀
        pid=$(get_pid)
        if [ -n "$pid" ]; then
            echo "⚠️  进程未响应，强制终止..."
            kill -9 $pid 2>/dev/null || true
            sleep 0.5
        fi
        echo "✅ 旧进程已终止"
    fi
}

cmd_start() {
    # 先杀旧进程
    kill_port

    echo "🚀 启动 Research Buddy API..."
    echo "   地址: http://localhost:$PORT"
    echo "   日志: $LOG_FILE"
    echo "   按 Ctrl+C 停止"
    echo ""

    uv run uvicorn research_buddy.api:app \
        --host $HOST \
        --port $PORT \
        --reload \
        --log-level info
}

cmd_stop() {
    local pid
    pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo "🛑 正在停止 Research Buddy (PID: $pid)..."
        kill_port
        echo "✅ 已停止"
    else
        echo "ℹ️  端口 $PORT 没有运行中的服务"
    fi
}

cmd_restart() {
    echo "🔄 重启 Research Buddy..."
    kill_port
    sleep 1
    cmd_start
}

cmd_status() {
    local pid
    pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo "✅ Research Buddy 正在运行 (PID: $pid, 端口: $PORT)"
        # 试试健康检查
        if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
            echo "   健康检查: ✅ 正常"
        else
            echo "   健康检查: ⚠️  无响应（可能正在启动中）"
        fi
    else
        echo "❌ Research Buddy 未运行"
    fi
}

# 解析命令
case "${1:-start}" in
    start)   cmd_start   ;;
    stop)    cmd_stop    ;;
    restart) cmd_restart ;;
    status)  cmd_status  ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        echo ""
        echo "  start    启动服务（默认，自动杀旧进程）"
        echo "  stop     停止服务"
        echo "  restart  重启服务"
        echo "  status   查看运行状态"
        exit 1
        ;;
esac
