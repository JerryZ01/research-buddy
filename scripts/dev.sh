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

# 杀掉整棵 research-buddy 进程树（reload 父 + worker + dev.sh 包装）
# 关键：uvicorn --reload 是「reload 父进程 + worker 子进程」两个进程，
# worker 占端口、父进程不占端口。只杀端口占用者的话，reload 父会带着
# 旧环境变量重新拉起 worker —— 这就是改 .env 不生效、进程越堆越多的根源。
kill_port() {
    # 1) 杀所有 uvicorn（cmdline 都含 research_buddy.api，reload 父 + worker 一网打尽）
    pkill -9 -f "research_buddy.api" 2>/dev/null || true
    sleep 0.5
    # 2) 端口兜底（万一有非 uvicorn 进程占着）
    local pid
    pid=$(get_pid)
    if [ -n "$pid" ]; then
        echo "🛑 发现端口 $PORT 残留进程 $pid，强制终止..."
        echo "$pid" | xargs -r kill -9 2>/dev/null || true
    fi
    # 3) 清理遗留的 dev.sh 包装进程（排除自己，防止自杀）
    local me=$$
    for p in $(pgrep -f "scripts/dev.sh" 2>/dev/null); do
        if [ "$p" != "$me" ] && [ "$p" != "$PPID" ]; then
            kill -9 "$p" 2>/dev/null || true
        fi
    done
    sleep 1
    if get_pid >/dev/null 2>&1; then
        echo "⚠️  端口 $PORT 仍被占用（可能是其他终端/沙箱外的进程），请手动检查"
    else
        echo "✅ 端口已释放"
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
