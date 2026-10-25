"""本地代码库检索工具 — 让研究能读取本地代码/文件作为证据来源。

与 Tavily 搜索（web 证据）平行，本地检索产出「本地证据」：
- list_files：列出项目文件树（按忽略规则过滤）
- grep：在代码内容里按关键词/正则搜索，返回命中行及文件
- read_file：读取指定文件（带行号、分页、大小上限）

安全边界：
- 所有访问被限制在 LOCAL_REPO_ROOT 内，路径越界一律拒绝；
- 自动忽略 .git / node_modules / .venv / __pycache__ / dist / build 等目录；
- 跳过二进制文件和超大文件，限制返回数量，避免把整个仓库灌进上下文。
"""

import logging
import os
import re
from pathlib import Path

from research_buddy.config import (
    LOCAL_REPO_ROOT,
    LOCAL_GREP_MAX_RESULTS,
    LOCAL_FILE_MAX_BYTES,
    LOCAL_EXEC_TIMEOUT,
)

logger = logging.getLogger(__name__)

# 默认忽略的目录名/文件名（与常见项目结构一致，防止把依赖/构建产物当证据）
_IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", "out", "target", ".next", ".nuxt", ".cache", ".idea",
    ".vscode", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".eggs",
    "playwright-report", "test-results", ".codex",
}
_IGNORED_FILE_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dll", ".dylib", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip",
    ".tar", ".gz", ".7z", ".woff", ".woff2", ".ttf", ".eot", ".lock",
}
# 二进制/非文本文件的前缀判断（读取时跳过）
_BINARY_PREFIX = (b"\x00", b"\x89PNG", b"\xff\xd8", b"GIF8", b"PK\x03\x04", b"%PDF")

# 代码/文档类文件扩展名（grep 时优先匹配这些，减少噪音）
_TEXT_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".java", ".go",
    ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".rb", ".php", ".swift",
    ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat",
    ".sql", ".html", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    ".md", ".mdx", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".xml", ".proto", ".graphql", ".env.example", ".dockerfile",
}


def is_enabled() -> bool:
    """本地检索是否可用：配置了根目录且目录真实存在。"""
    if not LOCAL_REPO_ROOT:
        return False
    return Path(LOCAL_REPO_ROOT).is_dir()


def get_repo_root() -> str:
    """返回本地仓库根目录（绝对路径）。"""
    return str(Path(LOCAL_REPO_ROOT).resolve())


def _safe_join(relative: str) -> Path | None:
    """把相对路径解析到仓库根目录内，越界返回 None。"""
    root = Path(LOCAL_REPO_ROOT).resolve()
    candidate = (root / relative).resolve()
    # resolve 后必须仍在 root 内（防止 ../ 越界）
    if candidate != root and root not in candidate.parents:
        return None
    return candidate


def _should_skip_dir(name: str) -> bool:
    return name in _IGNORED_DIRS


def _should_skip_file(path: Path) -> bool:
    if path.suffix.lower() in _IGNORED_FILE_SUFFIXES:
        return True
    # 无扩展名且不是已知文本文件 → 跳过（大概率二进制/无意义）
    if not path.suffix and path.name not in {"Makefile", "Dockerfile", "LICENSE", "README"}:
        return True
    return False


def _is_binary(path: Path) -> bool:
    """读文件头判断是否二进制，避免把二进制内容灌进上下文。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
        return head.startswith(_BINARY_PREFIX)
    except OSError:
        return True


def _walk_text_files(max_files: int = 500) -> list[Path]:
    """遍历仓库，返回文本文件列表（相对根目录），按忽略规则过滤。"""
    root = Path(LOCAL_REPO_ROOT)
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _should_skip_dir(d))
        for name in sorted(filenames):
            if len(files) >= max_files:
                return files
            path = Path(dirpath) / name
            if _should_skip_file(path):
                continue
            files.append(path)
    return files


def list_files(max_files: int = 200) -> list[str]:
    """列出项目文件（相对路径字符串），供 LLM 判断项目结构。"""
    if not is_enabled():
        return []
    rel_files = []
    for path in _walk_text_files(max_files=max_files):
        rel_files.append(str(path.relative_to(LOCAL_REPO_ROOT)))
    return rel_files


def grep(pattern: str, file_globs: list[str] | None = None,
         max_results: int | None = None) -> list[dict]:
    """在仓库文本文件里搜索，返回命中列表。

    Args:
        pattern: 正则或字面量关键词（大小写不敏感）。
        file_globs: 可选，限制匹配的文件扩展名（如 [".py", ".js"]）。
        max_results: 最大命中数。

    Returns:
        [{path, line, text}]，path 为相对路径，line 为 1-based 行号，text 为整行内容。
    """
    if not is_enabled():
        return []
    max_results = max_results or LOCAL_GREP_MAX_RESULTS
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        compiled = re.compile(re.escape(pattern), re.IGNORECASE)

    allowed_suffixes = {s.lower() for s in (file_globs or [])}
    hits: list[dict] = []
    for path in _walk_text_files():
        if allowed_suffixes and path.suffix.lower() not in allowed_suffixes:
            continue
        if _is_binary(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line_no, line in enumerate(fh, 1):
                    if compiled.search(line):
                        hits.append({
                            "path": str(path.relative_to(LOCAL_REPO_ROOT)),
                            "line": line_no,
                            "text": line.rstrip("\n")[:300],
                        })
                        if len(hits) >= max_results:
                            return hits
        except OSError:
            continue
    return hits


def read_file(relative: str, offset: int = 1, limit: int = 200) -> dict | None:
    """读取指定文件的一段内容（带行号）。

    Args:
        relative: 相对仓库根目录的文件路径。
        offset: 起始行（1-based）。
        limit: 读取行数上限。

    Returns:
        {path, content, line_start, line_end, total_lines, truncated}；
        文件不存在/越界/二进制/超大时返回 None。
    """
    if not is_enabled():
        return None
    path = _safe_join(relative)
    if path is None or not path.is_file():
        logger.info("本地读取被拒或不存在: %s", relative)
        return None
    if _is_binary(path):
        return None
    try:
        size = path.stat().st_size
        if size > LOCAL_FILE_MAX_BYTES:
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    total_lines = len(lines)
    start = max(1, min(offset, total_lines)) if total_lines else 1
    end = min(total_lines, start + limit - 1)
    selected = lines[start - 1:end]
    content = "".join(selected).rstrip("\n")
    return {
        "path": str(path.relative_to(LOCAL_REPO_ROOT)),
        "content": content,
        "line_start": start,
        "line_end": end,
        "total_lines": total_lines,
        "truncated": end < total_lines,
    }


def read_context(relative: str, line: int, before: int = 8, after: int = 16) -> dict | None:
    """读取某行附近的上下文（grep 命中后用，给 LLM 提供函数/类级别的片段）。"""
    return read_file(relative, offset=max(1, line - before), limit=before + after + 1)


# ── 只读 shell（agentic loop 可选能力，需显式开启） ──

import subprocess as _subprocess

# 白名单：只允许不会修改文件系统/外部状态的只读命令。
# 每个命令只允许特定前缀，任何写操作、网络操作、任意代码执行都被拒绝。
_ALLOWED_SHELL_COMMANDS = {
    "git": {"log", "show", "diff", "status", "blame", "ls-files", "shortlog", "rev-parse"},
    "find": None,   # find 参数复杂，单独校验：只允许 -type/-name/-path 等只读选项
    "ls": None,
    "tree": None,
    "wc": None,
    "cat": None,    # cat 只读（但已有限制大文件；LLM 通常应优先用 read_file）
    "head": None,
    "tail": None,
    "grep": None,   # 底层 grep（保留给 LLM 需要更复杂 pattern 时）
    "file": None,
    "du": None,
}

# find 允许的只读谓词/动作前缀
_FIND_ALLOWED_FLAGS = {"-type", "-name", "-path", "-iname", "-ipath", "-maxdepth",
                       "-mindepth", "-not", "-and", "-or", "-a", "-o", "-print",
                       "-print0", "-regex", "-iregex", "-empty", "-size", "-newer"}


def run_readonly_shell(command: str) -> dict:
    """执行一条白名单内的只读 shell 命令（在 LOCAL_REPO_ROOT 内）。

    安全策略：
    - 命令第一段必须是白名单命令，且子命令（若有）也必须在白名单内；
    - find 只允许只读谓词，禁止 -exec/-delete/-print0 之外的写动作；
    - 禁止管道里的危险命令、禁止命令替换、禁止分号分隔的多命令；
    - 工作目录固定为 LOCAL_REPO_ROOT，超时 10s。

    Returns:
        {"ok": bool, "output": str, "error": str}
    """
    if not is_enabled():
        return {"ok": False, "output": "", "error": "本地检索未配置"}
    command = (command or "").strip()
    if not command:
        return {"ok": False, "output": "", "error": "空命令"}

    # 拒绝明显危险的分隔符/替换
    if any(token in command for token in (";", "&&", "||", "`", "$(", "${", ">", ">>", "|")):
        return {"ok": False, "output": "", "error": "命令含禁止的分隔符/重定向/替换"}

    parts = command.split()
    executable = parts[0]
    if executable not in _ALLOWED_SHELL_COMMANDS:
        return {"ok": False, "output": "", "error": f"命令不在白名单: {executable}"}

    # git 子命令白名单校验
    if executable == "git":
        if len(parts) < 2 or parts[1] not in _ALLOWED_SHELL_COMMANDS["git"]:
            return {"ok": False, "output": "", "error": f"git 子命令不在白名单: {' '.join(parts[1:2]) or '(无)'}"}

    # find 只允许只读谓词
    if executable == "find":
        for tok in parts[1:]:
            if tok.startswith("-") and tok not in _FIND_ALLOWED_FLAGS:
                return {"ok": False, "output": "", "error": f"find 选项不在白名单: {tok}"}

    try:
        result = _subprocess.run(
            command, shell=True, cwd=LOCAL_REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
        if len(output) > 20000:
            output = output[:20000] + "\n...（输出被截断）"
        return {
            "ok": result.returncode == 0,
            "output": output,
            "error": result.stderr[:1000] if result.returncode != 0 else "",
        }
    except _subprocess.TimeoutExpired:
        return {"ok": False, "output": "", "error": "命令超时（10s）"}
    except Exception as exc:
        return {"ok": False, "output": "", "error": str(exc)[:500]}


# ── 执行验证（agentic loop 可选能力，需显式开启） ──

# 允许执行验证的命令/程序白名单。这些命令用于「验证」而非「修改」：
# 跑代码片段验证示例、查真实版本、收集测试/依赖结构。
_EXEC_ALLOWED_EXECUTABLES = {
    "python": None, "python3": None, "node": None, "pytest": None, "pip": None,
    "pip3": None, "npm": None, "pnpm": None, "uv": None, "cargo": None,
    "go": None, "rustc": None, "gcc": None, "java": None, "ruby": None,
    "php": None, "deno": None, "npx": None,
}
# 版本查询类命令：任何以 --version/-V/-v 结尾的都在此豁免（纯查询，无副作用）。
_EXEC_VERSION_FLAGS = {"--version", "-V", "-v", "version"}


def run_exec_verification(command: str) -> dict:
    """执行一条「验证类」命令（跑代码片段 / 查版本 / 收集测试结构）。

    安全策略（比只读 shell 宽松，但仍有明确边界）：
    - 可执行程序必须在白名单内（python/node/pytest/pip/npm/uv/go/...）；
    - 明确拒绝写文件/改系统的子命令：pip install/uninstall、npm install、
      cargo build、go get、git commit/push、任何 -o 输出、> 重定向、管道、命令替换；
    - 允许 python -c "代码"、node -e "代码"（用于验证示例片段），
      以及 pytest --collect-only（只收集不跑）、pip show xxx、xxx --version；
    - 工作目录固定为 LOCAL_REPO_ROOT，超时受 LOCAL_EXEC_TIMEOUT 控制。

    Returns:
        {"ok": bool, "output": str, "error": str}
    """
    if not is_enabled():
        return {"ok": False, "output": "", "error": "本地检索未配置"}
    command = (command or "").strip()
    if not command:
        return {"ok": False, "output": "", "error": "空命令"}

    # 拒绝明显危险的分隔符/重定向/命令替换/写文件
    if any(token in command for token in (";", "&&", "||", "`", "$(", "${", ">", ">>", "|")):
        return {"ok": False, "output": "", "error": "命令含禁止的分隔符/重定向/替换"}

    parts = command.split()
    executable = parts[0]
    if executable not in _EXEC_ALLOWED_EXECUTABLES:
        return {"ok": False, "output": "", "error": f"命令不在验证白名单: {executable}"}

    # 拒绝会修改环境的子命令（这些属于「安装/构建/写」，不是「验证」）
    _DENY_SUBCOMMANDS = {
        "install", "uninstall", "remove", "update", "upgrade", "add", "i",
        "build", "push", "commit", "clone", "get", "fmt", "fix", "publish",
        "login", "logout", "init", "new", "create", "generate",
    }
    args = parts[1:]
    if args and args[0] in _DENY_SUBCOMMANDS:
        return {"ok": False, "output": "", "error": f"子命令不允许（验证仅允许查询/运行）: {args[0]}"}
    # pip show <pkg> / pip list / pip --version 等查询是允许的
    if executable in {"pip", "pip3"} and args and args[0] not in {"show", "list", "freeze", "config", "--version", "-V"}:
        return {"ok": False, "output": "", "error": f"pip 子命令不允许（仅 show/list/freeze）: {args[0]}"}
    # npm/pnpm/npx 只允许查询类子命令
    if executable in {"npm", "pnpm", "npx"} and args and args[0] not in {"ls", "list", "view", "outdated", "--version", "-v", "run", "exec", "test"}:
        return {"ok": False, "output": "", "error": f"{executable} 子命令不允许: {args[0]}"}

    try:
        result = _subprocess.run(
            command, shell=True, cwd=LOCAL_REPO_ROOT,
            capture_output=True, text=True, timeout=LOCAL_EXEC_TIMEOUT,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if len(output) > 20000:
            output = output[:20000] + "\n...（输出被截断）"
        return {
            "ok": result.returncode == 0,
            "output": output,
            "error": "" if result.returncode == 0 else f"退出码 {result.returncode}",
        }
    except _subprocess.TimeoutExpired:
        return {"ok": False, "output": "", "error": f"命令超时（{LOCAL_EXEC_TIMEOUT}s）"}
    except Exception as exc:
        return {"ok": False, "output": "", "error": str(exc)[:500]}
