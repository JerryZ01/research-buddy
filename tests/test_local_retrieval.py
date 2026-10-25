"""本地代码检索工具的单元测试。

覆盖：
- list_files / grep / read_file / read_context 的基础行为与安全边界
- 本地证据正确合并进证据账本与参考文献表
"""

from importlib import import_module

import pytest

local = import_module("research_buddy.tools.local")
editorial_planner = import_module("research_buddy.nodes.editorial_planner")
synthesizer = import_module("research_buddy.nodes.synthesizer")


@pytest.fixture
def repo(tmp_path):
    """构造一个最小可检索仓库。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def build_evidence_ledger(state):\n    return []\n\n"
        "def build_source_table(state):\n    return []\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Demo\n\nThis is a demo project.\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("// ignored\n", encoding="utf-8")
    (tmp_path / "binary.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return tmp_path


def _patch_root(monkeypatch, root):
    monkeypatch.setattr(local, "LOCAL_REPO_ROOT", str(root))


def test_list_files_excludes_ignored_dirs(monkeypatch, repo):
    _patch_root(monkeypatch, repo)
    files = local.list_files(max_files=100)
    paths = set(files)
    assert "src/main.py" in paths
    assert "README.md" in paths
    # node_modules 被忽略
    assert not any("node_modules" in p for p in paths)
    # 二进制文件被跳过
    assert "binary.png" not in paths


def test_grep_finds_matches(monkeypatch, repo):
    _patch_root(monkeypatch, repo)
    hits = local.grep("build_evidence_ledger", file_globs=[".py"])
    assert hits, "应在 main.py 命中"
    assert hits[0]["path"] == "src/main.py"
    assert hits[0]["line"] == 1


def test_grep_respects_file_globs(monkeypatch, repo):
    _patch_root(monkeypatch, repo)
    # 只搜 .md，不应命中 .py 里的函数名
    hits = local.grep("build_evidence_ledger", file_globs=[".md"])
    assert hits == []


def test_read_file_with_line_numbers(monkeypatch, repo):
    _patch_root(monkeypatch, repo)
    data = local.read_file("src/main.py", offset=1, limit=1)
    assert data is not None
    assert data["line_start"] == 1
    assert data["line_end"] == 1
    assert "build_evidence_ledger" in data["content"]


def test_read_file_rejects_path_traversal(monkeypatch, repo):
    _patch_root(monkeypatch, repo)
    # 越界路径必须被拒绝
    assert local.read_file("../outside.py") is None


def test_read_file_skips_binary(monkeypatch, repo):
    _patch_root(monkeypatch, repo)
    assert local.read_file("binary.png") is None


def test_is_enabled_false_when_root_missing(monkeypatch, repo):
    _patch_root(monkeypatch, repo / "does-not-exist")
    assert local.is_enabled() is False
    assert local.list_files() == []
    assert local.grep("anything") == []


def test_local_evidence_merges_into_ledger():
    state = {
        "search_results": [
            {"title": "Web", "url": "https://example.com/a", "content": "web",
             "score": 0.8, "sub_question_id": "sq_01"},
        ],
        "local_evidence": [
            {"path": "src/main.py", "title": "本地文件：src/main.py",
             "content": "def f(): pass", "score": 1.0},
        ],
        "evidence_assessments": [],
    }
    ledger = editorial_planner.build_evidence_ledger(state)
    ids = [e["id"] for e in ledger]
    assert ids == ["E1", "E2"]
    # agentic 证据（local_evidence）现在优先于 Tavily 摘要，排在账本前部
    local_entry = ledger[0]
    assert local_entry["source_type"] == "local"
    assert local_entry["assessment_status"] == "local"
    assert local_entry["path"] == "src/main.py"
    # Tavily 摘要排在其后
    assert ledger[1]["url"] == "https://example.com/a"


def test_local_evidence_merges_into_source_table():
    state = {
        "search_results": [
            {"title": "Web", "url": "https://example.com/a", "content": "web"},
        ],
        "local_evidence": [
            {"path": "src/main.py", "title": "本地文件：src/main.py", "content": "x"},
        ],
        "known_source_urls": [],
    }
    table = synthesizer.build_source_table(state)
    sources = [t["source"] for t in table]
    assert sources == ["search", "local"]
    assert table[1]["url"] == "src/main.py"


# ── 网页正文提取 ──

def test_extract_text_from_html_strips_noise():
    web = import_module("research_buddy.tools.web")
    html = ("<html><head><title>T</title><script>var x=1;</script></head>"
            "<body><nav>导航</nav><article><h1>标题</h1>"
            "<p>第一段正文。</p><p>第二段正文。</p></article>"
            "<footer>页脚</footer></body></html>")
    text = web.extract_text_from_html(html)
    assert "第一段正文" in text
    assert "var x" not in text   # script 被去掉
    assert "导航" not in text     # nav 被去掉
    assert "页脚" not in text     # footer 被去掉


# ── web 证据合并 ──

def test_web_evidence_merges_into_ledger():
    state = {
        "search_results": [],
        "local_evidence": [
            {"url": "https://example.com/doc", "title": "官方文档",
             "content": "这是抓取的网页全文内容", "score": 1.0, "source_type": "web"},
        ],
        "evidence_assessments": [],
    }
    ledger = editorial_planner.build_evidence_ledger(state)
    assert len(ledger) == 1
    assert ledger[0]["source_type"] == "web"
    assert ledger[0]["url"] == "https://example.com/doc"
    assert ledger[0]["assessment_status"] == "web"


def test_web_evidence_merges_into_source_table():
    state = {
        "search_results": [],
        "local_evidence": [
            {"url": "https://example.com/doc", "title": "官方文档",
             "content": "全文", "source_type": "web"},
        ],
        "known_source_urls": [],
    }
    table = synthesizer.build_source_table(state)
    assert table[0]["source"] == "web"
    assert table[0]["url"] == "https://example.com/doc"


# ── 执行验证工具安全边界 ──

def test_exec_verification_allows_verification_commands():
    r = local.run_exec_verification("python -c \"print(1+1)\"")
    # 只要能跑通（ok=True 或输出含 2 即可；不同环境 python 可能受限）
    assert r["ok"] is True or "2" in r["output"]


def test_exec_verification_rejects_install():
    r = local.run_exec_verification("pip install requests")
    assert r["ok"] is False
    assert "install" in r["error"]


def test_exec_verification_rejects_redirection():
    r = local.run_exec_verification("python -c \"print(1)\" > /tmp/x")
    assert r["ok"] is False


def test_exec_verification_rejects_non_whitelist():
    r = local.run_exec_verification("rm -rf /")
    assert r["ok"] is False
