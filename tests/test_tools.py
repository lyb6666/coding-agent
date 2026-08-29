"""tools.py 的单元测试：验证工具函数的正确性与安全边界。"""
from coding_agent.tools import (
    read_file,
    write_file,
    list_files,
    search_content,
    run_command,
    _match_dangerous,
    _clip_output,
)


def test_write_then_read_file(tmp_path):
    p = tmp_path / "hello.txt"
    write_file(str(p), "hello\nworld")
    result = read_file(str(p))
    assert "hello" in result
    assert "world" in result


def test_write_file_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "c.txt"
    write_file(str(p), "hi")
    assert p.exists()


def test_read_file_offset_limit(tmp_path):
    p = tmp_path / "lines.txt"
    p.write_text("1\n2\n3\n4\n5\n", encoding="utf-8")
    result = read_file(str(p), offset=2, limit=2)
    assert "2:" in result  # 第 2 行
    assert "3:" in result  # 第 3 行
    assert "4:" not in result  # 第 4 行不应出现


def test_read_file_out_of_bounds(tmp_path):
    p = tmp_path / "small.txt"
    p.write_text("only line\n", encoding="utf-8")
    result = read_file(str(p), offset=999)
    assert "超出范围" in result


def test_read_file_missing():
    result = read_file("/no/such/file.txt")
    assert "不存在" in result


def test_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    result = list_files(str(tmp_path))
    assert "a.txt" in result
    assert "sub" in result


def test_search_content_finds_match(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")
    result = search_content(str(tmp_path), "foo")
    assert "a.py" in result
    assert "b.py" not in result  # foo 只在 a.py 里


def test_search_content_no_match(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    result = search_content(str(tmp_path), "zzzz_nothing")
    assert "未找到" in result


def test_run_command_dangerous_denied():
    result = run_command("rm -rf /")
    assert "已拒绝执行" in result


def test_run_command_basic():
    result = run_command("echo hello")
    assert "hello" in result
    assert "退出码 0" in result


def test_run_command_stdin_redirects():
    # 修复验证：input() 应立即得到 EOF 报错，而不是挂起等待输入
    result = run_command('python -c "input()"')
    assert "退出码" in result
    assert "EOFError" in result


def test_dangerous_detection_is_exact():
    # 分词后精确匹配：rm -rf / 命中，rm -rf /tmp 不误伤
    assert _match_dangerous("rm -rf /") is not None
    assert _match_dangerous("rm -rf /tmp") is None
    assert _match_dangerous("rm -rf .cache") is None
    assert _match_dangerous("echo hello") is None


def test_clip_output_truncates_long_text():
    long = "x" * 10000
    result = _clip_output(long)
    assert len(result) < 10000  # 被截断了
    assert "省略" in result


def test_clip_output_keeps_short_text():
    short = "hello"
    assert _clip_output(short) == short
