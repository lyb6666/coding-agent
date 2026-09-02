"""agent.py 的单元测试：验证上下文管理与历史压缩逻辑（不调用真实模型）。"""
import coding_agent.agent as agent_mod
from coding_agent import config
from coding_agent.agent import estimate_tokens, compress_history, _is_failure, _classify_error


def test_estimate_tokens():
    messages = [{"content": "hello world"}, {"content": "12345678"}]
    # "hello world"=11 字符 → 3, "12345678"=8 字符 → 2, 共 5
    assert estimate_tokens(messages) == 5


def test_estimate_tokens_counts_cjk():
    # 中文按 1 字符≈1 token，4 个中文字 = 4（而非旧的 1）
    assert estimate_tokens([{"content": "你好世界"}]) == 4


def test_estimate_tokens_counts_tool_calls():
    # tool_calls 的 name + arguments 也应计入（write_file 参数里是文件全文）
    messages = [{
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "write_file", "arguments": "abcdefghijkl"}}],
    }]
    # name "write_file"=10 字符 → 3；arguments "abcdefghijkl"=12 字符 → 3；共 6
    assert estimate_tokens(messages) == 6


def _make_rounds(n=2):
    """构造 n 轮工具调用的对话历史。"""
    messages = [
        {"role": "system", "content": "你是编程智能体"},
        {"role": "user", "content": "写一个脚本并运行"},
    ]
    for i in range(n):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": str(i), "type": "function", "function": {"name": "write_file", "arguments": "{}"}},
        ]})
        messages.append({"role": "tool", "tool_call_id": str(i), "content": f"已写入 file{i}.py"})
    return messages


def test_compress_history_replaces_with_summary(monkeypatch):
    # 伪造 _summarize_text，避免真实调用模型
    monkeypatch.setattr(agent_mod, "_summarize_text", lambda text: "已写入并运行成功")
    monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 5)  # 调小阈值触发压缩

    messages = _make_rounds(2)
    compressed = compress_history(messages, None)

    summaries = [m for m in compressed if "[历史摘要]" in str(m.get("content", ""))]
    assert len(summaries) >= 1  # 出现了摘要
    assert len(compressed) < len(_make_rounds(2))  # 消息数减少


def test_compress_history_noop_when_small(monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 100000)  # 阈值很大，不触发

    messages = _make_rounds(2)
    compressed = compress_history(messages, None)

    assert len(compressed) == len(messages)  # 消息数不变
    summaries = [m for m in compressed if "[历史摘要]" in str(m.get("content", ""))]
    assert len(summaries) == 0  # 没有摘要


def test_compress_history_batches_multiple_rounds(monkeypatch):
    # 严重超限时，多轮历史应「一次性」压缩，而不是一轮一轮调摘要模型
    calls = []

    def fake_summarize(text):
        calls.append(text)
        return "已压缩"

    monkeypatch.setattr(agent_mod, "_summarize_text", fake_summarize)
    monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 5)

    messages = _make_rounds(5)  # 5 轮历史
    compress_history(messages, None)

    assert len(calls) == 1  # 只调用一次摘要模型，而非多次


def test_is_failure_detects_errors():
    assert _is_failure("错误：文件不存在") is True
    assert _is_failure("退出码 1（失败）\n...") is True
    assert _is_failure("已拒绝执行：危险命令") is True
    assert _is_failure("已写入 hello.py（33 字符）") is False
    assert _is_failure("退出码 0\nhello") is False
    assert _is_failure("未找到包含 xxx 的内容") is False


def test_classify_error():
    class FakeError(Exception):
        def __init__(self, status=None):
            self.status_code = status

    assert _classify_error(FakeError(401)) == "permanent"  # 无效密钥
    assert _classify_error(FakeError(400)) == "permanent"  # 请求格式错误
    assert _classify_error(FakeError(429)) == "rate_limit"  # 限流
    assert _classify_error(FakeError(500)) == "server"  # 服务端错误
    assert _classify_error(Exception("network")) == "unknown"  # 无 status_code
