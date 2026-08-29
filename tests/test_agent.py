"""agent.py 的单元测试：验证上下文管理与历史压缩逻辑（不调用真实模型）。"""
import coding_agent.agent as agent_mod
from coding_agent import config
from coding_agent.agent import estimate_tokens, compress_history


def test_estimate_tokens():
    messages = [{"content": "hello world"}, {"content": "12345678"}]
    # "hello world"=11 字符, "12345678"=8 字符, 共 19, //4 = 4
    assert estimate_tokens(messages) == 4


def _make_round_messages():
    """构造一段含两轮工具调用的对话历史。"""
    return [
        {"role": "system", "content": "你是编程智能体"},
        {"role": "user", "content": "写一个脚本并运行"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "已写入 hello.py"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "2", "type": "function", "function": {"name": "run_command", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "2", "content": "退出码 0"},
    ]


def test_compress_history_replaces_with_summary(monkeypatch):
    # 伪造 _summarize_text，避免真实调用模型
    monkeypatch.setattr(agent_mod, "_summarize_text", lambda text: "已写入并运行成功")
    monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 5)  # 调小阈值触发压缩

    messages = _make_round_messages()
    compressed = compress_history(messages, None)

    summaries = [m for m in compressed if "[历史摘要]" in str(m.get("content", ""))]
    assert len(summaries) >= 1  # 出现了摘要
    assert len(compressed) < 6  # 消息数减少


def test_compress_history_noop_when_small(monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 100000)  # 阈值很大，不触发

    messages = _make_round_messages()
    compressed = compress_history(messages, None)

    assert len(compressed) == 6  # 消息数不变
    summaries = [m for m in compressed if "[历史摘要]" in str(m.get("content", ""))]
    assert len(summaries) == 0  # 没有摘要
