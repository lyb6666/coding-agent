"""agent 核心循环。

这是整个项目的“心脏”，负责：
1. 维护对话历史与上下文（含超限裁剪）；
2. 调用模型并解析其 tool_calls；
3. 驱动工具在本地执行，把结果回填给模型；
4. 决定循环何时终止；
5. 兜底处理模型调用失败等各类错误。
"""
from __future__ import annotations

import time

from openai import OpenAI

from . import config
from .tools import TOOL_SCHEMAS, execute_tool

client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

SYSTEM_PROMPT = """你是一个编程智能体（coding agent），目标是在工作目录下独立完成用户交给你的编程任务。
你可以使用 read_file / write_file / run_command / list_files / search_content 等工具来读写文件、执行命令、搜索代码。
工作方式：
1. 先了解现状（list_files / read_file），再动手；
2. 编写代码用 write_file 写入，用 run_command 运行验证；
3. 根据运行结果迭代修正，直到任务完成；
4. 完成后用一段简洁文字总结：你做了什么、结果如何。
如果没有需要执行的操作，直接输出最终回答即可。"""


def estimate_tokens(messages: list[dict]) -> int:
    """粗略估算上下文 token 数（字符数 / 4），用于决定是否裁剪历史。"""
    return sum(len(str(m.get("content", ""))) for m in messages) // 4


def _summarize_text(text: str) -> str:
    """调用模型把一段执行历史压缩成一句要点。"""
    resp = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": (
                "你是历史压缩助手。把下面这段编程智能体的执行记录压缩成一句简洁要点，"
                "说明做了什么操作、涉及哪些文件、结果如何。直接输出要点，不要解释。"
            )},
            {"role": "user", "content": text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def compress_history(messages: list[dict], emit) -> list[dict]:
    """上下文超限时，把最早的工具轮次「压缩成摘要」而非直接丢弃。

    始终保留最前的 system + 首条 user（任务目标）与最新内容；中间的历史轮次被
    压缩成一句话摘要插回，这样既省 token 又不丢失「已经做到哪一步」的信息。
    """
    while estimate_tokens(messages) > config.MAX_CONTEXT_TOKENS:
        idx = None
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                idx = i
                break
        if idx is None:
            break  # 没有可压缩的工具轮次

        end = idx + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1

        round_text = "\n".join(str(m.get("content", "")) for m in messages[idx:end])
        try:
            summary = _summarize_text(round_text)
        except Exception:
            summary = None  # 摘要失败就退回「丢弃」的旧行为
        if emit:
            emit("summary", message="上下文超限，已压缩最早的一轮历史")

        del messages[idx:end]
        if summary:
            messages.insert(idx, {"role": "user", "content": f"[历史摘要] {summary}"})
    return messages


def _call_model(messages: list[dict], emit):
    """调用模型，指数退避重试（最多 3 次），处理限流与网络抖动。"""
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            return client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except Exception as e:
            last_err = e
            if attempt < 3:
                emit("retry", message=f"模型调用失败：{e}，{delay:.0f} 秒后重试")
                time.sleep(delay)
                delay *= 2
    raise last_err


def run(task: str, on_event=None) -> str:
    """执行 agent 主循环，返回最终回答文本。

    on_event(event_type, **data) 是可选的事件回调，用于把 agent 的状态（轮次、
    工具调用、工具结果、错误）汇报给 UI；UI 只负责渲染，与核心逻辑解耦。
    """

    def emit(event_type: str, **data) -> None:
        if on_event:
            on_event(event_type, **data)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    last_call_key = None  # 上一次工具调用的 (name, args)，用于检测死循环
    repeat_count = 0

    for step in range(1, config.MAX_ITERATIONS + 1):
        messages = compress_history(messages, emit)
        emit("step", n=step)

        try:
            resp = _call_model(messages, emit)
        except Exception as e:
            return f"模型调用失败：{e}"

        msg = resp.choices[0].message
        finish_reason = resp.choices[0].finish_reason

        # 没有请求任何工具 -> 视为任务完成
        if not msg.tool_calls:
            content = msg.content or "(模型未返回文本)"
            if finish_reason == "length":
                # 输出达到长度上限被截断：如实标注，避免静默丢失
                content += "\n[注意：回答因长度限制被截断，可能不完整]"
            messages.append({"role": "assistant", "content": msg.content or ""})
            return content

        # 记入 assistant 消息（含 tool_calls）
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # 逐个在本地执行工具，并把结果作为 tool 消息回填
        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments

            # 死循环检测：连续多次调用同一工具且参数完全相同 → 判定无进展，主动停止
            key = (name, args)
            if key == last_call_key:
                repeat_count += 1
            else:
                last_call_key = key
                repeat_count = 1
            if repeat_count >= config.REPEAT_LIMIT:
                return (f"检测到连续 {repeat_count} 次重复调用 {name}（参数相同），"
                        f"疑似陷入死循环，已停止执行。")

            emit("tool_call", name=name, args=args)
            result = execute_tool(name, args)
            emit("tool_result", name=name, result=result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "达到最大迭代轮数，任务未在限制内完成。"
