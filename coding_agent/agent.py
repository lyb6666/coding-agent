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


def trim_history(messages: list[dict]) -> list[dict]:
    """上下文超限时，从中间丢弃最旧的一轮「工具调用 + 工具结果」。

    始终保留最前的 system + 首条 user（任务目标）与最新内容（最近进展），
    只牺牲中间的历史工具轮次——这是上下文管理策略的核心取舍。
    """
    while estimate_tokens(messages) > config.MAX_CONTEXT_TOKENS:
        idx = None
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                idx = i
                break
        if idx is None:
            break  # 没有可裁剪的工具轮次，放弃裁剪
        del messages[idx]  # 删除 assistant(tool_calls) 消息
        while idx < len(messages) and messages[idx].get("role") == "tool":
            del messages[idx]  # 连带删除其对应的 tool 结果，保证消息序列合法
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
        messages = trim_history(messages)
        emit("step", n=step)

        try:
            resp = _call_model(messages, emit)
        except Exception as e:
            return f"模型调用失败：{e}"

        msg = resp.choices[0].message

        # 没有请求任何工具 -> 视为任务完成
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "(模型未返回文本)"

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
