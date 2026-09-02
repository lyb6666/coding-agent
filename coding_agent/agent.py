"""agent 核心循环。

这是整个项目的“心脏”，负责：
1. 维护对话历史与上下文（含超限裁剪）；
2. 调用模型并解析其 tool_calls；
3. 驱动工具在本地执行，把结果回填给模型；
4. 决定循环何时终止；
5. 兜底处理模型调用失败等各类错误。
"""
from __future__ import annotations

import platform
import time

from openai import OpenAI

from . import config
from .storage import save_history
from .tools import TOOL_SCHEMAS, execute_tool

client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

def _shell_hint() -> str:
    """根据操作系统给出 shell 命令提示，避免模型生成错误的命令语法。"""
    if platform.system() == "Windows":
        return (
            "你运行在 Windows 上，命令请用 cmd.exe 语法：列目录用 `dir`（不是 `ls`），"
            "路径用反斜杠 `C:\\...`（不是 `/c/...`），不要用 `sleep`、`curl`、`(cmd &)` 这类 Unix/bash 写法。"
        )
    return "你运行在 Unix/Linux 上，命令请用 bash 语法。"


def _workspace_info() -> str:
    """工作目录与路径约定，注入提示词，让模型用相对路径。"""
    return (
        f"工作目录：{config.WORKING_DIR}。读写文件请用「相对路径」（会自动解析到工作目录下），"
        f"不要随意用绝对路径。"
    )


SYSTEM_PROMPT = f"""你是一个编程智能体（coding agent），目标是在工作目录下独立完成用户交给你的编程任务。
你可以使用 read_file / write_file / run_command / list_files / search_content 等工具来读写文件、执行命令、搜索代码。

{_workspace_info()}
{_shell_hint()}
注意：不要启动会一直运行的进程（HTTP 服务器、守护进程等），它们会卡住执行环境；静态网页这类任务直接写好文件让用户打开即可。

工作方式：
1. 先了解现状（list_files / read_file），再动手；
2. 编写代码用 write_file 写入，用 run_command 运行验证；
3. 根据运行结果迭代修正，直到任务完成；
4. 完成后用一段简洁文字总结：你做了什么、结果如何。
如果没有需要执行的操作，直接输出最终回答即可。"""

PLANNER_PROMPT = f"""你是任务规划器。把用户的任务分解成「这个编程智能体将要执行」的步骤清单。
这些步骤是给一个会用工具（write_file / run_command / read_file / list_files / search_content）的编程智能体执行的，不是给人手动操作的。

{_workspace_info()}
{_shell_hint()}
注意：不要规划「启动服务器 / 常驻进程」这类步骤（会卡住执行环境）。

要求：
- 每步一行，用「1. 2. 3.」编号；
- 每步写清「用哪个工具做什么」，例如「用 write_file 创建 xxx.py」「用 run_command 运行 python xxx.py」；
- 只输出计划本身，不要解释，不要执行，不要用工具。
"""


def _estimate_text_tokens(text: str) -> int:
    """估算单段文本的 token 数。

    中文等全角字符按 1 字符≈1 token，其余（英文/数字）按 4 字符≈1 token；
    向上取整略高估，宁可早一点压缩，也不让上下文溢出。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_tokens(messages: list[dict]) -> int:
    """估算上下文 token 数，用于决定是否压缩历史。

    除了 content，还把 assistant 消息里 tool_calls 的 name/arguments 计入——
    因为 write_file 的 arguments 就包含要写入的完整文件内容，漏掉会导致低估。
    """
    total = 0
    for m in messages:
        total += _estimate_text_tokens(str(m.get("content", "")))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            total += _estimate_text_tokens(str(fn.get("name", "")))
            total += _estimate_text_tokens(str(fn.get("arguments", "")))
    return total


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
    """上下文超限时，把最早的多轮工具历史「一次性压缩成摘要」，而非一轮一轮压。

    始终保留最前的 system + 首条 user（任务目标）与最近一轮（避免打断正在进行的
    操作），中间的所有旧轮次打包成一条摘要插回。
    """
    KEEP_ROUNDS = 1  # 保留最近几轮不压缩，确保执行连续性

    while estimate_tokens(messages) > config.MAX_CONTEXT_TOKENS:
        round_starts = [
            i for i, m in enumerate(messages)
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        if len(round_starts) <= KEEP_ROUNDS:
            break  # 没有足够多的旧轮次可压缩

        first = round_starts[0]
        last_keep = round_starts[-KEEP_ROUNDS]
        old_text = "\n".join(str(m.get("content", "")) for m in messages[first:last_keep])

        try:
            summary = _summarize_text(old_text)
        except Exception:
            summary = None  # 摘要失败就退回「丢弃」的旧行为
        if emit:
            emit("summary", message=f"上下文超限，已压缩 {len(round_starts) - KEEP_ROUNDS} 轮历史")

        del messages[first:last_keep]
        if summary:
            messages.insert(first, {"role": "user", "content": f"[历史摘要] {summary}"})
    return messages


def _is_permanent_error(e: Exception) -> bool:
    """判断是否为「永久性错误」（重试无意义），如无效密钥 401、请求格式错误 4xx。"""
    status = getattr(e, "status_code", None)
    return status is not None and 400 <= status < 500 and status != 429


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
            if _is_permanent_error(e):
                raise  # 永久错误重试无意义，立即抛出
            if attempt < 3:
                emit("retry", message=f"模型调用失败：{e}，{delay:.0f} 秒后重试")
                time.sleep(delay)
                delay *= 2
    raise last_err


def _generate_plan(task: str) -> str:
    """规划阶段：把任务分解成带步骤的计划，返回计划文本。"""
    resp = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": task},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _is_failure(result: str) -> bool:
    """判断工具执行结果是否表示失败（只看状态/错误前缀，避免误判输出内容）。"""
    first = result.splitlines()[0] if result else ""
    return (
        first.startswith("错误")
        or "（失败）" in first
        or "已拒绝" in first
        or "超时" in first
    )


def run(task: str, on_event=None, history: list[dict] | None = None) -> str:
    """执行 agent 主循环，返回最终回答文本。

    on_event(event_type, **data) 是可选的事件回调，用于把 agent 的状态（轮次、
    工具调用、工具结果、错误）汇报给 UI；UI 只负责渲染，与核心逻辑解耦。
    history 为上次会话的消息列表（用于恢复对话），缺省则新开会话。
    """

    def emit(event_type: str, **data) -> None:
        if on_event:
            on_event(event_type, **data)

    # 阶段一：规划（恢复会话时跳过——上下文已建立，直接延续对话，不再生成计划）
    plan = ""
    if history is None:
        emit("planning")
        try:
            plan = _generate_plan(task)
        except Exception as e:
            plan = ""
            emit("error", message=f"规划失败，跳过规划直接执行：{e}")
        if plan:
            emit("plan", plan=plan)

    # 阶段二：按计划执行（恢复会话则延续旧历史）
    if history:
        messages = list(history)
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": SYSTEM_PROMPT}  # 用最新 system prompt
        messages.append({"role": "user", "content": task})
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
    if plan:
        messages.append({"role": "user", "content": f"[执行计划]\n{plan}\n\n请按上面的计划逐步执行。"})

    try:
        return _run_loop(messages, emit)
    finally:
        save_history(config.WORKING_DIR, messages)  # 无论成功失败都保存历史，便于下次恢复


def _run_loop(messages: list[dict], emit) -> str:
    """执行 agent 主循环（调用模型、执行工具、循环直到终止），返回最终回答。"""
    last_call_key = None  # 上一次工具调用的 (name, args)，用于检测死循环
    repeat_count = 0
    consecutive_failures = 0  # 连续失败次数，用于检测"无法完成任务"

    for step in range(1, config.MAX_ITERATIONS + 1):
        messages = compress_history(messages, emit)  # ① 上下文超限时压缩成摘要
        emit("step", n=step)

        try:
            resp = _call_model(messages, emit)  # ② 调用大模型
        except Exception as e:
            return f"模型调用失败：{e}"

        msg = resp.choices[0].message  # ③ 解析模型的回复
        finish_reason = resp.choices[0].finish_reason

        # 输出达到长度上限被截断：无论有无工具调用，都主动提示（避免静默丢尾）
        if finish_reason == "length":
            emit("retry", message="模型输出达到长度上限，已被截断")

        # 没有请求任何工具 -> 视为任务完成
        if not msg.tool_calls:  # ④ 没有工具调用 → 任务完成，返回回答
            content = msg.content or "(模型未返回文本)"
            if finish_reason == "length":
                # 输出达到长度上限被截断：如实标注，避免静默丢失
                content += "\n[注意：回答因长度限制被截断，可能不完整]"
            messages.append({"role": "assistant", "content": msg.content or ""})
            return content

        # ⑤ 有工具调用：记入 assistant 消息（含 tool_calls）
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
            result = execute_tool(name, args)  # ⑥ 在本地执行工具
            emit("tool_result", name=name, result=result)

            # 连续失败检测：连续多次工具执行失败（不一定是同一命令）→ 判定无法完成任务
            if _is_failure(result):
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if consecutive_failures >= config.FAILURE_LIMIT:
                return (f"连续 {consecutive_failures} 次工具执行失败，"
                        f"疑似无法完成任务，已停止执行。")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})  # ⑦ 结果回填，进入下一轮循环

    return "达到最大迭代轮数，任务未在限制内完成。"
