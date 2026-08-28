"""命令行界面（CLI）：彩色终端 + 交互式 REPL。

设计要点：
- 通过结构化事件回调接收 agent 的状态，UI 只负责渲染，与核心逻辑解耦；
- Windows 下自写 ANSI 转义序列启用（不依赖 colorama 等第三方库）；
- 提供 help / tools / clear 等命令，让功能清晰可发现。
"""
from __future__ import annotations

import ctypes
import os
import sys

from coding_agent import config
from coding_agent.agent import run
from coding_agent.tools import TOOLS, set_confirm_callback


# ---- ANSI 颜色（自写，无第三方依赖）----
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def enable_ansi() -> None:
    """在 Windows 上启用 ANSI 转义序列（VT 模式），让颜色生效。"""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            # 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass  # 不支持就退化为纯文本输出


def _setup_utf8() -> None:
    """输出被重定向（非交互控制台）时改为 UTF-8，避免 emoji/特殊字符因 GBK 崩溃。

    交互控制台由 Python 的 Unicode 控制台 API 处理，无需改动；这里只处理输出被
    重定向到文件/管道的情况。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


BANNER = f"""\
{C.MAGENTA}{C.BOLD}  🐱 coding agent {C.RESET}{C.DIM}· 你的编程小助手{C.RESET}
{C.DIM}  ───────────────────────────────────────────────{C.RESET}
  直接输入编程任务，回车执行
  {C.CYAN}help{C.RESET} 看命令 · {C.CYAN}tools{C.RESET} 看工具 · {C.CYAN}exit{C.RESET} 退出
{C.DIM}  ───────────────────────────────────────────────{C.RESET}
"""

HELP_TEXT = f"""\
{C.BOLD}可用命令{C.RESET}
  {C.CYAN}help{C.RESET}    查看帮助
  {C.CYAN}tools{C.RESET}   列出 agent 可用的工具
  {C.CYAN}clear{C.RESET}   清屏
  {C.CYAN}exit{C.RESET}    退出（也可 quit / q / 退出）
  {C.DIM}其它输入都会当作编程任务交给 agent 执行{C.RESET}
"""


def _render_event(event_type: str, **data) -> None:
    """把 agent 的结构化事件渲染成彩色终端输出。"""
    if event_type == "step":
        print(f"  {C.CYAN}{C.BOLD}▸ 第 {data['n']} 轮{C.RESET} 思考中…")
    elif event_type == "retry":
        print(f"  {C.YELLOW}⚠ {data['message']}{C.RESET}")
    elif event_type == "tool_call":
        args = str(data.get("args", ""))
        if len(args) > 100:
            args = args[:100] + "…"
        print(f"    {C.BLUE}🔧 {data['name']}{C.RESET}({C.DIM}{args}{C.RESET})")
    elif event_type == "tool_result":
        result = str(data.get("result", ""))
        first = result.splitlines()[0] if result else ""
        if len(first) > 120:
            first = first[:120] + "…"
        print(f"       {C.GREEN}└─ {first}{C.RESET}")
    elif event_type == "error":
        print(f"  {C.RED}✗ {data['message']}{C.RESET}")
    elif event_type == "summary":
        print(f"  {C.YELLOW}📝 {data['message']}{C.RESET}")


def _confirm_dangerous(command: str) -> bool:
    """危险命令确认：打印警告并询问用户是否执行（默认拒绝）。"""
    print(f"{C.YELLOW}⚠ 检测到危险命令：{command}{C.RESET}")
    try:
        ans = input(f"{C.YELLOW}  是否执行？[y/N] {C.RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False  # 无法交互时安全优先，拒绝执行
    return ans in ("y", "yes")


def _print_result(answer: str) -> None:
    print()
    print(f"{C.GREEN}{C.BOLD}┌─ 结果 ──────────────────{C.RESET}")
    for line in answer.splitlines():
        print(f"{C.GREEN}│{C.RESET} {line}")
    print(f"{C.GREEN}└──────────────────────────{C.RESET}")
    print()


def _print_tools() -> None:
    print(f"{C.BOLD}可用工具{C.RESET}")
    for name, (_fn, schema) in TOOLS.items():
        desc = schema["function"]["description"]
        print(f"  {C.CYAN}{name}{C.RESET}  {C.DIM}{desc}{C.RESET}")


def run_once(task: str) -> None:
    print(f"{C.DIM}任务：{task}{C.RESET}")
    print(f"{C.DIM}{'─' * 48}{C.RESET}")
    try:
        answer = run(task, on_event=_render_event)
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}⏹ 已中断当前任务，回到提示符。{C.RESET}\n")
        return
    _print_result(answer)


def repl() -> None:
    print(BANNER)
    while True:
        try:
            task = input(f"{C.MAGENTA}{C.BOLD}🐱 coding>{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}再见啦～{C.RESET}")
            return
        if not task:
            continue
        low = task.lower()
        if low in ("exit", "quit", "q", "退出"):
            print(f"{C.DIM}再见啦～{C.RESET}")
            return
        if low in ("help", "?", "帮助"):
            print(HELP_TEXT)
            continue
        if low in ("clear", "cls", "清屏"):
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if low in ("tools", "工具"):
            _print_tools()
            continue
        run_once(task)


def main() -> None:
    enable_ansi()
    _setup_utf8()
    set_confirm_callback(_confirm_dangerous)
    if not config.API_KEY:
        print(f"{C.RED}未检测到 DEEPSEEK_API_KEY。{C.RESET}")
        print("请复制 .env.example 为 .env 并填入密钥，或设置环境变量 DEEPSEEK_API_KEY。")
        return

    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
        return

    repl()


if __name__ == "__main__":
    main()
