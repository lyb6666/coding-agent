"""命令行入口。

用法：
    python main.py "你的编程任务"     # 单次任务，执行完即退出
    python main.py                    # 进入交互式会话，可连续提问，exit/quit 退出
"""
from __future__ import annotations

import sys

from coding_agent import config
from coding_agent.agent import run


def on_event(msg: str) -> None:
    print(msg, flush=True)


def run_once(task: str) -> None:
    print("=" * 60)
    answer = run(task, on_event=on_event)
    print("=" * 60)
    print(f"\n【结果】\n{answer}\n")


def repl() -> None:
    """交互式会话：持续读取任务、执行、再读取，直到输入 exit/quit 或按 Ctrl-C。"""
    print("编程智能体已启动。直接输入编程任务即可；输入 exit / quit 退出。")
    while True:
        try:
            task = input("\nagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return
        if not task:
            continue
        if task.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            return
        run_once(task)


def main() -> None:
    if not config.API_KEY:
        print("未检测到 DEEPSEEK_API_KEY。")
        print("请复制 .env.example 为 .env 并填入密钥，或设置环境变量 DEEPSEEK_API_KEY。")
        return

    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
        return

    repl()


if __name__ == "__main__":
    main()
