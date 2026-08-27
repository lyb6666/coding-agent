"""命令行入口。

用法：
    python main.py "你的编程任务"
不带参数则进入交互式输入（输入空行结束）。
"""
from __future__ import annotations

import sys

import config
from agent import run


def on_event(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    if not config.API_KEY:
        print("未检测到 DEEPSEEK_API_KEY。")
        print("请复制 .env.example 为 .env 并填入密钥，或设置环境变量 DEEPSEEK_API_KEY。")
        return

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("请输入编程任务（输入空行结束）：")
        lines: list[str] = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        task = "\n".join(lines)
        if not task.strip():
            print("未输入任务，退出。")
            return

    print(f"\n任务：{task}\n" + "=" * 60)
    answer = run(task, on_event=on_event)
    print("=" * 60)
    print(f"\n【最终结果】\n{answer}")


if __name__ == "__main__":
    main()
