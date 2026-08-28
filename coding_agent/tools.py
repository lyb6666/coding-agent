"""工具定义与本地执行。

本模块只做两件事：
1. TOOL_SCHEMAS —— 用 JSON Schema 描述每个工具，通过模型原生的 tool calling 接口下发；
2. execute_tool —— 在「本地」真正执行这些工具（读写文件、跑命令），不依赖任何服务端
   代码执行或文件工具（题目明令禁止 Code Interpreter / Files API）。

安全边界（设计决策）：所有相对路径都以 WORKING_DIR 为根，危险命令会被直接拒绝，
避免 agent 越界读写或误删文件。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import WORKING_DIR

# ---- 危险命令黑名单：命中即拒绝执行，作为 agent 的“安全护栏” ----
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf .",
    "rm -rf *",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
    "shutdown",
    "reboot",
    "halt",
    "git push --force",
    "git reset --hard",
]


# 危险命令确认回调：由 CLI 注入。返回 True 表示允许执行，False 表示拒绝。
# 默认未注入时一律拒绝（安全优先），体现「agent 的自主性必须有安全边界」。
_confirm_dangerous = None


def set_confirm_callback(fn) -> None:
    """注入危险命令确认回调（由 CLI 调用）。"""
    global _confirm_dangerous
    _confirm_dangerous = fn


def _resolve(path: str) -> Path:
    """把相对路径解析到工作目录下，并规范化为绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = WORKING_DIR / p
    return p.resolve()


def read_file(path: str) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"错误：文件不存在 {p}"
    if p.is_dir():
        return f"错误：{p} 是目录，请改用 list_files"
    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > 8000:
        data = data[:8000] + "\n...(内容过长，已截断)..."
    return data


def write_file(path: str, content: str) -> str:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {p}（{len(content)} 字符）"


def list_files(path: str) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"错误：路径不存在 {p}"
    if not p.is_dir():
        return f"错误：{p} 不是目录"
    entries = []
    for e in sorted(p.iterdir()):
        kind = "D" if e.is_dir() else "F"
        entries.append(f"{kind} {e.name}")
    return "\n".join(entries) if entries else f"{p} 是空目录"


# 搜索时跳过的目录
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".idea", ".egg-info", "build", "dist"}


def search_content(path: str, pattern: str, max_results: int = 200) -> str:
    """在目录下递归搜索文件内容，返回「文件路径:行号: 该行内容」。"""
    p = _resolve(path)
    if not p.exists():
        return f"错误：路径不存在 {p}"

    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = []
        for f in p.rglob("*"):
            if f.is_file() and not any(part in _SKIP_DIRS for part in f.parts):
                files.append(f)
    else:
        return f"错误：{p} 既不是文件也不是目录"

    matches = []
    for f in files:
        if len(matches) >= max_results:
            break
        try:
            if f.stat().st_size > 1_000_000:  # 跳过 >1MB 的大文件
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern in line:
                matches.append(f"{f}:{lineno}: {line.strip()[:200]}")
                if len(matches) >= max_results:
                    break

    if not matches:
        return f"未找到包含 `{pattern}` 的内容"
    out = "\n".join(matches)
    if len(matches) >= max_results:
        out += f"\n...(结果已达上限 {max_results} 条，已截断)"
    return out


def run_command(command: str, timeout: int = 60) -> str:
    for pat in DANGEROUS_PATTERNS:
        if pat in command:
            allowed = _confirm_dangerous(command) if _confirm_dangerous else False
            if not allowed:
                return f"已拒绝执行：命令含危险模式 `{pat}`（未获用户确认）。"
            break  # 用户已确认，继续执行

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=WORKING_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",  # 输出含非 UTF-8 字节时不崩溃
        )
    except subprocess.TimeoutExpired:
        return f"错误：命令超时（>{timeout}s）"

    parts = []
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append(f"[stderr]\n{proc.stderr.rstrip()}")
    if not parts:
        parts.append("(无输出)")

    status = f"退出码 {proc.returncode}"
    if proc.returncode != 0:
        status += "（失败）"
    return f"{status}\n" + "\n".join(parts)


# ---- 工具注册表：名字 -> (执行函数, JSON Schema) ----
TOOLS = {
    "read_file": (read_file, {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                },
                "required": ["path"],
            },
        },
    }),
    "write_file": (write_file, {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把内容写入文件（覆盖式），父目录不存在会自动创建。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的完整内容"},
                },
                "required": ["path", "content"],
            },
        },
    }),
    "run_command": (run_command, {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "在工作目录下执行一条 shell 命令，返回标准输出、错误输出与退出码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的 shell 命令"},
                },
                "required": ["command"],
            },
        },
    }),
    "list_files": (list_files, {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录下的文件与子目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径"},
                },
                "required": ["path"],
            },
        },
    }),
    "search_content": (search_content, {
        "type": "function",
        "function": {
            "name": "search_content",
            "description": "在目录下递归搜索文件内容，返回「文件路径:行号: 该行内容」。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要搜索的目录或文件路径"},
                    "pattern": {"type": "string", "description": "要搜索的关键字"},
                },
                "required": ["path", "pattern"],
            },
        },
    }),
}

TOOL_SCHEMAS = [schema for (_fn, schema) in TOOLS.values()]


def execute_tool(name: str, arguments: str) -> str:
    """执行一次工具调用。arguments 是模型给出的 JSON 字符串。

    无论成功失败，都返回一段可读文本交还给模型，让模型据此决定下一步——这是
    agent 能够「根据执行结果自我修正」的关键。
    """
    if name not in TOOLS:
        return f"错误：未知工具 {name}"

    fn, _schema = TOOLS[name]
    try:
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return f"错误：工具参数不是合法 JSON：{e}\n原始参数：{arguments}"

    try:
        return str(fn(**kwargs))
    except TypeError as e:
        return f"错误：工具参数不匹配：{e}"
    except Exception as e:  # 兜底：任何执行异常都反馈给模型，而不是让 agent 崩溃
        return f"错误：执行 {name} 时发生异常：{type(e).__name__}: {e}"
