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
import shlex
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

# 常驻服务命令：这些命令会启动一直运行的进程，会卡住执行环境
LONG_RUNNING_PATTERNS = [
    "http.server",
    "flask run",
    "uvicorn",
    "gunicorn",
    "runserver",
    "nodemon",
    "npm start",
    "npm run dev",
    "yarn dev",
    "ng serve",
]


def _split_command(command: str) -> list[str]:
    """把命令拆成词元（用于危险检测）。shlex 失败时退化为按空白切分。"""
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _match_dangerous(command: str) -> str | None:
    """检测命令是否命中危险模式，命中则返回该模式，否则返回 None。

    用「分词后精确匹配」而非朴素子串匹配：`rm -rf /` 不再误伤 `rm -rf /tmp`。
    """
    tokens = _split_command(command)
    for pat in DANGEROUS_PATTERNS:
        pat_tokens = _split_command(pat)
        n = len(pat_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == pat_tokens:
                return pat
    return None


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


def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
    """读取文件内容。offset 是起始行号（从 1 开始），limit 是最大行数，用于分段读大文件。"""
    p = _resolve(path)
    if not p.exists():
        return f"错误：文件不存在 {p}"
    if p.is_dir():
        return f"错误：{p} 是目录，请改用 list_files"
    try:
        raw = p.read_bytes()
    except Exception as e:
        return f"错误：读取 {p} 失败：{type(e).__name__}: {e}"
    if 0 in raw:
        return f"错误：{p} 是二进制文件，无法按文本读取"
    lines = raw.decode("utf-8", errors="replace").splitlines()

    total = len(lines)
    offset = max(1, offset)
    start = offset - 1
    if start >= total:
        return f"{p} 只有 {total} 行，offset={offset} 已超出范围"

    end = min(total, start + limit)
    numbered = [f"{i}: {lines[i - 1]}" for i in range(start + 1, end + 1)]
    header = f"{p}（共 {total} 行，显示第 {start + 1}–{end} 行）"
    body = "\n".join(numbered)
    if end < total:
        body += f"\n...(剩余 {total - end} 行未显示，可设 offset={end + 1} 继续读)"
    return header + "\n" + body


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
    if not pattern:
        return f"错误：搜索关键字不能为空"
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
            raw = f.read_bytes()
            if 0 in raw:  # 跳过二进制文件
                continue
            text = raw.decode("utf-8", errors="replace")
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


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """把文件里第一处 old_string 替换成 new_string（局部编辑，避免整文件重写）。

    old_string 必须在文件中唯一出现，否则报错让模型提供更长的上下文。
    """
    if not old_string:
        return f"错误：old_string 不能为空"
    p = _resolve(path)
    if not p.exists():
        return f"错误：文件不存在 {p}"
    if p.is_dir():
        return f"错误：{p} 是目录"
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"错误：读取 {p} 失败：{type(e).__name__}: {e}"

    count = content.count(old_string)
    if count == 0:
        return f"错误：在 {p} 中未找到要替换的内容"
    if count > 1:
        return f"错误：要替换的内容在文件中出现了 {count} 次，请提供更长的上下文使其唯一"

    try:
        p.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    except Exception as e:
        return f"错误：写入 {p} 失败：{type(e).__name__}: {e}"
    return f"已替换 {p} 中的一处内容"


def delete_file(path: str) -> str:
    """删除指定文件。"""
    p = _resolve(path)
    if not p.exists():
        return f"错误：文件不存在 {p}"
    if p.is_dir():
        return f"错误：{p} 是目录，删除目录请用 run_command"
    try:
        p.unlink()
    except Exception as e:
        return f"错误：删除 {p} 失败：{type(e).__name__}: {e}"
    return f"已删除 {p}"


def move_file(src: str, dst: str) -> str:
    """移动或重命名文件。"""
    s = _resolve(src)
    d = _resolve(dst)
    if not s.exists():
        return f"错误：源文件不存在 {s}"
    if d.exists():
        return f"错误：目标已存在 {d}"
    try:
        d.parent.mkdir(parents=True, exist_ok=True)
        s.rename(d)
    except Exception as e:
        return f"错误：移动 {s} 失败：{type(e).__name__}: {e}"
    return f"已移动 {s} → {d}"


MAX_COMMAND_OUTPUT = 4000  # 单条命令输出最多返回的字符数，避免撑爆上下文


def _clip_output(text: str) -> tuple[str, bool]:
    """截断过长的命令输出，返回 (截断后的文本, 是否被截断)。"""
    text = text.rstrip()
    if len(text) <= MAX_COMMAND_OUTPUT:
        return text, False
    half = MAX_COMMAND_OUTPUT // 2
    clipped = f"{text[:half]}\n...(中间省略 {len(text) - MAX_COMMAND_OUTPUT} 字符)...\n{text[-half:]}"
    return clipped, True


def run_command(command: str, timeout: int = 60) -> str:
    if not command.strip():
        return f"错误：命令不能为空"
    pat = _match_dangerous(command)
    if pat:
        allowed = _confirm_dangerous(command) if _confirm_dangerous else False
        if not allowed:
            return f"已拒绝执行：命令含危险模式 `{pat}`（未获用户确认）。"

    # 常驻服务命令会一直运行并卡住执行环境，直接拦截
    for server_pat in LONG_RUNNING_PATTERNS:
        if server_pat in command:
            return (f"已拦截：命令会启动常驻服务（`{server_pat}`），它会一直运行并卡住执行环境。"
                    f"请不要启动服务器；静态网页直接写好文件让用户打开即可。")

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=WORKING_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",  # 输出含非 UTF-8 字节时不崩溃
            stdin=subprocess.DEVNULL,  # 命令读 stdin 时立即得到 EOF，避免交互式程序挂起
        )
    except subprocess.TimeoutExpired:
        return f"错误：命令超时（>{timeout}s）"

    parts = []
    truncated = False
    if proc.stdout:
        clipped, was_trunc = _clip_output(proc.stdout)
        truncated = truncated or was_trunc
        parts.append(clipped)
    if proc.stderr:
        clipped, was_trunc = _clip_output(proc.stderr)
        truncated = truncated or was_trunc
        parts.append(f"[stderr]\n{clipped}")
    if not parts:
        parts.append("(无输出)")

    status = f"退出码 {proc.returncode}"
    if proc.returncode != 0:
        status += "（失败）"
    if truncated:
        status += "（输出过长已截断）"
    return f"{status}\n" + "\n".join(parts)


# ---- 工具注册表：名字 -> (执行函数, JSON Schema) ----
TOOLS = {
    "read_file": (read_file, {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容，可指定起始行号 offset 和最大行数 limit，用于分段读取大文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                    "offset": {"type": "integer", "description": "起始行号，从 1 开始，默认 1"},
                    "limit": {"type": "integer", "description": "最多读取的行数，默认 400"},
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
    "edit_file": (edit_file, {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "把文件里第一处 old_string 替换成 new_string，用于局部编辑；old_string 需唯一。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "old_string": {"type": "string", "description": "要替换的原文"},
                    "new_string": {"type": "string", "description": "替换后的新内容"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    }),
    "delete_file": (delete_file, {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "删除指定文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要删除的文件路径"},
                },
                "required": ["path"],
            },
        },
    }),
    "move_file": (move_file, {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "移动或重命名文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string", "description": "源文件路径"},
                    "dst": {"type": "string", "description": "目标文件路径"},
                },
                "required": ["src", "dst"],
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
