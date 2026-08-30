"""会话历史持久化：每个工作目录下存一个 .coding_history.json。

不同文件夹各自有一份独立历史，互不干扰；历史文件就放在工作目录内，一目了然。
"""
from __future__ import annotations

import json
from pathlib import Path

HISTORY_FILENAME = ".coding_history.json"


def _path(working_dir: Path) -> Path:
    return working_dir / HISTORY_FILENAME


def save_history(working_dir: Path, messages: list[dict]) -> None:
    """把会话历史保存到工作目录下的 .coding_history.json。"""
    _path(working_dir).write_text(
        json.dumps({"messages": messages}, ensure_ascii=False),
        encoding="utf-8",
    )


def load_history(working_dir: Path) -> list[dict] | None:
    """加载工作目录下的会话历史；没有则返回 None。"""
    p = _path(working_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("messages")
    except Exception:
        return None


def clear_history(working_dir: Path) -> None:
    """删除工作目录下的会话历史。"""
    p = _path(working_dir)
    if p.exists():
        p.unlink()
