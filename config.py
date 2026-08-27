"""配置加载。

密钥只从「环境变量」或「项目根目录下未入库的 .env 文件」读取，绝不硬编码进代码。
这是题目明确要求的合规做法，也是本项目的第一个设计决策：凭据与代码分离。
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_env(path: Path | None = None) -> None:
    """极简 .env 解析器（不依赖 python-dotenv，逻辑自己写）。

    只处理 `KEY=VALUE` 行，忽略空行与 `#` 注释；不覆盖系统里已存在的环境变量，
    因此优先级为：系统环境变量 > .env 文件。
    """
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env()

# 模型与连接配置
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# agent 行为参数（都可通过环境变量覆盖）
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "20"))
MAX_CONTEXT_TOKENS = int(os.environ.get("AGENT_MAX_CONTEXT_TOKENS", "32000"))
WORKING_DIR = Path(os.environ.get("AGENT_WORKING_DIR", str(PROJECT_ROOT)))
