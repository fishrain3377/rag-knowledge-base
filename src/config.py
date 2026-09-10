"""配置加载：读取 configs/config.yaml，并支持多来源覆盖关键项。

优先级（从高到低）：
1. 系统环境变量（本地由项目根目录 ``.env`` 注入）
2. Streamlit Secrets（云端在社区云控制台配置；本地为 ``.streamlit/secrets.toml``）
3. ``configs/config.yaml`` 默认值

支持的关键项：
- OPENAI_API_KEY：LLM 的 API Key
- OPENAI_BASE_URL：OpenAI 兼容接口地址
- LLM_MODEL：模型名
- HF_ENDPOINT：HuggingFace 端点（本地可配国内镜像，云端留空走官方）
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# 允许被环境变量 / Streamlit Secrets 覆盖的 LLM 配置项
_ENV_OVERRIDES = {
    "api_key": "OPENAI_API_KEY",
    "base_url": "OPENAI_BASE_URL",
    "model": "LLM_MODEL",
}


def _load_streamlit_secrets() -> dict[str, str]:
    """读取 Streamlit Secrets；非 Streamlit 运行时（如 API 服务/测试）安全返回空字典。"""
    try:
        import streamlit as st

        return {str(k): str(v) for k, v in st.secrets.items()}
    except Exception:  # noqa: BLE001 - 非 Streamlit 运行环境（API/测试）下安全降级
        return {}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载配置并依次叠加 Streamlit Secrets、环境变量，返回嵌套字典。"""
    load_dotenv(PROJECT_ROOT / ".env")
    st_secrets = _load_streamlit_secrets()

    # HuggingFace 端点：显式配置时才覆盖（本地 .env 用国内镜像；云端不配置即走官方）
    hf_endpoint = os.getenv("HF_ENDPOINT") or st_secrets.get("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    llm = config.setdefault("llm", {})
    for key, env_name in _ENV_OVERRIDES.items():
        # 环境变量优先，其次 Streamlit Secrets
        value = os.getenv(env_name) or st_secrets.get(env_name)
        if value:
            llm[key] = value

    return config
