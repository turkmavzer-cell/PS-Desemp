"""MeChi Studio yol, .env ve ayar yonetimi."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

MECHI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MECHI_DIR)
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
ASSETS_DIR = os.path.join(MECHI_DIR, "assets")
ASSETS_MASCOT_DIR = os.path.join(ASSETS_DIR, "mascot")
PROFILES_DIR = os.path.join(MECHI_DIR, "profiles")
DATA_DIR = os.path.join(MECHI_DIR, "data")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
ENV_PATH = os.path.join(MECHI_DIR, ".env")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
PROMPT_DB_PATH = os.path.join(DATA_DIR, "prompt_history.db")
LAUNCH_CTX_PATH = os.path.join(os.environ.get("TEMP", DATA_DIR), "mechi_launch_ctx.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "api_provider": "none",
    "api_model": "",
    "ps_preferred_version": "auto",
    "always_on_top": False,
    "mascot_enabled": True,
    "mascot_pause_between_sec": 2.0,
    "output_dir": OUTPUTS_DIR,
}

ENV_KEYS = (
    "GEMINI_API_KEY",
    "CF_ACCOUNT_ID",
    "CF_API_TOKEN",
    "HF_TOKEN",
    "FAL_KEY",
    "OPENAI_API_KEY",
)


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUTS_DIR, ASSETS_MASCOT_DIR, PROFILES_DIR):
        os.makedirs(d, exist_ok=True)


def load_env(env_path: str = ENV_PATH) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return values
    with open(env_path, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (len(v) >= 2) and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                v = v[1:-1]
            if k:
                values[k] = v
    return values


def get_env_key(key: str, default: str = "", env_path: str = ENV_PATH) -> str:
    env_map = load_env(env_path)
    val = env_map.get(key)
    if val:
        return val
    return os.environ.get(key, default)


def save_env_values(updates: Dict[str, str], env_path: str = ENV_PATH) -> None:
    existing = load_env(env_path)
    for k, v in updates.items():
        clean_k = str(k).strip()
        clean_v = str(v).strip().replace("\r", "").replace("\n", "")
        if clean_v:
            existing[clean_k] = clean_v
        else:
            existing.pop(clean_k, None)

    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    lines = ["# MeChi Studio yerel API anahtarlari (repoya asla yuklenmez)"]
    for k in sorted(existing.keys()):
        val = existing[k].replace('"', '\\"')
        lines.append(f'{k}="{val}"')
    with open(env_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def load_settings(settings_path: str = SETTINGS_PATH) -> Dict[str, Any]:
    ensure_dirs()
    cfg = dict(DEFAULT_SETTINGS)
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception:
            pass
    if not cfg.get("output_dir"):
        cfg["output_dir"] = OUTPUTS_DIR
    return cfg


def save_settings(cfg: Dict[str, Any], settings_path: str = SETTINGS_PATH) -> None:
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    merged = dict(DEFAULT_SETTINGS)
    merged.update(cfg)
    with open(settings_path, "w", encoding="utf-8") as fp:
        json.dump(merged, fp, indent=2, ensure_ascii=False)


def save_to_downloads(
    src_path: str,
    downloads_dir: Optional[str] = None,
    reveal_in_explorer: bool = True,
) -> Tuple[bool, str]:
    """Cikti dosyasini Windows Indirilenler klasorune guvenle kopyalar (zaten oradaysa tekrar kopyalamaz)."""
    if not src_path or not os.path.exists(src_path):
        return False, f"Dosya bulunamadı: {src_path}"

    dl_dir = downloads_dir or os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(dl_dir, exist_ok=True)

    abs_src = os.path.abspath(src_path)
    abs_dl = os.path.abspath(dl_dir)

    if os.path.normcase(os.path.dirname(abs_src)) == os.path.normcase(abs_dl):
        dst = abs_src
    else:
        base = os.path.basename(abs_src)
        stem, ext = os.path.splitext(base)
        dst = os.path.join(abs_dl, base)
        idx = 2
        while os.path.exists(dst):
            dst = os.path.join(abs_dl, f"{stem}_{idx}{ext}")
            idx += 1
        shutil.copy2(abs_src, dst)

    if reveal_in_explorer and os.name == "nt":
        try:
            subprocess.Popen(["explorer", "/select,", os.path.normpath(dst)])
        except OSError:
            pass

    return True, dst
