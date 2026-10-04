#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 DiceFrame 的「图像生成」能力指向本地 comfy-bridge。

DiceFrame 设了访问密码时，写接口需要 `Authorization: Bearer <访问密码>`：

    python wire_diceframe.py --password 你的访问密码

不传 --password 时只做「干跑」：读配置、打印将要提交的内容，不写任何东西。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

DF_DEFAULT = "http://127.0.0.1:18000"
# 桥接的 base_url。端口必须和桥接实际监听的一致（start-bridge.bat 里的 CB_PORT）。
BRIDGE_DEFAULT = "http://127.0.0.1:8192/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(HERE, "_diceframe_backup")

PROVIDER_ID = "comfyui-local"
# 实测：Anima（anima-base-v1.0）的二次元画质最好，且 img2img 保人物一致性
# 明显强于 SD1.5 的 DreamShaper，所以默认指向它。要换回 SD1.5 就 --model DreamShaper_8_pruned.safetensors
MODEL_ID = "anima-base-v1.0.safetensors"
# Anima 在 1792x1024 上约 80s，DiceFrame 的 imagegen_timeout_seconds 必须放到 300
IMAGEGEN_TIMEOUT = 300.0


STALE_BRIDGE_URLS = ("http://127.0.0.1:8190/v1", "http://127.0.0.1:8190")


def build_body(cfg: dict, bridge_url: str, model: str) -> dict:
    providers = []
    for entry in cfg.get("ai_providers") or []:
        if not isinstance(entry, dict):
            continue
        keep = {k: v for k, v in entry.items()
                if k in ("id", "name", "base_url", "api_format", "models", "model_capabilities")}
        # 手工加过的桥接条目可能还指着旧端口（8190 被僵尸进程占了），顺手改过来，
        # 免得界面上挑到那一条就绕过了新桥接。
        if str(keep.get("base_url") or "").rstrip("/") in STALE_BRIDGE_URLS:
            keep["base_url"] = bridge_url
        providers.append(keep)                      # 不把 masked 的 api_key 回传
    providers = [p for p in providers if p.get("id") != PROVIDER_ID]
    providers.append({
        "id": PROVIDER_ID,
        "name": "ComfyUI (本地桥接)",
        "base_url": bridge_url,
        "api_format": "openai",
        "models": [model],
        "model_capabilities": {model: "image"},
    })
    return {
        "ai_providers": providers,
        "imagegen_provider_ref": PROVIDER_ID,
        "imagegen_model": model,
        "imagegen_provider": "openai-compatible",
        "imagegen_enabled": True,
        "imagegen_timeout_seconds": IMAGEGEN_TIMEOUT,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="让 DiceFrame 的图像生成走本地 comfy-bridge")
    ap.add_argument("--diceframe", default=os.environ.get("DF_URL", DF_DEFAULT),
                    help="DiceFrame 地址")
    ap.add_argument("--bridge", default=os.environ.get("CB_URL", BRIDGE_DEFAULT),
                    help="桥接的 OpenAI 兼容 base_url")
    ap.add_argument("--model", default=MODEL_ID, help="imagegen_model / 底模文件名")
    ap.add_argument("--password", default=os.environ.get("DF_PASSWORD", ""),
                    help="DiceFrame 访问密码（写接口需要；不传则只干跑预览）")
    ap.add_argument("--apply", action="store_true",
                    help="即使没传 --password 也强行提交（会 401）")
    a = ap.parse_args(argv)

    headers = {"X-TRPG-Confirm": "true"}
    if a.password:
        headers["Authorization"] = "Bearer " + a.password

    try:
        cfg = requests.get(a.diceframe + "/api/config", timeout=30).json()
    except Exception as exc:  # noqa: BLE001
        print(f"读不到 DiceFrame 配置（{a.diceframe}/api/config）：{exc}")
        return 1

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = os.path.join(BACKUP_DIR, f"diceframe-config-{stamp}.json")
    with open(backup, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    print("已备份 DiceFrame 公开配置到:", backup)

    body = build_body(cfg, a.bridge, a.model)
    print("\n将要提交的改动：")
    print("  imagegen_provider_ref =", body["imagegen_provider_ref"])
    print("  imagegen_model        =", body["imagegen_model"])
    print("  imagegen_enabled      =", body["imagegen_enabled"])
    print("  imagegen_timeout      =", body["imagegen_timeout_seconds"], "秒")
    print("  服务商 base_url       =", a.bridge, "(api_format=openai)")
    print("  保留的原有服务商      =",
          ", ".join(p.get("name") or p.get("id") or "?" for p in body["ai_providers"][:-1]) or "(无)")

    if not a.password and not a.apply:
        print("\n[干跑] 没有传 --password，未做任何修改。")
        print("       确认无误后重新运行： python wire_diceframe.py --password 你的访问密码")
        return 0

    r = requests.post(a.diceframe + "/api/config", json=body, headers=headers, timeout=60)
    print("\nPOST /api/config ->", r.status_code)
    print(r.text[:1200])
    if r.status_code >= 400:
        if r.status_code == 401:
            print("提示：401 说明访问密码不对，或者根本没传 --password。")
        return 1

    after = requests.get(a.diceframe + "/api/config", timeout=30).json()
    print("\n现在的生图配置：")
    for k in ("imagegen_enabled", "imagegen_provider", "imagegen_model",
              "imagegen_provider_ref", "imagegen_square_size",
              "imagegen_landscape_size", "imagegen_timeout_seconds"):
        print(f"  {k} = {after.get(k)!r}")
    st = requests.get(a.diceframe + "/api/image-generation", timeout=30)
    print("\nGET /api/image-generation ->", st.status_code, st.text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
