#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对运行中的桥接服务做端到端 HTTP 测试（模拟 DiceFrame 的真实调用形态）。"""
import base64
import json
import os
import sys
import time

import requests

# 默认对着 8192 的桥接跑；用 CB_BASE 覆盖，例如
#   set CB_BASE=http://127.0.0.1:8190
BASE = os.environ.get("CB_BASE", "http://127.0.0.1:8192").rstrip("/")
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CB_OUT", os.path.join(_HERE, "_test_out"))
os.makedirs(OUT, exist_ok=True)


def hr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


hr("GET /  （服务信息）")
r = requests.get(BASE + "/", timeout=30)
print(r.status_code, json.dumps(r.json(), ensure_ascii=False, indent=2)[:900])

hr("GET /v1/models")
r = requests.get(BASE + "/v1/models", timeout=30)
print(r.status_code, json.dumps(r.json(), ensure_ascii=False)[:400])

hr("GET /health")
r = requests.get(BASE + "/health", timeout=30)
print(r.status_code, json.dumps(r.json(), ensure_ascii=False, indent=2)[:900])

hr("POST /v1/images/generations  (DiceFrame 第 1 次尝试的 body，1792x1024)")
payload = {
    "model": "comfyui",
    "prompt": "Wide cinematic environment scene: an ancient stone castle on a cliff, "
              "stormy sea below, dramatic clouds, torchlight in the windows, "
              "fantasy concept art, highly detailed, no text, no interface elements.",
    "n": 1,
    "size": "1792x1024",
    "response_format": "b64_json",
}
t0 = time.time()
r = requests.post(BASE + "/v1/images/generations", json=payload, timeout=180)
dt = time.time() - t0
print("HTTP", r.status_code, f"{dt:.1f}s")
body = r.json()
if r.status_code >= 400:
    print(json.dumps(body, ensure_ascii=False)[:1500])
    sys.exit(1)
item = body["data"][0]
raw = base64.b64decode(item["b64_json"], validate=True)
print("usage:", json.dumps(body.get("usage"), ensure_ascii=False))
print("bytes:", len(raw), "magic:", raw[:8])
sys.path.insert(0, _HERE)
import comfy_bridge as cb
print("dims:", cb.image_size(raw))
p1 = os.path.join(OUT, "gen_1792x1024.png")
open(p1, "wb").write(raw)
print("saved:", p1)

hr("POST /v1/images/generations  (正方形 1024x1024，中文提示词)")
payload2 = {
    "model": "comfyui",
    "prompt": "a red dragon perched on a snowy mountain peak, epic fantasy, detailed scales",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json",
}
t0 = time.time()
r = requests.post(BASE + "/v1/images/generations", json=payload2, timeout=180)
dt = time.time() - t0
print("HTTP", r.status_code, f"{dt:.1f}s")
body2 = r.json()
if r.status_code >= 400:
    print(json.dumps(body2, ensure_ascii=False)[:1500])
else:
    raw2 = base64.b64decode(body2["data"][0]["b64_json"], validate=True)
    print("bytes:", len(raw2), "dims:", cb.image_size(raw2))
    p2 = os.path.join(OUT, "gen_1024x1024.png")
    open(p2, "wb").write(raw2)
    print("saved:", p2)

hr("POST /v1/images/edits  （multipart，模拟 DiceFrame 的 image[] 参考图）")
ref = open(p1, "rb").read()
form = [("model", (None, "comfyui")), ("prompt", (None, "same castle at golden sunset, warm light")),
        ("n", (None, "1")), ("size", (None, "1024x1024")),
        ("response_format", (None, "b64_json"))]
files = [("image", ("reference-1.png", ref, "image/png"))] + form
t0 = time.time()
r = requests.post(BASE + "/v1/images/edits", files=files, timeout=180)
dt = time.time() - t0
print("HTTP", r.status_code, f"{dt:.1f}s")
body3 = r.json()
if r.status_code >= 400:
    print(json.dumps(body3, ensure_ascii=False)[:1500])
else:
    raw3 = base64.b64decode(body3["data"][0]["b64_json"], validate=True)
    print("bytes:", len(raw3), "dims:", cb.image_size(raw3))
    p3 = os.path.join(OUT, "edit_1024x1024.png")
    open(p3, "wb").write(raw3)
    print("saved:", p3)

hr("错误路径")
r = requests.post(BASE + "/v1/images/generations", json={"prompt": ""}, timeout=30)
print("空 prompt ->", r.status_code, r.text[:200])
r = requests.post(BASE + "/v1/images/generations", json={"prompt": "x", "size": "huge"}, timeout=30)
print("坏 size   ->", r.status_code, r.text[:200])
r = requests.get(BASE + "/nope", timeout=30)
print("未知路由  ->", r.status_code, r.text[:200])

hr("本地实际出图文件")
import glob
import os

out_dir = os.environ.get("COMFY_OUTPUT_DIR", "")
if not out_dir:
    print("（跳过：把 COMFY_OUTPUT_DIR 指向 ComfyUI 的 output\\DiceFrame 目录即可列出产物）")
else:
    for f in glob.glob(os.path.join(out_dir, "*.png"))[-5:]:
        print(f)
print("DONE")
