#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证可选的「中文提示词翻译」链路：用本地 stub chat 服务做正向测试，再验证失败时 fail-open。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import comfy_bridge as cb

REQ = {}


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        REQ["path"] = self.path
        REQ["auth"] = self.headers.get("Authorization")
        REQ["body"] = json.loads(self.rfile.read(n).decode("utf-8"))
        out = json.dumps({
            "choices": [{"message": {
                "content": '"An ancient castle floating above a sea of clouds, '
                           'sunset, warm golden light, fantasy concept art, highly detailed."'}}]
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {extra}")


print("== chat_url ==")
check("裸域名补 /v1/chat/completions",
      cb.chat_url("https://api.deepseek.com") == "https://api.deepseek.com/v1/chat/completions")
check("/v1 结尾", cb.chat_url("https://api.deepseek.com/v1")
      == "https://api.deepseek.com/v1/chat/completions")
check("已是完整地址保持不动", cb.chat_url("http://x/chat/completions")
      == "http://x/chat/completions")

print("== needs_translation ==")
check("纯英文不翻", cb.needs_translation("a red dragon on a tower") is False)
check("中文要翻", cb.needs_translation("一座漂浮在云海之上的城堡") is True)
check("中英混排要翻", cb.needs_translation("dragon 城堡 sunset") is True)
check("日文汉字要翻", cb.needs_translation("浮遊する城") is True)

print("== 正向翻译（stub 服务）==")
ZH = "一座漂浮在云海之上的古老城堡，夕阳西下，暖金色的光芒，奇幻概念艺术，细节丰富"
cfg = cb.Config(translate_url=f"http://127.0.0.1:{port}",
                translate_model="deepseek-chat", translate_key="sk-test")
out = cb.translate_prompt(ZH, cfg)
check("翻成英文", out.startswith("An ancient castle floating"), out)
check("去掉了包裹引号", '"' not in out and "\n" not in out, out)
check("请求打到正确路径", REQ.get("path") == "/v1/chat/completions", REQ.get("path"))
check("带上 Authorization", REQ.get("auth") == "Bearer sk-test", REQ.get("auth"))
check("temperature=0", REQ["body"].get("temperature") == 0.0)
check("system 提示词存在", REQ["body"]["messages"][0]["role"] == "system")
check("原文进 user 消息", REQ["body"]["messages"][1]["content"] == ZH)
again = cb.translate_prompt(ZH, cfg)
check("第二次命中缓存（不再发请求）", again == out)

print("== translate_with_reason（给上层报错用的失败原因）==")
ZH3 = "雪原上的一只白狼，月光，冷色调"
good = cb.Config(translate_url=f"http://127.0.0.1:{port}", translate_model="deepseek-chat")
t_out, t_err = cb.translate_with_reason(ZH3, good)
check("成功时原因为 None", t_err is None and t_out.startswith("An ancient castle"), f"{t_out!r} {t_err!r}")
check("英文 + 未配置 -> (原文, None)",
      cb.translate_with_reason("a red dragon", cb.Config()) == ("a red dragon", None))
dead = cb.Config(translate_url="http://127.0.0.1:1", translate_timeout=0.6,
                 translate_model="dead-model")  # 换模型名，避免命中上面 good 写进缓存的那条
d_out, d_err = cb.translate_with_reason(ZH3, dead)
check("失败时原样返回 + 带回原因", d_out == ZH3 and bool(d_err), repr(d_err)[:200])
check("失败原因是人能看懂的一句话", any(k in (d_err or "") for k in ("Connection", "Max retries", "refused")),
      repr(d_err)[:200])

print("== 复用 DiceFrame 服务商 + 热更新 ==")
import os
import tempfile
import time

tmp = tempfile.mkdtemp(prefix="dfdata-")


def _write_df(base_url, key):
    with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"llm_provider_ref": "p1",
                   "ai_providers": [{"id": "p1", "name": "FakeLLM", "base_url": base_url,
                                     "api_format": "openai", "models": ["fake-model"]}]}, fh)
    with open(os.path.join(tmp, "secrets.json"), "w", encoding="utf-8") as fh:
        json.dump({"ai_provider_key_p1": key}, fh)


_write_df("http://example.invalid/v1", "sk-one")
hot = cb.Config(translate_from_diceframe=tmp)
cb.refresh_diceframe_credentials(hot)
check("读到 url", hot.translate_url == "http://example.invalid/v1", hot.translate_url)
check("读到 model", hot.translate_model == "fake-model", hot.translate_model)
check("读到 key", hot.translate_key == "sk-one", hot.translate_key)

sig_before = cb._diceframe_data_sig(tmp)
time.sleep(0.02)
_write_df("http://example.invalid/v1", "sk-two")
check("文件指纹会变", cb._diceframe_data_sig(tmp) != sig_before)
cb.refresh_diceframe_credentials(hot)
check("改了 Key 后热更新（不用重启桥接）", hot.translate_key == "sk-two", hot.translate_key)

time.sleep(0.02)
_write_df("http://127.0.0.1:8199/v1", "sk-three")
cb.refresh_diceframe_credentials(hot)
check("本机服务商被跳过（防翻译打回自己）",
      hot.translate_url == "http://example.invalid/v1", hot.translate_url)

print("== 边界 ==")
check("未配置翻译时原样返回",
      cb.translate_prompt(ZH, cb.Config()) == ZH)
check("英文在未配置时原样返回",
      cb.translate_prompt("a red dragon", cb.Config()) == "a red dragon")
ZH2 = "幽暗森林里的一座废弃神庙，藤蔓缠绕，雾气弥漫"
bad = cb.Config(translate_url="http://127.0.0.1:1", translate_timeout=1.0)
check("翻译服务不可用时 fail-open 返回原文",
      cb.translate_prompt(ZH2, bad) == ZH2)
check("失败的翻译不进缓存", ZH2 not in cb._TRANSLATE_CACHE)

srv.shutdown()
print(f"\nPASS={ok} FAIL={fail}")
raise SystemExit(1 if fail else 0)
