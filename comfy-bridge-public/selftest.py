#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""comfy_bridge.py 的自检脚本：纯函数 + 工作流 + multipart + 可选真实出图。"""

import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import comfy_bridge as cb  # noqa: E402

ok = 0
fail = 0


def check(label, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label} {extra}")
    else:
        fail += 1
        print(f"  FAIL  {label} {extra}")


print("== 1. 尺寸解析与换算 ==")
check("parse_size 1792x1024", cb.parse_size("1792x1024") == (1792, 1024))
check("parse_size 1024*1024", cb.parse_size("1024*1024") == (1024, 1024))
check("parse_size 1024", cb.parse_size("1024") == (1024, 1024))
check("parse_size auto", cb.parse_size("auto") == (1024, 1024))
try:
    cb.parse_size("wide")
    check("parse_size 非法值应报错", False)
except cb.BridgeError as e:
    check("parse_size 非法值报错", e.status == 400, str(e))

cfg = cb.Config(port=8190, url_store=os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated"))
for size in ("1792x1024", "1024x1024", "1024x1792", "512x512"):
    w, h = cb.parse_size(size)
    bw, bh, hires = cb.plan_sizes(w, h, cfg)
    check(f"plan_sizes {size}", bw % 8 == 0 and bh % 8 == 0 and bw * bh <= 700000,
          f"-> base {bw}x{bh}, hires {hires}")
check("clamp_dims 上限", cb.clamp_dims(4096, 4096, 2048) == (2048, 2048))

print("\n== 2. txt2img 工作流结构 ==")
graph, save = cb.build_txt2img(cfg, ckpt="DreamShaper_8_pruned.safetensors", prompt="a castle",
                               negative="blurry", out_w=1792, out_h=1024, seed=42, batch=1)
classes = {k: v["class_type"] for k, v in graph.items()}
print("  节点:", json.dumps(classes, ensure_ascii=False))
check("含 CheckpointLoaderSimple", "CheckpointLoaderSimple" in classes.values())
check("含 2 个 KSampler（hires）", list(classes.values()).count("KSampler") == 2)
check("含 LatentUpscale", "LatentUpscale" in classes.values())
check("末尾是 SaveImage", graph[save]["class_type"] == "SaveImage")
check("SaveImage 尺寸链路正确",
      graph[save]["inputs"]["images"][0] != save)
scale_node = [k for k, v in graph.items() if v["class_type"] == "ImageScale"][0]
check("ImageScale 输出目标尺寸", (graph[scale_node]["inputs"]["width"],
                                  graph[scale_node]["inputs"]["height"]) == (1792, 1024))

print("\n== 3. img2img 工作流结构 ==")
g2, s2 = cb.build_img2img(cfg, ckpt="DreamShaper_8_pruned.safetensors", prompt="a forest",
                          negative="blurry", image_ref="ref.png", out_w=1024, out_h=1024,
                          seed=7, denoise=0.6)
c2 = {k: v["class_type"] for k, v in g2.items()}
print("  节点:", json.dumps(c2, ensure_ascii=False))
check("含 LoadImage", "LoadImage" in c2.values())
check("含 VAEEncode", "VAEEncode" in c2.values())
kn = [k for k, v in g2.items() if v["class_type"] == "KSampler"][0]
check("img2img denoise 生效", abs(g2[kn]["inputs"]["denoise"] - 0.6) < 1e-6)

print("\n== 4. multipart 解析（模拟 DiceFrame 的 image[] 形态）==")
boundary = "----testboundary123"
parts = []
for name, value in (("model", "comfyui"), ("prompt", "森林中的城堡"), ("n", "1"),
                    ("size", "1792x1024"), ("response_format", "b64_json")):
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
png = cb._dry_png(1, 1)
img_heads = []
for i in (1, 2):
    img_heads.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image[]\"; "
        f"filename=\"reference-{i}.webp\"\r\nContent-Type: image/webp\r\n\r\n")
body = "".join(parts).encode("utf-8")
for head in img_heads:
    body += head.encode("utf-8") + png + b"\r\n"
body += f"--{boundary}--\r\n".encode("utf-8")
parsed = cb.parse_multipart(body, f"multipart/form-data; boundary={boundary}")
names = [(p.name, p.filename, len(p.data)) for p in parsed]
print("  解析结果:", names)
check("字段数量 7", len(parsed) == 7, str(len(parsed)))
check("中文 prompt 正确", [p for p in parsed if p.name == "prompt"][0].text == "森林中的城堡")
check("image[] 两张", len([p for p in parsed if p.name == "image[]"]) == 2)
check("二进制未被破坏", [p for p in parsed if p.name == "image[]"][0].data == png)

print("\n== 5. 图片头解析 ==")
check("PNG 尺寸解析", cb.image_size(png) == (1, 1))
check("PNG 魔数识别", cb.sniff_image(png) == "image/png")
check("非图片返回 None", cb.sniff_image(b"hello world") is None)
check("ext_for webp", cb.ext_for("image/webp", "a.webp") == ".webp")

print("\n== 6. dry-run 端到端（走 Bridge.generate，不碰 ComfyUI）==")
dry = cb.Config(**{**cfg.__dict__, "dry_run": True})
b = cb.Bridge(dry)
res = b.generate(prompt="test", size="1792x1024", model="")
check("返回 data 数组", isinstance(res.get("data"), list) and len(res["data"]) == 1)
check("b64_json 可解码为 PNG",
      res["data"][0]["b64_json"] and base64.b64decode(res["data"][0]["b64_json"])[:8] == b"\x89PNG\r\n\x1a\n")

print("\n== 7. Anima 引擎：尺寸策略 + 工作流结构 ==")
for size, expect_base, expect_hires in (("1024x1024", (1024, 1024), None),
                                        ("1792x1024", (1024, 584), (1536, 880)),
                                        ("1024x1792", (584, 1024), (880, 1536))):
    w, h = cb.parse_size(size)
    bw, bh, hires = cb.plan_sizes_anima(w, h, cfg)
    check(f"plan_sizes_anima {size}", (bw, bh) == expect_base and hires == expect_hires,
          f"-> base {bw}x{bh}, hires {hires}")

ga, sa = cb.build_txt2img_anima(cfg, unet="anima-base-v1.0.safetensors", prompt="1girl, sunset",
                                negative="lowres", out_w=1792, out_h=1024, seed=1, batch=1)
ca = {k: v["class_type"] for k, v in ga.items()}
print("  节点:", json.dumps(ca, ensure_ascii=False))
check("Anima 用 UNETLoader", "UNETLoader" in ca.values())
check("Anima 用 CLIPLoader", "CLIPLoader" in ca.values())
check("Anima 用 VAELoader", "VAELoader" in ca.values())
check("Anima 用 EmptySD3LatentImage", "EmptySD3LatentImage" in ca.values())
check("Anima 不用 CheckpointLoaderSimple", "CheckpointLoaderSimple" not in ca.values())
check("Anima hires 有 2 个 KSampler", list(ca.values()).count("KSampler") == 2)
check("Anima 末尾 SaveImage", ga[sa]["class_type"] == "SaveImage")
un = [k for k, v in ga.items() if v["class_type"] == "UNETLoader"][0]
check("Anima UNETLoader 参数", ga[un]["inputs"]["unet_name"] == "anima-base-v1.0.safetensors"
      and ga[un]["inputs"]["weight_dtype"] == "default")
cl = [k for k, v in ga.items() if v["class_type"] == "CLIPLoader"][0]
check("Anima CLIPLoader type=stable_diffusion",
      ga[cl]["inputs"]["type"] == "stable_diffusion" and ga[cl]["inputs"]["device"] == "default")
va = [k for k, v in ga.items() if v["class_type"] == "VAELoader"][0]
check("Anima VAE 是 qwen_image_vae", ga[va]["inputs"]["vae_name"] == "qwen_image_vae.safetensors")
ka = [k for k, v in ga.items() if v["class_type"] == "KSampler"][0]
check("Anima KSampler 用 euler/simple/cfg8/steps20",
      ga[ka]["inputs"]["sampler_name"] == "euler" and ga[ka]["inputs"]["scheduler"] == "simple"
      and ga[ka]["inputs"]["cfg"] == 8.0 and ga[ka]["inputs"]["steps"] == 20)
check("Anima latent 已按长边上限收住",
      ga[[k for k, v in ga.items() if v["class_type"] == "EmptySD3LatentImage"][0]]["inputs"]["width"] == 1024)

# 回归护栏：默认负向词里绝不能出现构图类词汇。
# 症状：DiceFrame 头像后缀含 "no frame"，负向里的 frame/bars 会把人物缩成画布正中一小块、四周全白。
_banned_neg = ("letterboxed", "black bars", "borders", "frame", "film strip")
_hit = [w for w in _banned_neg if w in cb.DEFAULT_NEGATIVE_ANIMA.lower()]
check("Anima 默认负向词不含构图类词汇（白边回归护栏）", not _hit, f"混入了 {_hit}")
check("Anima 默认负向词仍是基础质量词",
      "lowres" in cb.DEFAULT_NEGATIVE_ANIMA and "bad anatomy" in cb.DEFAULT_NEGATIVE_ANIMA)

lo = cb.Config(**{**cfg.__dict__, "anima_lora": "deepseek_whale_girl_maid_anima_lora_clean.safetensors"})
gl, _ = cb.build_txt2img_anima(lo, unet="anima-base-v1.0.safetensors", prompt="x", negative="",
                               out_w=1024, out_h=1024, seed=1, batch=1)
check("配了 LoRA 才插入 LoraLoaderModelOnly",
      "LoraLoaderModelOnly" in {v["class_type"] for v in gl.values()})

ga2, sa2 = cb.build_img2img_anima(cfg, unet="anima-base-v1.0.safetensors", prompt="x", negative="",
                                  image_ref="ref.png", out_w=1024, out_h=1024, seed=2, denoise=0.6)
ca2 = {k: v["class_type"] for k, v in ga2.items()}
check("Anima img2img 含 LoadImage/VAEEncode", "LoadImage" in ca2.values() and "VAEEncode" in ca2.values())
k2 = [k for k, v in ga2.items() if v["class_type"] == "KSampler"][0]
check("Anima img2img denoise 生效", abs(ga2[k2]["inputs"]["denoise"] - 0.6) < 1e-6)

print("\n== 8. 引擎路由 ==")
live_route = cb.Bridge(cb.Config(port=8190, url_store=cfg.url_store))
if live_route.comfy.ping()["ok"]:
    check("auto: anima 文件名 -> anima", live_route._pick_engine("anima-base-v1.0.safetensors")[0] == "anima")
    check("auto: 别名 anima -> anima", live_route._pick_engine("anima")[0] == "anima")
    check("auto: checkpoint -> sd15",
          live_route._pick_engine("DreamShaper_8_pruned.safetensors")[0] == "sd15")
    check("auto: 空 model -> sd15", live_route._pick_engine("")[0] == "sd15")
    forced = cb.Bridge(cb.Config(**{**cfg.__dict__, "engine": "anima"}))
    check("engine=anima 强制走 Anima", forced._pick_engine("")[0] == "anima")
else:
    print("  SKIP  ComfyUI 不可达，跳过引擎路由测试")

print("\n== 9. 固定提示词（--positive-prefix / --positive-suffix）==")
empty = cb.Config()
check("都没配时空操作", cb.apply_fixed_prompt("a castle", empty) == "a castle")
only_pre = cb.Config(positive_prefix="masterpiece, best quality")
check("只配前缀", cb.apply_fixed_prompt("a castle", only_pre) == "masterpiece, best quality, a castle")
only_suf = cb.Config(positive_suffix="anime style, detailed")
check("只配后缀", cb.apply_fixed_prompt("a castle", only_suf) == "a castle, anime style, detailed")
both = cb.Config(positive_prefix="masterpiece", positive_suffix="anime style")
check("前后都配", cb.apply_fixed_prompt("a castle", both) == "masterpiece, a castle, anime style")
check("配置里的空白被裁掉", cb.apply_fixed_prompt("a castle", cb.Config(positive_prefix="  ", positive_suffix="  ")) == "a castle")
# 固定提示词必须原样进入工作流的正向 CLIPTextEncode，且不参与翻译
fx = cb.Config(positive_prefix="masterpiece", positive_suffix="anime style")
g_fx, s_fx = cb.build_txt2img_anima(fx, unet="anima-base-v1.0.safetensors",
                                    prompt=cb.apply_fixed_prompt("a castle", fx), negative="bad",
                                    out_w=1024, out_h=1024, seed=1, batch=1)
check("固定提示词落到 Anima 正向节点",
      g_fx["4"]["inputs"]["text"] == "masterpiece, a castle, anime style",
      repr(g_fx["4"]["inputs"]["text"]))
check("固定提示词不含中文触发的翻译",
      cb.needs_translation("masterpiece, anime style") is False)
fx_default = cb.Config()
check("默认两者皆空（不改变原有行为）",
      fx_default.positive_prefix == "" and fx_default.positive_suffix == "")
check("Config 暴露 positive_prefix/positive_suffix",
      hasattr(fx, "positive_prefix") and hasattr(fx, "positive_suffix"))

print("\n== 9b. 负向词追加（--anima-negative-extra / --negative-extra）==")
check("都不配时原样返回", cb.merge_negative("lowres, bad anatomy", "") == "lowres, bad anatomy")
check("只有 extra 时返回 extra", cb.merge_negative("", "extra fingers") == "extra fingers")
check("base 和 extra 都空返回空", cb.merge_negative("", "") == "")
check("追加用逗号连接",
      cb.merge_negative("lowres, bad anatomy", "extra fingers, mutated hands")
      == "lowres, bad anatomy, extra fingers, mutated hands")
check("两端空白被裁掉", cb.merge_negative("  lowres  ", "  extra fingers  ")
      == "lowres, extra fingers")
check("追加不会丢掉内置质量词",
      "lowres" in cb.merge_negative(cb.DEFAULT_NEGATIVE_ANIMA, "extra fingers")
      and "extra fingers" in cb.merge_negative(cb.DEFAULT_NEGATIVE_ANIMA, "extra fingers"))
neg_extra_cfg = cb.Config(anima_negative_extra="extra fingers", negative_extra="mutated hands")
check("Config 暴露负向 extra 字段且已 strip",
      neg_extra_cfg.anima_negative_extra == "extra fingers"
      and neg_extra_cfg.negative_extra == "mutated hands")
check("默认负向 extra 为空（不改变原有行为）",
      cb.Config().anima_negative_extra == "" and cb.Config().negative_extra == "")
# 追加后的负向词要真的进到 Anima 的负向 CLIPTextEncode 节点
g_neg, _ = cb.build_txt2img_anima(
    fx, unet="anima-base-v1.0.safetensors", prompt="a castle",
    negative=cb.merge_negative(cb.DEFAULT_NEGATIVE_ANIMA, "extra fingers"),
    out_w=1024, out_h=1024, seed=1, batch=1)
check("追加的负向词落到 Anima 负向节点",
      "extra fingers" in g_neg["5"]["inputs"]["text"]
      and "lowres" in g_neg["5"]["inputs"]["text"],
      repr(g_neg["5"]["inputs"]["text"])[:80])

print("\n== 10. 真实 ComfyUI 连通性 ==")
live = cb.Config(port=8190, url_store=cfg.url_store)
lb = cb.Bridge(live)
ping = lb.comfy.ping()
if not ping["ok"]:
    print("  SKIP  ComfyUI 不可达:", ping.get("error"))
else:
    dev = (ping["stats"].get("devices") or [{}])[0]
    check("ComfyUI 可达", True, f"{dev.get('name')} 显存空闲 "
          f"{(dev.get('vram_free') or 0)/1073741824:.1f}GB")
    ck = lb.comfy.resolve_checkpoint("")
    check("解析到可用底模", bool(ck), ck)
    check("未知 model 回退", lb.comfy.resolve_checkpoint("does-not-exist") == ck)
    dm = lb.comfy.diffusion_models()
    check("读到 diffusion_models", bool(dm), str(dm))
    uk = lb.comfy.resolve_anima("anima-base-v1.0.safetensors")
    check("解析 Anima 扩散模型", uk == "anima-base-v1.0.safetensors", uk)
    check("Anima 未知 model 回退到默认", lb.comfy.resolve_anima("nope-xyz") == "anima-base-v1.0.safetensors")

print("\n== 11. wait() 超时（曾在 history 已有未完成记录时永不超时）==")
import http.server  # noqa: E402
import threading  # noqa: E402

_HIST_PID = "deadbeef-cafe-4000-8000-000000000000"


class _HistStub(http.server.BaseHTTPRequestHandler):
    """/history/<id> 永远返回「记录已存在但没完成」，用来复现那个死循环。"""

    def do_GET(self):  # noqa: N802
        pid = self.path.rstrip("/").rsplit("/", 1)[-1]
        body = json.dumps({pid: {"status": {"completed": False, "status_str": "running"}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):  # noqa: D102
        pass


_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _HistStub)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
_stub_cfg = cb.Config(comfy=f"http://127.0.0.1:{_srv.server_address[1]}", job_timeout=0.5)
_result = {}


def _run_wait():
    try:
        cb.ComfyUI(_stub_cfg).wait(_HIST_PID)
        _result["outcome"] = "没有超时（BUG 回来了）"
    except cb.BridgeError as exc:
        _result["outcome"] = exc.status
    except Exception as exc:  # noqa: BLE001
        _result["outcome"] = type(exc).__name__


_t = threading.Thread(target=_run_wait, daemon=True)
_t.start()
_t.join(10.0)
check("history 有未完成记录时 wait() 抛 504", _result.get("outcome") == 504, str(_result))
check("wait() 没有卡死（在 job_timeout 附近就返回）", not _t.is_alive(), f"线程存活={_t.is_alive()}")
_srv.shutdown()

print(f"\n===== 自检结果：PASS={ok}  FAIL={fail} =====")
sys.exit(1 if fail else 0)
