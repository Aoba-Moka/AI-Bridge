#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comfy_bridge.py —— DiceFrame  ⇄  ComfyUI  桥接服务

作用
----
把本地 ComfyUI 包装成一个 **OpenAI 兼容的图像生成 API**，让 DiceFrame 的
「OpenAI 兼容服务商」生图通道可以直接把命令发给本服务，本服务再转成
ComfyUI 的 /prompt 工作流执行，并把出图按 OpenAI 响应格式返回给 DiceFrame。

对外接口（OpenAI Images API 子集）
---------------------------------
    GET  /                     服务信息
    GET  /health               健康检查（含 ComfyUI 可达性 / 队列 / 显存）
    GET  /v1/models            OpenAI 风格模型列表
    POST /v1/images/generations  文生图  (JSON)      -> {"data":[{"b64_json":...}]}
    POST /v1/images/edits        图生图  (multipart) -> {"data":[{"b64_json":...}]}
    GET  /files/<name>         当 response_format=url 时取回图片

依赖
----
仅使用 Python 标准库 + 原生 `requests`（用户明确要求，不使用 comfy-sdk / flask 等）。

ComfyUI 侧使用的节点（全部为内置节点，无需额外插件）
--------------------------------------------------
CheckpointLoaderSimple / CLIPTextEncode / EmptyLatentImage / LatentUpscale /
KSampler / VAEDecode / VAEEncode / LoadImage / ImageScale / SaveImage

SD1.5 画布策略
--------------
DiceFrame 默认请求 1792x1024（横）与 1024x1024（方）。DreamShaper 8 是 SD1.5
系底模，直接按 1792x1024 出图会崩。所以本服务：
  1) 按目标宽高比把「基础分辨率」压到约 base_pixels 像素（默认 393216 ≈ 768x512）；
  2) 用 LatentUpscale + 第二个 KSampler 做一次 hires-fix（长边至多 hires_max，默认 1280）；
  3) 最后用 ImageScale(lanczos) 精确缩放到 DiceFrame 要求的尺寸。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import mimetypes
import os
import random
import re
import struct
import sys
import threading
import time
import uuid
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import requests

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

VERSION = "1.4.1"
MAX_BODY_BYTES = 48 * 1024 * 1024          # 请求体上限（参考图最多 12MB + 开销）
MAX_OUT_IMAGE_BYTES = 20 * 1024 * 1024     # DiceFrame 端硬上限
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")

DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry, artist name, ugly, deformed, out of frame"
)

# Anima（anima-base-v1.0 / miaomiaoHarem_aniaAnime…）是二次元 DiT 底模，
# 用 danbooru 风格标签，负向词跟 SD1.5 那套不一样。
DEFAULT_NEGATIVE_ANIMA = (
    "lowres, worst quality, low quality, bad anatomy, bad hands, extra digits, "
    "fewer digits, missing fingers, jpeg artifacts, watermark, signature, username, "
    "text, error, blurry, censored, bar censor"
)

# ⚠️ 千万不要往上面这段负向词里加 letterboxed / black bars / borders / frame / film strip。
# 曾经因为一张 scene 图偶然出现影院黑边而加过这 5 个词，结果 DiceFrame 头像用途的后缀
#     "Single character portrait, centered composition, clear face, no text, no frame."
# 会让 Anima 把人物画成 1024x1024 画布正中一个约 256x371 的小方块，四周一大片纯白。
# 2026-10-02 实测（同一条头像提示词，只改负向词）：
#     负向含这 5 个词          -> 纯白占比 90.9%（画面正中一小块人像）
#     负向只含 letterboxed, black bars -> 纯白占比 90.1%（照样坏）
#     负向不含这 5 个词        -> 正常满幅人像（白底人像，主体占满画幅高度）
# 同时确认：scene 提示词在去掉这 5 个词之后依旧满幅、没有再出现黑边。
# 也就是说那 5 个词既没用又有害，负面"frame"和正向"no frame"打架会把构图搞崩。

# Anima 的官方推荐装载方式（取自用户本机 ComfyUI 工作流 跑团.json）：
#   UNETLoader(diffusion_models/anima-base-v1.0.safetensors, weight_dtype=default)
#   CLIPLoader(text_encoders/qwen_3_06b_base.safetensors, type=stable_diffusion, device=default)
#   VAELoader(vae/qwen_image_vae.safetensors)
#   EmptySD3LatentImage  ->  KSampler(steps=20, cfg=8.0, euler/simple)  ->  VAEDecode
ANIMA_UNET = "anima-base-v1.0.safetensors"
ANIMA_CLIP = "qwen_3_06b_base.safetensors"
ANIMA_CLIP_TYPE = "stable_diffusion"
ANIMA_VAE = "qwen_image_vae.safetensors"
ANIMA_BASE_PIXELS = 1048576          # 1024x1024，Anima 的原生分辨率
ANIMA_BASE_MAX_LONG = 1024           # 基础分辨率的长边上限（超出会变慢且画质反而退化）
ANIMA_STEPS = 20
ANIMA_CFG = 8.0
ANIMA_SAMPLER = "euler"
ANIMA_SCHEDULER = "simple"
ANIMA_HIRES_MAX = 1536
ANIMA_HIRES_STEPS = 8

LOG = logging.getLogger("comfy-bridge")


class BridgeError(Exception):
    """带 HTTP 状态码的业务错误。"""

    def __init__(self, status: int, message: str,
                 kind: str = "invalid_request_error", code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind
        self.code = code


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

class Config:
    def __init__(self, **kw: Any) -> None:
        self.host: str = kw.get("host", "127.0.0.1")
        self.port: int = int(kw.get("port", 8190))
        self.comfy: str = str(kw.get("comfy", "http://127.0.0.1:8188")).rstrip("/")
        self.checkpoint: str = kw.get("checkpoint", "") or ""
        self.steps: int = int(kw.get("steps", 28))
        self.cfg: float = float(kw.get("cfg", 7.0))
        self.sampler: str = kw.get("sampler", "dpmpp_2m")
        self.scheduler: str = kw.get("scheduler", "karras")
        self.negative: str = kw.get("negative", DEFAULT_NEGATIVE)
        self.negative_extra: str = str(kw.get("negative_extra", "") or "").strip()
        self.base_pixels: int = int(kw.get("base_pixels", 393216))
        self.hires: bool = bool(kw.get("hires", True))
        self.hires_max: int = int(kw.get("hires_max", 1280))
        self.hires_steps: int = int(kw.get("hires_steps", 12))
        self.hires_denoise: float = float(kw.get("hires_denoise", 0.45))
        self.edits_denoise: float = float(kw.get("edits_denoise", 0.65))
        self.edits_fit: str = kw.get("edits_fit", "center")
        self.edits_mode: str = kw.get("edits_mode", "img2img")
        # ---- Anima（二次元 DiT 引擎）-------------------------------------- #
        # engine=auto 时按请求的 model 字段自动选：命中 diffusion_models 里的
        # 文件就走 Anima，否则走 SD1.5 的 CheckpointLoaderSimple 流程。
        self.engine: str = kw.get("engine", "auto")            # auto | sd15 | anima
        self.anima_unet: str = kw.get("anima_unet", ANIMA_UNET)
        self.anima_clip: str = kw.get("anima_clip", ANIMA_CLIP)
        self.anima_clip_type: str = kw.get("anima_clip_type", ANIMA_CLIP_TYPE)
        self.anima_vae: str = kw.get("anima_vae", ANIMA_VAE)
        self.anima_negative: str = kw.get("anima_negative", DEFAULT_NEGATIVE_ANIMA)
        self.anima_negative_extra: str = str(kw.get("anima_negative_extra", "") or "").strip()
        self.anima_base_pixels: int = int(kw.get("anima_base_pixels", ANIMA_BASE_PIXELS))
        self.anima_base_max_long: int = int(kw.get("anima_base_max_long", ANIMA_BASE_MAX_LONG))
        self.anima_steps: int = int(kw.get("anima_steps", ANIMA_STEPS))
        self.anima_cfg: float = float(kw.get("anima_cfg", ANIMA_CFG))
        self.anima_sampler: str = kw.get("anima_sampler", ANIMA_SAMPLER)
        self.anima_scheduler: str = kw.get("anima_scheduler", ANIMA_SCHEDULER)
        self.anima_hires: bool = bool(kw.get("anima_hires", True))
        self.anima_hires_max: int = int(kw.get("anima_hires_max", ANIMA_HIRES_MAX))
        self.anima_hires_steps: int = int(kw.get("anima_hires_steps", ANIMA_HIRES_STEPS))
        self.anima_hires_denoise: float = float(kw.get("anima_hires_denoise", 0.45))
        self.anima_lora: str = kw.get("anima_lora", "") or ""
        self.anima_lora_strength: float = float(kw.get("anima_lora_strength", 0.9))
        # 可选：把中文提示词先翻成英文（SD1.5 的 CLIP 只认英文）
        self.translate_url: str = str(kw.get("translate_url") or "").rstrip("/")
        self.translate_model: str = kw.get("translate_model", "deepseek-chat")
        self.translate_key: str = kw.get("translate_key", "")
        self.translate_timeout: float = float(kw.get("translate_timeout", 20.0))
        # 中文翻译失败时怎么办：error=直接报错（默认，避免白图）；warn=照旧硬画
        self.on_translate_failure: str = str(kw.get("on_translate_failure", "error")).lower()
        # 指向 DiceFrame 的 data 目录时，每次翻译前会按文件 mtime 重新读一遍它的
        # 服务商/Key，这样在 DiceFrame 界面里改完 Key 不用重启桥接。
        self.translate_from_diceframe: str = str(kw.get("translate_from_diceframe", "") or "")
        # 固定提示词：每次生图都自动带上（加在正文前/后，拼在翻译之后）
        self.positive_prefix: str = str(kw.get("positive_prefix") or "").strip()
        self.positive_suffix: str = str(kw.get("positive_suffix") or "").strip()
        self.max_size: int = int(kw.get("max_size", 2048))
        self.job_timeout: float = float(kw.get("job_timeout", 280.0))
        self.poll_interval: float = float(kw.get("poll_interval", 0.4))
        self.filename_prefix: str = kw.get("filename_prefix", "DiceFrame/DF")
        self.url_store: str = kw.get("url_store", "")
        self.dry_run: bool = bool(kw.get("dry_run", False))
        self.verbose: bool = bool(kw.get("verbose", False))

    def __repr__(self) -> str:
        return f"<Config {self.host}:{self.port} comfy={self.comfy} ckpt={self.checkpoint or 'auto'}>"


# --------------------------------------------------------------------------- #
# 小工具：尺寸 / 图片头解析
# --------------------------------------------------------------------------- #

def _r8(v: float) -> int:
    """取 8 的倍数（Stable Diffusion 的 VAE 下采样要求）。"""
    return max(8, int(round(v / 8.0)) * 8)


def fit_dims(aspect: float, budget: int) -> Tuple[int, int]:
    """按给定宽高比和像素预算，算出一组 SD 友好的基础分辨率。"""
    aspect = max(0.05, min(20.0, float(aspect)))
    w = math.sqrt(float(budget) * aspect)
    h = float(budget) / w
    return _r8(w), _r8(h)


def parse_size(size: Any) -> Tuple[int, int]:
    """解析 '1792x1024' / '1024*1024' / '1024' / 'auto'。"""
    text = str(size or "").strip().lower().replace(" ", "")
    if not text or text in ("auto", "default"):
        return 1024, 1024
    if "x" in text:
        a, _, b = text.partition("x")
    elif "*" in text:
        a, _, b = text.partition("*")
    else:
        a = b = text
    try:
        w, h = int(float(a)), int(float(b))
    except ValueError as exc:
        raise BridgeError(400, f"size 参数无法解析：{size!r}，应为 '宽x高'，例如 '1024x1024'") from exc
    return w, h


def clamp_dims(w: int, h: int, max_size: int) -> Tuple[int, int]:
    """把目标尺寸收进 [64, max_size]，并保证是 8 的倍数。"""
    if w <= 0 or h <= 0:
        raise BridgeError(400, f"size 必须为正数：{w}x{h}")
    if max(w, h) > max_size:
        k = max_size / float(max(w, h))
        w, h = int(w * k), int(h * k)
    return _r8(max(64, w)), _r8(max(64, h))


def image_size(data: bytes) -> Optional[Tuple[int, int]]:
    """不依赖 Pillow，从图片字节头解析 (width, height)。失败返回 None。"""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", data[16:24])
            return int(w), int(h)
        if data[:3] == b"\xff\xd8\xff":
            i = 2
            n = len(data)
            while i + 9 < n:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return int(w), int(h)
                i += 2 + seg_len
            return None
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            fourcc = data[12:16]
            if fourcc == b"VP8X":
                w = 1 + int.from_bytes(data[24:27], "little")
                h = 1 + int.from_bytes(data[27:30], "little")
                return w, h
            if fourcc == b"VP8 ":
                m = re.search(rb"\x9d\x01\x2a(..)(..)", data[:64], re.S)
                if m:
                    w = struct.unpack("<H", m.group(1))[0] & 0x3FFF
                    h = struct.unpack("<H", m.group(2))[0] & 0x3FFF
                    return w, h
            if fourcc == b"VP8L":
                b = int.from_bytes(data[21:25], "little")
                return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
            return None
        if data[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", data[6:10])
            return int(w), int(h)
        if data[:2] == b"BM":
            w, h = struct.unpack("<ii", data[18:26])
            return abs(int(w)), abs(int(h))
    except Exception:  # noqa: BLE001 - 解析失败就当未知
        return None
    return None


def sniff_image(data: bytes) -> Optional[str]:
    """按魔数返回 MIME。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:2] == b"BM":
        return "image/bmp"
    return None


def ext_for(mime: Optional[str], filename: str = "") -> str:
    if mimetypes.guess_extension(mime or "") in (".png", ".jpg", ".jpeg", ".webp"):
        return mimetypes.guess_extension(mime or "")  # type: ignore[return-value]
    m = os.path.splitext(filename or "")[1].lower()
    if m in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        return m
    return ".png"


# --------------------------------------------------------------------------- #
# multipart/form-data 解析（自写，避免 3.13 起移除的 cgi 模块）
# --------------------------------------------------------------------------- #

class FormPart:
    __slots__ = ("name", "filename", "content_type", "data")

    def __init__(self, name: Optional[str], filename: Optional[str],
                 content_type: str, data: bytes) -> None:
        self.name = name
        self.filename = filename
        self.content_type = content_type
        self.data = data

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", "replace")


def _split_headers_body(part: bytes) -> Tuple[bytes, bytes]:
    i = part.find(b"\r\n\r\n")
    if i >= 0:
        return part[:i], part[i + 4:]
    i = part.find(b"\n\n")
    if i >= 0:
        return part[:i], part[i + 2:]
    return part, b""


def parse_multipart(body: bytes, content_type: str) -> List[FormPart]:
    wrapper = Message()
    wrapper["content-type"] = content_type
    boundary = wrapper.get_param("boundary", header="content-type")
    if not boundary:
        raise BridgeError(400, "multipart 请求缺少 boundary")
    delim = b"--" + boundary.encode("utf-8", "replace")
    out: List[FormPart] = []
    for chunk in body.split(delim)[1:]:
        if chunk.startswith(b"--"):          # 结束分隔符
            break
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        elif chunk.startswith(b"\n"):
            chunk = chunk[1:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        elif chunk.endswith(b"\n"):
            chunk = chunk[:-1]
        raw_head, data = _split_headers_body(chunk)
        header_msg = Message()
        for line in raw_head.decode("utf-8", "replace").replace("\r\n", "\n").split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                header_msg[key.strip()] = value.strip()
        disp = header_msg.get("content-disposition", "")
        name = filename = None
        if disp:
            dm = Message()
            dm["content-disposition"] = disp
            name = dm.get_param("name", header="content-disposition")
            filename = dm.get_param("filename", header="content-disposition")
        out.append(FormPart(name, filename, header_msg.get_content_type(), data))
    return out


# --------------------------------------------------------------------------- #
# ComfyUI 客户端
# --------------------------------------------------------------------------- #

class ComfyUI:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.base = cfg.comfy
        self.client_id = str(uuid.uuid4())
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"comfy-bridge/{VERSION}"})

    # -- 基础 ------------------------------------------------------------- #

    def ping(self) -> Dict[str, Any]:
        try:
            r = self.session.get(f"{self.base}/system_stats", timeout=10)
            r.raise_for_status()
            return {"ok": True, "stats": r.json()}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def checkpoints(self) -> List[str]:
        r = self.session.get(f"{self.base}/object_info/CheckpointLoaderSimple", timeout=30)
        r.raise_for_status()
        info = r.json()
        return list(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])

    def resolve_checkpoint(self, requested: str = "") -> str:
        """把 DiceFrame 传来的 model 字段映射到实际底模文件名。"""
        available = self.checkpoints()
        if not available:
            raise BridgeError(503, "ComfyUI 里没有任何 checkpoint，请先在 models/checkpoints 放置底模",
                              kind="server_error")
        wanted = (requested or self.cfg.checkpoint or "").strip()
        if not wanted:
            return self.cfg.checkpoint if self.cfg.checkpoint in available else available[0]
        if wanted in available:
            return wanted
        for name in available:
            if name.lower() == wanted.lower():
                return name
        stem = os.path.splitext(wanted)[0].lower()
        for name in available:
            if os.path.splitext(name)[0].lower() == stem:
                return name
        for name in available:
            if stem and stem in name.lower():
                return name
        LOG.warning("model=%r 未匹配到已安装底模，回退到 %r", requested, available[0])
        return available[0]

    # -- 模型清单 --------------------------------------------------------- #

    def models(self, folder: str) -> List[str]:
        """用 GET /models/<folder> 列出某类模型（比 object_info 便宜）。"""
        try:
            r = self.session.get(f"{self.base}/models/{folder}", timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            LOG.debug("GET /models/%s 失败：%s", folder, exc)
            return []
        if isinstance(data, list):
            return [str(x) for x in data]
        return []

    def diffusion_models(self) -> List[str]:
        """models/diffusion_models —— Anima 这类 UNETLoader 用的模型。"""
        return self.models("diffusion_models")

    def resolve_anima(self, requested: str = "") -> str:
        """把 DiceFrame 传来的 model 字段映射到 diffusion_models 里的文件名。"""
        available = self.diffusion_models()
        if not available:
            raise BridgeError(503, "ComfyUI 的 models/diffusion_models 里没有任何模型",
                              kind="server_error")
        wanted = (requested or "").strip()
        if wanted in available:
            return wanted
        if wanted:
            low = wanted.lower()
            stem = os.path.splitext(wanted)[0].lower()
            for name in available:
                if name.lower() == low or os.path.splitext(name)[0].lower() == stem:
                    return name
            for name in available:
                if stem and (stem in name.lower() or name.lower() in stem):
                    return name
            LOG.warning("model=%r 未匹配到 diffusion_models，回退到默认 %r", requested, available[0])
        if self.cfg.anima_unet in available:
            return self.cfg.anima_unet
        return available[0]

    def loaders_available(self) -> Dict[str, List[str]]:
        """健康检查用：Anima 需要的三个加载器各有哪些文件。"""
        return {
            "diffusion_models": self.diffusion_models(),
            "text_encoders": self.models("text_encoders"),
            "vae": self.models("vae"),
        }

    def upload_image(self, filename: str, data: bytes, subfolder: str = "") -> str:
        """上传参考图到 ComfyUI input 目录，返回 LoadImage 可用的名称。"""
        files = {"image": (filename, data, sniff_image(data) or "image/png")}
        form = {"type": "input", "overwrite": "true"}
        if subfolder:
            form["subfolder"] = subfolder
        r = self.session.post(f"{self.base}/upload/image", files=files, data=form, timeout=120)
        if r.status_code >= 400:
            raise BridgeError(502, f"向 ComfyUI 上传参考图失败 HTTP {r.status_code}: {r.text[:300]}",
                              kind="server_error")
        info = r.json()
        name = str(info.get("name") or filename)
        sub = str(info.get("subfolder") or "")
        return f"{sub}/{name}" if sub else name

    def queue(self, graph: Dict[str, Any]) -> str:
        payload = {"prompt": graph, "client_id": self.client_id}
        r = self.session.post(f"{self.base}/prompt", json=payload, timeout=60)
        if r.status_code >= 400:
            raise BridgeError(502, f"ComfyUI 拒绝工作流 HTTP {r.status_code}: {r.text[:500]}",
                              kind="server_error")
        data = r.json()
        if data.get("error"):
            err = data["error"]
            detail = err.get("details") or err.get("message") or json.dumps(err, ensure_ascii=False)
            raise BridgeError(502, f"ComfyUI 工作流校验失败：{detail}", kind="server_error")
        node_errors = data.get("node_errors") or {}
        if node_errors:
            raise BridgeError(502, f"ComfyUI 节点错误：{json.dumps(node_errors, ensure_ascii=False)[:500]}",
                              kind="server_error")
        pid = data.get("prompt_id")
        if not pid:
            raise BridgeError(502, "ComfyUI 未返回 prompt_id", kind="server_error")
        return str(pid)

    @staticmethod
    def history_error(entry: Dict[str, Any]) -> str:
        status = entry.get("status") or {}
        for message in status.get("messages") or []:
            try:
                kind, payload = message[0], message[1]
            except Exception:  # noqa: BLE001
                continue
            if kind in ("execution_error", "execution_interrupted"):
                return (f"{kind} @ node {payload.get('node_id')} "
                        f"({payload.get('node_type')}): {payload.get('exception_message')}")
        return ""

    def wait(self, prompt_id: str) -> Dict[str, Any]:
        """轮询 /history 直到出图或失败。

        超时判断必须放在循环**最顶部**。早期版本把它塞在「history 里还没这条记录」
        那个分支里，于是只要 ComfyUI 已经写进 history、状态却迟迟不 completed，
        循环就会一直 `continue` 而永远不超时（实测有一次硬等到 699 秒才回 504，
        客户端早该在 180 秒就断开了）。
        """
        deadline = time.monotonic() + self.cfg.job_timeout
        last_note = 0.0
        while True:
            if time.monotonic() > deadline:
                try:
                    self.session.post(f"{self.base}/interrupt", timeout=10)
                except Exception:  # noqa: BLE001
                    pass
                raise BridgeError(504, f"ComfyUI 生成超时（>{self.cfg.job_timeout:.0f}s）", kind="server_error")
            try:
                r = self.session.get(f"{self.base}/history/{prompt_id}", timeout=20)
                r.raise_for_status()
                hist = r.json()
            except Exception as exc:  # noqa: BLE001
                raise BridgeError(502, f"轮询 ComfyUI 历史失败：{type(exc).__name__}: {exc}",
                                  kind="server_error") from exc
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status and not status.get("completed") and status.get("status_str") not in ("success", "error"):
                    time.sleep(self.cfg.poll_interval)
                    continue
                return entry
            if time.monotonic() - last_note > 15:
                last_note = time.monotonic()
                LOG.info("等待 ComfyUI 出图 prompt_id=%s ...", prompt_id[:8])
            time.sleep(self.cfg.poll_interval)

    def fetch_image(self, item: Dict[str, Any]) -> bytes:
        params = {
            "filename": item.get("filename", ""),
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
        r = self.session.get(f"{self.base}/view", params=params, timeout=120)
        if r.status_code >= 400:
            raise BridgeError(502, f"从 ComfyUI 取图失败 HTTP {r.status_code}", kind="server_error")
        return r.content


# --------------------------------------------------------------------------- #
# 工作流构建
# --------------------------------------------------------------------------- #

def plan_sizes(out_w: int, out_h: int, cfg: Config) -> Tuple[int, int, Optional[Tuple[int, int]]]:
    """返回 (基础宽, 基础高, hires 尺寸或 None)。"""
    aspect = out_w / float(out_h)
    bw, bh = fit_dims(aspect, cfg.base_pixels)
    if not cfg.hires:
        return bw, bh, None
    long_base = max(bw, bh)
    target_long = min(max(out_w, out_h), cfg.hires_max)
    if target_long <= long_base:
        return bw, bh, None
    scale = target_long / float(long_base)
    hw, hh = _r8(bw * scale), _r8(bh * scale)
    if (hw, hh) == (bw, bh):
        return bw, bh, None
    return bw, bh, (hw, hh)


def _sampler(node_id: str, title: str, *, model: Sequence[Any], positive: Sequence[Any],
             negative: Sequence[Any], latent: Sequence[Any], seed: int, steps: int,
             cfg: float, sampler: str, scheduler: str, denoise: float) -> Dict[str, Any]:
    return {
        "class_type": "KSampler",
        "inputs": {
            "model": list(model), "positive": list(positive), "negative": list(negative),
            "latent_image": list(latent), "seed": int(seed), "steps": int(steps),
            "cfg": float(cfg), "sampler_name": sampler, "scheduler": scheduler,
            "denoise": float(denoise),
        },
        "_meta": {"title": title},
    }


def _tail(graph: Dict[str, Any], cfg: Config, last_latent: Sequence[Any],
          vae: Sequence[Any], out_w: int, out_h: int) -> str:
    """接上 VAEDecode → ImageScale → SaveImage，返回 SaveImage 节点 id。"""
    n = len(graph) + 1

    def nid() -> str:
        nonlocal n
        while str(n) in graph:
            n += 1
        v = str(n)
        n += 1
        return v

    dec = nid()
    graph[dec] = {"class_type": "VAEDecode", "inputs": {"samples": list(last_latent), "vae": list(vae)},
                  "_meta": {"title": "VAE Decode"}}
    sc = nid()
    graph[sc] = {"class_type": "ImageScale",
                 "inputs": {"image": [dec, 0], "upscale_method": "lanczos",
                            "width": out_w, "height": out_h, "crop": "disabled"},
                 "_meta": {"title": f"Resize -> {out_w}x{out_h}"}}
    sv = nid()
    graph[sv] = {"class_type": "SaveImage",
                 "inputs": {"images": [sc, 0], "filename_prefix": cfg.filename_prefix},
                 "_meta": {"title": "Save (DiceFrame)"}}
    return sv


def build_txt2img(cfg: Config, *, ckpt: str, prompt: str, negative: str,
                  out_w: int, out_h: int, seed: int, batch: int,
                  steps: Optional[int] = None, cdim: Optional[float] = None,
                  sampler: Optional[str] = None, scheduler: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    steps = int(steps or cfg.steps)
    cdim = float(cdim if cdim is not None else cfg.cfg)
    sampler = sampler or cfg.sampler
    scheduler = scheduler or cfg.scheduler

    bw, bh, hires = plan_sizes(out_w, out_h, cfg)
    graph: Dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt},
              "_meta": {"title": f"Load {ckpt}"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]},
              "_meta": {"title": "Positive"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]},
              "_meta": {"title": "Negative"}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": bw, "height": bh, "batch_size": max(1, batch)},
              "_meta": {"title": f"Empty latent {bw}x{bh}"}},
        "5": _sampler("5", f"Base sampler {bw}x{bh}", model=["1", 0], positive=["2", 0],
                      negative=["3", 0], latent=["4", 0], seed=seed, steps=steps, cfg=cdim,
                      sampler=sampler, scheduler=scheduler, denoise=1.0),
    }
    last: List[Any] = ["5", 0]
    if hires:
        hw, hh = hires
        graph["6"] = {"class_type": "LatentUpscale",
                      "inputs": {"samples": last, "upscale_method": "bicubic",
                                 "width": hw, "height": hh, "crop": "disabled"},
                      "_meta": {"title": f"Latent upscale {hw}x{hh}"}}
        graph["7"] = _sampler("7", f"Hires sampler {hw}x{hh}", model=["1", 0], positive=["2", 0],
                              negative=["3", 0], latent=["6", 0], seed=seed + 1,
                              steps=cfg.hires_steps, cfg=cdim, sampler=sampler,
                              scheduler=scheduler, denoise=cfg.hires_denoise)
        last = ["7", 0]
    save = _tail(graph, cfg, last, ["1", 2], out_w, out_h)
    LOG.info("txt2img %dx%d | base %dx%d | hires %s | seed=%d steps=%d cfg=%.1f %s/%s",
             out_w, out_h, bw, bh, f"{hires[0]}x{hires[1]}" if hires else "off",
             seed, steps, cdim, sampler, scheduler)
    return graph, save


def build_img2img(cfg: Config, *, ckpt: str, prompt: str, negative: str, image_ref: str,
                  out_w: int, out_h: int, seed: int,
                  denoise: Optional[float] = None) -> Tuple[Dict[str, Any], str]:
    bw, bh, hires = plan_sizes(out_w, out_h, cfg)
    denoise = float(cfg.edits_denoise if denoise is None else denoise)
    graph: Dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt},
              "_meta": {"title": f"Load {ckpt}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_ref},
              "_meta": {"title": "Reference image"}},
        "3": {"class_type": "ImageScale",
              "inputs": {"image": ["2", 0], "upscale_method": "lanczos",
                         "width": bw, "height": bh, "crop": cfg.edits_fit},
              "_meta": {"title": f"Fit reference -> {bw}x{bh} ({cfg.edits_fit})"}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]},
              "_meta": {"title": "VAE Encode"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]},
              "_meta": {"title": "Positive"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]},
              "_meta": {"title": "Negative"}},
        "7": _sampler("7", f"Img2img sampler {bw}x{bh} d={denoise}", model=["1", 0],
                      positive=["5", 0], negative=["6", 0], latent=["4", 0], seed=seed,
                      steps=cfg.steps, cfg=cfg.cfg, sampler=cfg.sampler,
                      scheduler=cfg.scheduler, denoise=denoise),
    }
    last: List[Any] = ["7", 0]
    if hires:
        hw, hh = hires
        graph["8"] = {"class_type": "LatentUpscale",
                      "inputs": {"samples": last, "upscale_method": "bicubic",
                                 "width": hw, "height": hh, "crop": "disabled"},
                      "_meta": {"title": f"Latent upscale {hw}x{hh}"}}
        graph["9"] = _sampler("9", f"Hires sampler {hw}x{hh}", model=["1", 0], positive=["5", 0],
                              negative=["6", 0], latent=["8", 0], seed=seed + 1,
                              steps=cfg.hires_steps, cfg=cfg.cfg, sampler=cfg.sampler,
                              scheduler=cfg.scheduler, denoise=cfg.hires_denoise)
        last = ["9", 0]
    save = _tail(graph, cfg, last, ["1", 2], out_w, out_h)
    LOG.info("img2img %dx%d | ref=%s base %dx%d | hires %s | denoise=%.2f seed=%d",
             out_w, out_h, image_ref, bw, bh, f"{hires[0]}x{hires[1]}" if hires else "off",
             denoise, seed)
    return graph, save


# --------------------------------------------------------------------------- #
# Anima 引擎（二次元 DiT：anima-base-v1.0 / miaomiaoHarem…）
# --------------------------------------------------------------------------- #
#
# 结构照搬用户本机 ComfyUI 工作流 `user/default/workflows/跑团.json`：
#   UNETLoader(diffusion_models) → CLIPLoader(text_encoders, type=stable_diffusion)
#   → VAELoader(qwen_image_vae) → CLIPTextEncode x2 → EmptySD3LatentImage
#   → KSampler(euler/simple, steps=20, cfg=8.0) → VAEDecode → SaveImage
#
# 与 SD1.5 的区别：CLIP 是 Qwen3-0.6B（多语言），latent 是 SD3 风格的 16 通道，
# 所以要换 EmptySD3LatentImage 和三个独立 Loader。

def plan_sizes_anima(out_w: int, out_h: int, cfg: Config) -> Tuple[int, int, Optional[Tuple[int, int]]]:
    """Anima 的原生分辨率是 1024x1024，返回 (基础宽, 基础高, hires 尺寸或 None)。

    先按像素预算算一组尺寸，再把**长边**收回 anima_base_max_long（默认 1024）——
    不然 16:9 的目标会让基础长边跑到 1352，既慢又超出模型训练分辨率。
    之后如果有需要，再用 LatentUpscale + 第二个 KSampler 抬到 anima_hires_max。
    """
    aspect = out_w / float(out_h)
    bw, bh = fit_dims(aspect, cfg.anima_base_pixels)
    cap = int(cfg.anima_base_max_long or 0)
    if cap > 0 and max(bw, bh) > cap:
        k = cap / float(max(bw, bh))
        bw, bh = _r8(bw * k), _r8(bh * k)
    if not cfg.anima_hires:
        return bw, bh, None
    long_base = max(bw, bh)
    target_long = min(max(out_w, out_h), cfg.anima_hires_max)
    if target_long <= long_base:
        return bw, bh, None
    scale = target_long / float(long_base)
    hw, hh = _r8(bw * scale), _r8(bh * scale)
    if (hw, hh) == (bw, bh):
        return bw, bh, None
    return bw, bh, (hw, hh)


def _anima_loaders(cfg: Config, unet: str) -> Dict[str, Any]:
    """返回 Anima 的三个加载器节点 + (model, clip, vae) 引用。"""
    nodes: Dict[str, Any] = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "default"},
              "_meta": {"title": f"Load UNET {unet}"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": cfg.anima_clip, "type": cfg.anima_clip_type,
                         "device": "default"},
              "_meta": {"title": f"Load CLIP {cfg.anima_clip}"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": cfg.anima_vae},
              "_meta": {"title": f"Load VAE {cfg.anima_vae}"}},
    }
    model: List[Any] = ["1", 0]
    if cfg.anima_lora:
        nodes["10"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": ["1", 0], "lora_name": cfg.anima_lora,
                                  "strength_model": cfg.anima_lora_strength},
                       "_meta": {"title": f"LoRA {cfg.anima_lora}"}}
        model = ["10", 0]
    return {"nodes": nodes, "model": model, "clip": ["2", 0], "vae": ["3", 0]}


def build_txt2img_anima(cfg: Config, *, unet: str, prompt: str, negative: str,
                        out_w: int, out_h: int, seed: int, batch: int,
                        steps: Optional[int] = None, cdim: Optional[float] = None,
                        sampler: Optional[str] = None,
                        scheduler: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    steps = int(steps or cfg.anima_steps)
    cdim = float(cdim if cdim is not None else cfg.anima_cfg)
    sampler = sampler or cfg.anima_sampler
    scheduler = scheduler or cfg.anima_scheduler

    bw, bh, hires = plan_sizes_anima(out_w, out_h, cfg)
    L = _anima_loaders(cfg, unet)
    graph: Dict[str, Any] = dict(L["nodes"])
    graph["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": L["clip"]},
                  "_meta": {"title": "Positive"}}
    graph["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": L["clip"]},
                  "_meta": {"title": "Negative"}}
    graph["6"] = {"class_type": "EmptySD3LatentImage",
                  "inputs": {"width": bw, "height": bh, "batch_size": max(1, batch)},
                  "_meta": {"title": f"Empty SD3 latent {bw}x{bh}"}}
    graph["7"] = _sampler("7", f"Anima base {bw}x{bh}", model=L["model"], positive=["4", 0],
                          negative=["5", 0], latent=["6", 0], seed=seed, steps=steps,
                          cfg=cdim, sampler=sampler, scheduler=scheduler, denoise=1.0)
    last: List[Any] = ["7", 0]
    if hires:
        hw, hh = hires
        graph["8"] = {"class_type": "LatentUpscale",
                      "inputs": {"samples": last, "upscale_method": "bicubic",
                                 "width": hw, "height": hh, "crop": "disabled"},
                      "_meta": {"title": f"Latent upscale {hw}x{hh}"}}
        graph["9"] = _sampler("9", f"Anima hires {hw}x{hh}", model=L["model"], positive=["4", 0],
                              negative=["5", 0], latent=["8", 0], seed=seed + 1,
                              steps=cfg.anima_hires_steps, cfg=cdim, sampler=sampler,
                              scheduler=scheduler, denoise=cfg.anima_hires_denoise)
        last = ["9", 0]
    save = _tail(graph, cfg, last, L["vae"], out_w, out_h)
    LOG.info("anima txt2img %dx%d | base %dx%d | hires %s | unet=%s seed=%d steps=%d cfg=%.1f %s/%s",
             out_w, out_h, bw, bh, f"{hires[0]}x{hires[1]}" if hires else "off",
             unet, seed, steps, cdim, sampler, scheduler)
    return graph, save


def build_img2img_anima(cfg: Config, *, unet: str, prompt: str, negative: str, image_ref: str,
                        out_w: int, out_h: int, seed: int,
                        denoise: Optional[float] = None) -> Tuple[Dict[str, Any], str]:
    bw, bh, hires = plan_sizes_anima(out_w, out_h, cfg)
    denoise = float(cfg.edits_denoise if denoise is None else denoise)
    L = _anima_loaders(cfg, unet)
    graph: Dict[str, Any] = dict(L["nodes"])
    graph["4"] = {"class_type": "LoadImage", "inputs": {"image": image_ref},
                  "_meta": {"title": "Reference image"}}
    graph["5"] = {"class_type": "ImageScale",
                  "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                             "width": bw, "height": bh, "crop": cfg.edits_fit},
                  "_meta": {"title": f"Fit reference -> {bw}x{bh} ({cfg.edits_fit})"}}
    graph["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": L["vae"]},
                  "_meta": {"title": "VAE Encode"}}
    graph["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": L["clip"]},
                  "_meta": {"title": "Positive"}}
    graph["8"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": L["clip"]},
                  "_meta": {"title": "Negative"}}
    graph["9"] = _sampler("9", f"Anima img2img {bw}x{bh} d={denoise}", model=L["model"],
                          positive=["7", 0], negative=["8", 0], latent=["6", 0], seed=seed,
                          steps=cfg.anima_steps, cfg=cfg.anima_cfg, sampler=cfg.anima_sampler,
                          scheduler=cfg.anima_scheduler, denoise=denoise)
    last: List[Any] = ["9", 0]
    if hires:
        hw, hh = hires
        graph["11"] = {"class_type": "LatentUpscale",
                       "inputs": {"samples": last, "upscale_method": "bicubic",
                                  "width": hw, "height": hh, "crop": "disabled"},
                       "_meta": {"title": f"Latent upscale {hw}x{hh}"}}
        graph["12"] = _sampler("12", f"Anima hires {hw}x{hh}", model=L["model"],
                               positive=["7", 0], negative=["8", 0], latent=["11", 0],
                               seed=seed + 1, steps=cfg.anima_hires_steps, cfg=cfg.anima_cfg,
                               sampler=cfg.anima_sampler, scheduler=cfg.anima_scheduler,
                               denoise=cfg.anima_hires_denoise)
        last = ["12", 0]
    save = _tail(graph, cfg, last, L["vae"], out_w, out_h)
    LOG.info("anima img2img %dx%d | ref=%s base %dx%d | hires %s | denoise=%.2f seed=%d",
             out_w, out_h, image_ref, bw, bh, f"{hires[0]}x{hires[1]}" if hires else "off",
             denoise, seed)
    return graph, save


def collect_images(entry: Dict[str, Any], save_node: str) -> List[Dict[str, Any]]:
    outputs = entry.get("outputs") or {}
    order = [save_node] if save_node in outputs else sorted(outputs.keys())
    found: List[Dict[str, Any]] = []
    for node_id in order:
        for item in (outputs.get(node_id) or {}).get("images") or []:
            found.append(item)
    return found


# --------------------------------------------------------------------------- #
# 生成服务
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 可选：中文提示词 → 英文
# --------------------------------------------------------------------------- #

_TRANSLATE_CACHE: Dict[str, str] = {}
# DiceFrame 的 config.json/secrets.json 指纹 + 最近一次解析出的翻译后端，用于热更新
_DF_CREDS_CACHE: Dict[str, Any] = {"sig": None}
_TRANSLATE_SYSTEM = (
    "You translate prompts for a Stable Diffusion image generator. Translate the "
    "user's text into concise English. Keep every visual detail, style word, colour, "
    "camera angle and proper noun. Expand Chinese TRPG / game / scene terms into "
    "plain English visual descriptions (no jargon). Output ONLY the translated "
    "prompt: no quotes, no explanation, no preamble, single line."
)


def needs_translation(text: str) -> bool:
    """出现 CJK 字符就认为需要翻成英文。"""
    return any("\u2e80" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef" for ch in text)


def chat_url(base: str) -> str:
    """把用户填的 base 补成 chat/completions 的完整地址。"""
    base = str(base or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def translate_with_reason(text: str, cfg: Config) -> Tuple[str, Optional[str]]:
    """翻译中文提示词，返回 ``(结果, 失败原因)``；成功时原因为 ``None``。

    网络抖动很常见，所以失败会**重试一次**再放弃。真的失败时把原因（比如
    ``HTTP 401: Authentication Fails...``）带回去，让上层能给出可操作的报错，
    而不是默默出一张白图。
    """
    if not needs_translation(text):
        return text, None
    refresh_diceframe_credentials(cfg)   # 用户在 DiceFrame 里改了 Key 也能立刻生效
    if not cfg.translate_url:
        return text, None
    cache_key = f"{cfg.translate_model}\x00{text}"
    hit = _TRANSLATE_CACHE.get(cache_key)
    if hit is not None:
        LOG.info("命中翻译缓存：%s", text[:40])
        return hit, None
    headers = {"Content-Type": "application/json"}
    if cfg.translate_key:
        headers["Authorization"] = "Bearer " + cfg.translate_key
    payload = {
        "model": cfg.translate_model,
        "temperature": 0.0,
        "messages": [{"role": "system", "content": _TRANSLATE_SYSTEM},
                     {"role": "user", "content": text}],
    }
    last = ""
    for attempt in (1, 2):
        try:
            resp = requests.post(chat_url(cfg.translate_url), json=payload, headers=headers,
                                 timeout=cfg.translate_timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            obj = resp.json()
            out = str(obj["choices"][0]["message"]["content"] or "").strip()
            out = " ".join(out.strip().strip('"').strip().split())
            if not out:
                raise RuntimeError("翻译结果为空")
            LOG.info("中文提示词已翻译：%s → %s", text[:60], out[:200])
            if len(_TRANSLATE_CACHE) > 256:
                _TRANSLATE_CACHE.clear()
            _TRANSLATE_CACHE[cache_key] = out
            return out, None
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}" if not isinstance(exc, RuntimeError) else str(exc)
            if attempt == 1:
                LOG.warning("提示词翻译失败，重试一次：%s", last)
    return text, last


def translate_prompt(text: str, cfg: Config) -> str:
    """中文提示词翻英文；未配置翻译或翻译失败时**原样返回**。

    注意：``Bridge.generate`` 不直接用这个函数，而是用 ``translate_with_reason``，
    因为「翻译失败 + 中文」在 Anima 上必然出白图，需要报错而不是硬着头皮画。
    保留这个 fail-open 的入口是为了单测和外部调用方便。
    """
    out, reason = translate_with_reason(text, cfg)
    if reason:
        LOG.warning("提示词翻译失败，沿用原文：%s", reason)
    return out


def apply_fixed_prompt(text: str, cfg: Config) -> str:
    """把「每次生图都要带上」的固定提示词拼到正文外面。

    ``--positive-prefix`` 加在最前，``--positive-suffix`` 加在最后，用 ``, `` 连接。
    刻意在翻译之后调用：固定提示词是你自己写的英文/标签，不该被翻译器改写。
    两段都为空时原样返回（零开销）。
    """
    pre = str(getattr(cfg, "positive_prefix", "") or "").strip()
    suf = str(getattr(cfg, "positive_suffix", "") or "").strip()
    if not pre and not suf:
        return text
    parts = [p for p in (pre, text.strip(), suf) if p]
    return ", ".join(parts)


def merge_negative(base: str, extra: str) -> str:
    """把用户自己加的负向词**追加**到内置负向词末尾（而不是整体替换）。

    ``--anima-negative`` / ``--negative-prompt`` 是整体替换，很容易一不小心把内置的
    质量词（lowres / bad anatomy / watermark …）全丢掉；本机用户通常只是想"再加几个词"，
    所以单独给一个 ``--*-negative-extra`` 走这条路。

    ⚠️ 不要往这里塞构图类词汇（frame / borders / letterboxed / black bars / film strip）：
    DiceFrame 头像用途的正向后缀含 ``no frame``，负向再出现 frame 类词会把人物缩成画布
    正中一小块、四周全白。详见 DEFAULT_NEGATIVE_ANIMA 上方的注释。
    """
    base = str(base or "").strip()
    extra = str(extra or "").strip()
    if not extra:
        return base
    if not base:
        return extra
    return f"{base}, {extra}"


class Bridge:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.comfy = ComfyUI(cfg)
        self.lock = threading.Lock()
        if cfg.url_store:
            os.makedirs(cfg.url_store, exist_ok=True)

    # -- 引擎选择 --------------------------------------------------------- #

    def _pick_engine(self, model: str) -> Tuple[str, str]:
        """按请求的 model 字段决定用哪套管线，返回 (引擎, 实际模型文件名)。

        auto 模式：能在 models/diffusion_models 里匹配上的走 Anima，否则走 SD1.5。
        另外接受几个别名，方便在 DiceFrame 的「模型」框里手填。
        """
        want = (model or "").strip()
        low = want.lower()
        alias_anima = {"anima", "anime", "anima-base", "anima-base-v1.0", "animabase"}
        alias_sd = {"sd", "sd15", "sd1.5", "checkpoint", "default", "sdxl", ""}

        if self.cfg.engine == "anima":
            return "anima", self.comfy.resolve_anima(want)
        if self.cfg.engine == "sd15":
            return "sd15", self.comfy.resolve_checkpoint(want)

        if low in alias_anima:
            return "anima", self.comfy.resolve_anima(self.cfg.anima_unet)
        if low in alias_sd:
            return "sd15", self.comfy.resolve_checkpoint(want)

        dm = self.comfy.diffusion_models()
        stem = os.path.splitext(want)[0].lower()
        for name in dm:
            nlow = name.lower()
            if nlow == low or os.path.splitext(nlow)[0] == stem or (stem and stem in nlow):
                return "anima", name
        return "sd15", self.comfy.resolve_checkpoint(want)

    def _upload_ref(self, first: FormPart) -> str:
        name = (f"diceframe_ref_{uuid.uuid4().hex[:12]}"
                f"{ext_for(first.content_type, first.filename or '')}")
        return self.comfy.upload_image(name, first.data)

    # -- 单个请求 --------------------------------------------------------- #

    def generate(self, *, prompt: str, size: Any, model: str, n: int = 1,
                 negative: str = "", seed: Optional[int] = None,
                 response_format: str = "b64_json", want_url: bool = False,
                 refs: Sequence[FormPart] = (), denoise: Optional[float] = None,
                 steps: Optional[int] = None, cfg_scale: Optional[float] = None,
                 sampler: Optional[str] = None, scheduler: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.cfg
        if not str(prompt or "").strip():
            raise BridgeError(400, "prompt 不能为空")
        prompt = str(prompt).strip()
        raw_prompt = prompt
        negative = str(negative or "").strip()

        batch = max(1, min(int(n or 1), 4))
        n_note = "" if batch == 1 else f"（一次批出 {batch} 张）"

        out_w, out_h = parse_size(size)
        out_w, out_h = clamp_dims(out_w, out_h, cfg.max_size)
        seed = int(seed) if seed is not None else random.randint(0, 2 ** 31 - 1)

        with self.lock:
            engine, target = self._pick_engine(model)
            use_refs = bool(refs) and cfg.edits_mode == "img2img"

            # 实测结论：Anima 的 Qwen3-0.6B 文本编码器（CLIPLoader type=stable_diffusion）
            # 跟 SD1.5 的 CLIP 一样**不认中文** —— 中文提示词会出随机色块/白图。
            # 所以两个引擎都要先把中文翻成英文（未配置翻译服务时原样送出，并给出警告）。
            if needs_translation(prompt):
                prompt, reason = translate_with_reason(prompt, cfg)
                if needs_translation(prompt):
                    # 中文直接送模型只会得到白图（Anima 几乎全白、SD1.5 是无关画面），
                    # 与其画 35 秒交一张废图，不如把原因说清楚。
                    detail = reason or "没有配置翻译服务（--translate-url / --translate-from-diceframe）"
                    msg = (f"中文提示词没能翻译成英文：{detail}。"
                           f"Anima/SD1.5 的文本编码器都只认英文，中文直接送进去只会得到白图。"
                           f"请检查 DiceFrame「Deepseek」服务商的 API Key，"
                           f"或在本机 start-bridge.bat 里配置 CB_TRANSLATE_URL / CB_TRANSLATE_KEY，"
                           f"也可以直接把提示词写成英文。")
                    if cfg.on_translate_failure == "error":
                        raise BridgeError(502, msg, kind="translation_error",
                                          code="translation_failed")
                    LOG.error(msg)

            # 固定提示词：用户自备文案，刻意放在翻译之后拼接，保证它不会被翻掉。
            prompt = apply_fixed_prompt(prompt, cfg)

            if engine == "anima":
                neg = negative or cfg.anima_negative
                neg = merge_negative(neg, cfg.anima_negative_extra)
                if use_refs:
                    graph, save_node = build_img2img_anima(
                        cfg, unet=target, prompt=prompt, negative=neg,
                        image_ref=self._upload_ref(refs[0]), out_w=out_w, out_h=out_h,
                        seed=seed, denoise=denoise)
                else:
                    if refs:
                        LOG.info("edits-mode=txt2img：忽略 %d 张参考图", len(refs))
                    graph, save_node = build_txt2img_anima(
                        cfg, unet=target, prompt=prompt, negative=neg, out_w=out_w, out_h=out_h,
                        seed=seed, batch=batch, steps=steps, cdim=cfg_scale,
                        sampler=sampler, scheduler=scheduler)
            else:
                # SD1.5 管线（中文已在上面统一处理过）
                neg = negative or cfg.negative
                neg = merge_negative(neg, cfg.negative_extra)
                if use_refs:
                    graph, save_node = build_img2img(
                        cfg, ckpt=target, prompt=prompt, negative=neg,
                        image_ref=self._upload_ref(refs[0]), out_w=out_w, out_h=out_h,
                        seed=seed, denoise=denoise)
                else:
                    if refs:
                        LOG.info("edits-mode=txt2img：忽略 %d 张参考图", len(refs))
                    graph, save_node = build_txt2img(
                        cfg, ckpt=target, prompt=prompt, negative=neg, out_w=out_w, out_h=out_h,
                        seed=seed, batch=batch, steps=steps, cdim=cfg_scale,
                        sampler=sampler, scheduler=scheduler)

            if cfg.dry_run:
                LOG.info("dry-run：跳过 ComfyUI 执行")
                return {"created": int(time.time()), "model": target, "engine": engine,
                        "data": [{"b64_json": base64.b64encode(_dry_png(out_w, out_h)).decode("ascii"),
                                  "revised_prompt": prompt, "seed": seed}]}

            prompt_id = self.comfy.queue(graph)
            LOG.info("已排队 prompt_id=%s engine=%s model=%s %s", prompt_id, engine, target, n_note)
            started = time.monotonic()
            entry = self.comfy.wait(prompt_id)
            elapsed = time.monotonic() - started

            err = self.comfy.history_error(entry)
            if err:
                raise BridgeError(502, f"ComfyUI 执行失败：{err}", kind="server_error")

            images = collect_images(entry, save_node)
            if not images:
                raise BridgeError(502, f"ComfyUI 没有产出图片（prompt_id={prompt_id}）", kind="server_error")

            data: List[Dict[str, Any]] = []
            for item in images[:batch]:
                body = self.comfy.fetch_image(item)
                if not body:
                    continue
                if len(body) > MAX_OUT_IMAGE_BYTES:
                    raise BridgeError(502, "生成的图片超过 20 MB", kind="server_error")
                entry_out: Dict[str, Any] = {"revised_prompt": prompt[:4000], "seed": seed}
                if prompt != raw_prompt:
                    entry_out["original_prompt"] = raw_prompt[:4000]
                if want_url or response_format == "url":
                    entry_out["url"] = self._store(body, item)
                else:
                    entry_out["b64_json"] = base64.b64encode(body).decode("ascii")
                data.append(entry_out)

        if not data:
            raise BridgeError(502, "生成的图片为空", kind="server_error")
        LOG.info("完成 %dx%d，%d 张，耗时 %.1fs（%s）", out_w, out_h, len(data), elapsed, engine)
        return {
            "created": int(time.time()),
            "model": target,
            "engine": engine,
            "data": data,
            "usage": {"bridge_seconds": round(elapsed, 2), "size": f"{out_w}x{out_h}",
                      "seed": seed, "engine": engine, "model": target},
        }

    def _store(self, body: bytes, item: Dict[str, Any]) -> str:
        name = f"{uuid.uuid4().hex}.png"
        path = os.path.join(self.cfg.url_store, name)
        with open(path, "wb") as fh:
            fh.write(body)
        host = self.cfg.host if self.cfg.host not in ("0.0.0.0", "::") else "127.0.0.1"
        return f"http://{host}:{self.cfg.port}/files/{name}"

    def prune_files(self) -> None:
        if not self.cfg.url_store or not os.path.isdir(self.cfg.url_store):
            return
        cutoff = time.time() - 3600
        for name in os.listdir(self.cfg.url_store):
            path = os.path.join(self.cfg.url_store, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                pass


def _dry_png(w: int, h: int) -> bytes:
    """dry-run 用的 1x1 PNG（不引入 Pillow）。"""
    import zlib
    raw = b"\x00" + b"\x00\x00\x00"
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


# --------------------------------------------------------------------------- #
# HTTP 处理
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    server_version = f"ComfyBridge/{VERSION}"
    protocol_version = "HTTP/1.1"

    # -- 基础设施 --------------------------------------------------------- #

    @property
    def cfg(self) -> Config:
        return self.server.cfg  # type: ignore[attr-defined]

    @property
    def bridge(self) -> Bridge:
        return self.server.bridge  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        LOG.debug("%s %s", self.address_string(), fmt % args)

    def log_error(self, fmt: str, *args: Any) -> None:  # noqa: A003
        LOG.warning("%s %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _send_error_json(self, exc: BridgeError) -> None:
        self._send_json(exc.status, {"error": {"message": exc.message, "type": exc.kind,
                                               "code": exc.code, "param": None}})

    def _read_body(self) -> bytes:
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            buf = bytearray()
            while True:
                line = self.rfile.readline(65536).strip()
                if not line:
                    break
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                buf += self.rfile.read(size)
                self.rfile.read(2)
                if len(buf) > MAX_BODY_BYTES:
                    raise BridgeError(413, "请求体过大")
            return bytes(buf)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise BridgeError(413, f"请求体过大（{length} 字节）")
        return self.rfile.read(length) if length > 0 else b""

    def _json_body(self) -> Dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode("utf-8", "replace"))
        except ValueError as exc:
            raise BridgeError(400, f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(obj, dict):
            raise BridgeError(400, "请求体必须是 JSON 对象")
        return obj

    # -- 路由 ------------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        started = time.monotonic()
        status = 200
        try:
            if path in ("/", "/info"):
                self._send_json(200, self._info())
            elif path in ("/health", "/healthz"):
                self._send_json(200, self._health())
            elif path in ("/v1/models", "/models"):
                self._send_json(200, self._models())
            elif path == "/v1/images/generations" and method == "POST":
                self._send_json(200, self._api_generations())
            elif path == "/v1/images/edits" and method == "POST":
                self._send_json(200, self._api_edits())
            elif path.startswith("/files/") and method in ("GET", "HEAD"):
                self._serve_file(path[len("/files/"):])
            else:
                raise BridgeError(404, f"未知接口：{method} {path}")
        except BridgeError as exc:
            status = exc.status
            LOG.warning("%s %s -> %d %s", method, path, status, exc.message)
            self._send_error_json(exc)
        except Exception as exc:  # noqa: BLE001
            status = 500
            LOG.exception("处理 %s %s 时发生未预期错误", method, path)
            try:
                self._send_error_json(BridgeError(500, f"桥接内部错误：{type(exc).__name__}: {exc}",
                                                  kind="server_error"))
            except Exception:  # noqa: BLE001
                pass
        finally:
            LOG.info("%s %s -> %d (%.2fs)", method, path, status, time.monotonic() - started)

    # -- 端点实现 --------------------------------------------------------- #

    def _info(self) -> Dict[str, Any]:
        cfg = self.cfg
        return {
            "service": "comfy-bridge",
            "version": VERSION,
            "description": "把本地 ComfyUI 暴露为 OpenAI 兼容图像 API（供 DiceFrame 使用）",
            "comfyui": cfg.comfy,
            "endpoints": {
                "generate": "POST /v1/images/generations",
                "edit": "POST /v1/images/edits",
                "models": "GET /v1/models",
                "health": "GET /health",
            },
            "defaults": {
                "engine": cfg.engine,
                "checkpoint": cfg.checkpoint or "(auto: 第一个可用底模)",
                "steps": cfg.steps, "cfg": cfg.cfg,
                "sampler": cfg.sampler, "scheduler": cfg.scheduler,
                "hires": cfg.hires, "hires_max": cfg.hires_max,
                "base_pixels": cfg.base_pixels, "max_size": cfg.max_size,
                "edits_denoise": cfg.edits_denoise,
            },
            "anima": {
                "unet": cfg.anima_unet, "clip": cfg.anima_clip,
                "clip_type": cfg.anima_clip_type, "vae": cfg.anima_vae,
                "steps": cfg.anima_steps, "cfg": cfg.anima_cfg,
                "sampler": cfg.anima_sampler, "scheduler": cfg.anima_scheduler,
                "base_pixels": cfg.anima_base_pixels, "hires": cfg.anima_hires,
                "base_max_long": cfg.anima_base_max_long,
                "hires_max": cfg.anima_hires_max, "hires_steps": cfg.anima_hires_steps,
                "lora": cfg.anima_lora or None,
            },
            "translate": {
                "enabled": bool(cfg.translate_url),
                "url": cfg.translate_url or "(未启用；中文提示词会原样送给模型，出图基本是乱码)",
                "model": cfg.translate_model if cfg.translate_url else None,
                "on_failure": cfg.on_translate_failure,
                "note": "实测 SD1.5 的 CLIP 和 Anima 的 Qwen3-0.6B 都只认英文，中文必须翻译",
            },
            "fixed_prompt": {
                "prefix": cfg.positive_prefix or None,
                "suffix": cfg.positive_suffix or None,
                "note": "每次生图都会拼上，用逗号连接，且不参与翻译",
            },
            "negative": {
                "anima_extra": cfg.anima_negative_extra or None,
                "sd15_extra": cfg.negative_extra or None,
                "anima_full": cfg.anima_negative,
                "sd15_full": cfg.negative,
                "note": "用 --anima-negative-extra / --negative-extra 追加；"
                        "别加 frame / borders / letterboxed 这类构图词，会把人物缩成一小块白边图",
            },
        }

    def _health(self) -> Dict[str, Any]:
        ping = self.bridge.comfy.ping()
        out: Dict[str, Any] = {"status": "ok" if ping["ok"] else "degraded",
                               "bridge": "ok", "comfyui": ping["ok"]}
        if ping["ok"]:
            stats = ping["stats"]
            dev = (stats.get("devices") or [{}])[0]
            out["device"] = dev.get("name")
            out["vram_free_mb"] = round((dev.get("vram_free") or 0) / 1048576)
            try:
                q = self.bridge.comfy.session.get(f"{self.cfg.comfy}/queue", timeout=10).json()
                out["queue_running"] = len(q.get("queue_running") or [])
                out["queue_pending"] = len(q.get("queue_pending") or [])
            except Exception:  # noqa: BLE001
                pass
            try:
                out["checkpoints"] = self.bridge.comfy.checkpoints()
            except Exception:  # noqa: BLE001
                pass
            try:
                out["diffusion_models"] = self.bridge.comfy.diffusion_models()
            except Exception:  # noqa: BLE001
                pass
        else:
            out["error"] = ping.get("error")
        return out

    def _models(self) -> Dict[str, Any]:
        """同时列出 SD1.5 底模和 Anima 扩散模型，方便 DiceFrame 侧挑。"""
        names: List[str] = []
        try:
            names.extend(self.bridge.comfy.checkpoints())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("读取底模列表失败：%s", exc)
        try:
            for name in self.bridge.comfy.diffusion_models():
                if name not in names:
                    names.append(name)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("读取 diffusion_models 列表失败：%s", exc)
        if not names:
            raise BridgeError(502, "无法从 ComfyUI 读取任何底模/扩散模型列表", kind="server_error")
        return {"object": "list",
                "data": [{"id": name, "object": "model", "created": int(time.time()),
                          "owned_by": "comfyui",
                          "capabilities": ["image"] if name.endswith((".safetensors", ".ckpt", ".sft")) else []}
                         for name in names]}

    def _api_generations(self) -> Dict[str, Any]:
        body = self._json_body()
        want_url = str(body.get("response_format") or "").lower() == "url"
        return self.bridge.generate(
            prompt=body.get("prompt") or "",
            size=body.get("size") or body.get("resolution") or "1024x1024",
            model=str(body.get("model") or ""),
            n=int(body.get("n") or 1),
            negative=str(body.get("negative_prompt") or body.get("negative") or ""),
            seed=body.get("seed"),
            response_format=str(body.get("response_format") or "b64_json"),
            want_url=want_url,
            denoise=body.get("denoise"),
            steps=body.get("steps"),
            cfg_scale=body.get("cfg_scale") or body.get("guidance_scale"),
            sampler=body.get("sampler") or body.get("sampler_name"),
            scheduler=body.get("scheduler"),
        )

    def _api_edits(self) -> Dict[str, Any]:
        ctype = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in ctype.lower():
            raise BridgeError(400, "POST /v1/images/edits 需要 multipart/form-data 请求体")
        parts = parse_multipart(self._read_body(), ctype)
        fields: Dict[str, str] = {}
        refs: List[FormPart] = []
        for part in parts:
            if part.name in ("image", "image[]", "mask") and (part.filename or part.content_type.startswith("image/")):
                if part.name != "mask" and part.data:
                    refs.append(part)
            elif part.name:
                fields[part.name] = part.text
        if not refs:
            raise BridgeError(400, "请求里没有找到参考图字段（应为 image 或 image[]）")
        for ref in refs:
            if not sniff_image(ref.data):
                raise BridgeError(400, f"参考图 {ref.filename!r} 不是受支持的图片格式（PNG/JPEG/WEBP/GIF/BMP）")
        LOG.info("edits：收到 %d 张参考图 %s", len(refs),
                 [f"{r.filename}({len(r.data)}B,{image_size(r.data)})" for r in refs])
        want_url = str(fields.get("response_format") or "").lower() == "url"
        return self.bridge.generate(
            prompt=fields.get("prompt") or "",
            size=fields.get("size") or "1024x1024",
            model=str(fields.get("model") or ""),
            n=int(fields.get("n") or 1),
            negative=str(fields.get("negative_prompt") or fields.get("negative") or ""),
            seed=int(fields["seed"]) if str(fields.get("seed") or "").strip().isdigit() else None,
            response_format=str(fields.get("response_format") or "b64_json"),
            want_url=want_url,
            refs=refs,
            denoise=float(fields["denoise"]) if str(fields.get("denoise") or "").strip() else None,
        )

    def _serve_file(self, name: str) -> None:
        if not SAFE_NAME_RE.match(name or "") or ".." in (name or ""):
            raise BridgeError(400, "非法文件名")
        root = self.cfg.url_store
        if not root:
            raise BridgeError(404, "本桥接未启用 URL 图片存储")
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            raise BridgeError(404, "文件不存在")
        with open(path, "rb") as fh:
            body = fh.read()
        self._send(200, body, sniff_image(body) or "application/octet-stream")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: Tuple[str, int], cfg: Config, bridge: Bridge) -> None:
        super().__init__(addr, Handler)
        self.cfg = cfg
        self.bridge = bridge


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def _diceframe_data_sig(data_dir: str) -> str:
    """config.json / secrets.json 的 mtime+size 指纹，用来判断要不要重读。"""
    root = os.path.abspath(os.path.expanduser(str(data_dir or "")))
    if os.path.isfile(root):
        root = os.path.dirname(root)
    parts = []
    for name in ("config.json", "secrets.json"):
        try:
            st = os.stat(os.path.join(root, name))
            parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{name}:-")
    return "|".join(parts)


def refresh_diceframe_credentials(cfg: Config) -> None:
    """翻译前刷新一次后端凭据（按 mtime 判定，没变就零开销）。

    DiceFrame 的服务商和 API Key 是能在它界面里随时改的，如果只在启动时读一次，
    用户改完 Key 还得回来重启桥接——很容易踩坑，所以这里做成自动跟随。
    """
    data_dir = getattr(cfg, "translate_from_diceframe", "")
    if not data_dir:
        return
    sig = _diceframe_data_sig(data_dir)
    if sig == _DF_CREDS_CACHE.get("sig"):
        return
    _DF_CREDS_CACHE["sig"] = sig
    creds = diceframe_translate_credentials(data_dir)
    if not creds:
        return
    if (creds["url"] != cfg.translate_url or creds["model"] != cfg.translate_model
            or creds["key"] != cfg.translate_key):
        LOG.info("翻译后端已更新：%s（模型 %s，Key %s）", creds["url"], creds["model"],
                 "已读取" if creds["key"] else "为空")
    cfg.translate_url = creds["url"]
    cfg.translate_model = creds["model"]
    cfg.translate_key = creds["key"]


def diceframe_translate_credentials(data_dir: str) -> Optional[Dict[str, str]]:
    """从 DiceFrame 的 data 目录里直接复用它的 LLM 服务商，当作提示词翻译后端。

    读 ``config.json`` 的 ``llm_provider_ref``（退化顺序：fallback1 → fallback2 →
    第一个非本机的 openai 服务商）拿到 ``base_url`` 与 ``models[0]``，
    再从 ``secrets.json`` 取 ``ai_provider_key_<id>``。

    **刻意不用 ``imagegen_provider_ref``**：接线之后它指向本桥接自己，
    会把翻译请求打回自己形成死循环。本机地址（127.0.0.1/localhost）的服务商
    一律跳过。
    """
    root = os.path.abspath(os.path.expanduser(str(data_dir or "")))
    if os.path.isfile(root):
        root = os.path.dirname(root)
    cfg_path = os.path.join(root, "config.json")
    if not os.path.isfile(cfg_path):
        LOG.warning("--translate-from-diceframe：找不到 %s", cfg_path)
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("--translate-from-diceframe：读取 %s 失败：%s", cfg_path, exc)
        return None

    providers = [p for p in (cfg.get("ai_providers") or [])
                 if isinstance(p, dict) and str(p.get("base_url") or "").strip()]
    if not providers:
        LOG.warning("--translate-from-diceframe：config.json 里没有任何服务商")
        return None
    by_id = {str(p.get("id") or ""): p for p in providers}

    def _is_local(entry: Dict[str, Any]) -> bool:
        host = (urlsplit(str(entry.get("base_url") or "")).hostname or "").lower()
        return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "")

    chosen: Optional[Dict[str, Any]] = None
    for ref_key in ("llm_provider_ref", "fallback1_provider_ref", "fallback2_provider_ref"):
        entry = by_id.get(str(cfg.get(ref_key) or ""))
        if entry is not None and not _is_local(entry):
            chosen = entry
            break
    if chosen is None:
        for entry in providers:
            if str(entry.get("api_format") or "openai") == "openai" and not _is_local(entry):
                chosen = entry
                break
    if chosen is None:
        LOG.warning("--translate-from-diceframe：没有找到非本机的 OpenAI 兼容服务商")
        return None

    key = ""
    sec_path = os.path.join(root, "secrets.json")
    try:
        with open(sec_path, "r", encoding="utf-8") as fh:
            secrets = json.load(fh) or {}
        key = str(secrets.get("ai_provider_key_" + str(chosen.get("id") or "")) or "")
    except FileNotFoundError:
        LOG.warning("--translate-from-diceframe：没有 %s，翻译请求会不带 API Key", sec_path)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("--translate-from-diceframe：读取 %s 失败：%s", sec_path, exc)

    models = [m.strip() for m in (chosen.get("models") or [])
              if isinstance(m, str) and m.strip()]
    return {
        "url": str(chosen["base_url"]).rstrip("/"),
        "model": models[0] if models else "",
        "key": key,
        "name": str(chosen.get("name") or chosen.get("id") or "?"),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description="DiceFrame ⇄ ComfyUI 桥接服务（OpenAI 兼容图像 API）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", default=os.environ.get("CB_HOST", "127.0.0.1"),
                   help="监听地址；只给 DiceFrame 用建议保持 127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("CB_PORT", 8190)), help="监听端口")
    p.add_argument("--comfy", default=os.environ.get("CB_COMFY", "http://127.0.0.1:8188"),
                   help="ComfyUI 地址")
    p.add_argument("--checkpoint", default=os.environ.get("CB_CHECKPOINT", ""),
                   help="底模文件名；留空则自动取 ComfyUI 里第一个")
    p.add_argument("--steps", type=int, default=28, help="基础采样步数")
    p.add_argument("--cfg", type=float, default=7.0, help="CFG（DreamShaper 8 推荐 7）")
    p.add_argument("--sampler", default="dpmpp_2m", help="采样器")
    p.add_argument("--scheduler", default="karras", help="调度器")
    p.add_argument("--negative-prompt", default=os.environ.get("CB_NEGATIVE_PROMPT", DEFAULT_NEGATIVE),
                   help="SD1.5 负向提示词（**整体替换**内置词表）")
    p.add_argument("--negative-extra", default=os.environ.get("CB_NEGATIVE_EXTRA", ""),
                   help="追加到 SD1.5 负向词末尾（保留内置词表，只想再加几个词时用这个）")
    p.add_argument("--base-pixels", type=int, default=393216,
                   help="基础出图像素预算（SD1.5 建议 393216≈768x512，不要超过 589824）")
    p.add_argument("--no-hires", action="store_true", help="关闭 hires-fix 二次采样")
    p.add_argument("--hires-max", type=int, default=1280, help="hires-fix 允许的最大长边")
    p.add_argument("--hires-steps", type=int, default=12, help="hires-fix 步数")
    p.add_argument("--hires-denoise", type=float, default=0.45, help="hires-fix 重绘幅度")
    p.add_argument("--edits-denoise", type=float, default=0.65,
                   help="/v1/images/edits 的重绘幅度（越高越偏离参考图）")
    p.add_argument("--edits-fit", choices=("center", "disabled"), default="center",
                   help="参考图压到基础分辨率时的处理：center=居中裁剪，disabled=拉伸")
    p.add_argument("--edits-mode", choices=("img2img", "txt2img"), default="img2img",
                   help="/v1/images/edits：img2img=用第一张参考图做底图；txt2img=忽略参考图只按提示词生成")
    # ---- Anima（二次元 DiT）------------------------------------------------- #
    p.add_argument("--engine", choices=("auto", "sd15", "anima"), default="auto",
                   help="auto=按请求的 model 字段自动判断（命中 diffusion_models 走 Anima）；"
                        "sd15=一律用 CheckpointLoaderSimple；anima=一律用 Anima 三件套加载器")
    p.add_argument("--anima-unet", default=os.environ.get("CB_ANIMA_UNET", ANIMA_UNET),
                   help="Anima 扩散模型文件名（models/diffusion_models 下）")
    p.add_argument("--anima-clip", default=os.environ.get("CB_ANIMA_CLIP", ANIMA_CLIP),
                   help="Anima 文本编码器文件名（models/text_encoders 下）")
    p.add_argument("--anima-clip-type", default=os.environ.get("CB_ANIMA_CLIP_TYPE", ANIMA_CLIP_TYPE),
                   help="CLIPLoader 的 type 参数")
    p.add_argument("--anima-vae", default=os.environ.get("CB_ANIMA_VAE", ANIMA_VAE),
                   help="Anima VAE 文件名（models/vae 下）")
    p.add_argument("--anima-negative", default=os.environ.get("CB_ANIMA_NEGATIVE", DEFAULT_NEGATIVE_ANIMA),
                   help="Anima 负向提示词（**整体替换**内置词表）")
    p.add_argument("--anima-negative-extra", default=os.environ.get("CB_ANIMA_NEGATIVE_EXTRA", ""),
                   help="追加到 Anima 负向词末尾（保留内置词表，只想再加几个词时用这个）")
    p.add_argument("--anima-steps", type=int, default=ANIMA_STEPS, help="Anima 采样步数")
    p.add_argument("--anima-cfg", type=float, default=ANIMA_CFG, help="Anima CFG")
    p.add_argument("--anima-sampler", default=ANIMA_SAMPLER, help="Anima 采样器")
    p.add_argument("--anima-scheduler", default=ANIMA_SCHEDULER, help="Anima 调度器")
    p.add_argument("--anima-base-pixels", type=int, default=ANIMA_BASE_PIXELS,
                   help="Anima 基础出图像素预算（原生 1024x1024=1048576）")
    p.add_argument("--anima-base-max-long", type=int, default=ANIMA_BASE_MAX_LONG,
                   help="Anima 基础分辨率的长边上限；横版/竖版靠这个收住，再交给 hires 抬分辨率")
    p.add_argument("--anima-hires-steps", type=int, default=ANIMA_HIRES_STEPS,
                   help="Anima hires-fix 步数")
    p.add_argument("--anima-no-hires", action="store_true", help="关闭 Anima 的 hires-fix")
    p.add_argument("--anima-hires-max", type=int, default=ANIMA_HIRES_MAX,
                   help="Anima hires-fix 允许的最大长边")
    p.add_argument("--anima-lora", default=os.environ.get("CB_ANIMA_LORA", ""),
                   help="可选的 Anima LoRA 文件名（LoraLoaderModelOnly）")
    p.add_argument("--anima-lora-strength", type=float, default=0.9, help="Anima LoRA 权重")
    p.add_argument("--max-size", type=int, default=2048, help="输出尺寸上限")
    p.add_argument("--translate-url", default=os.environ.get("CB_TRANSLATE_URL", ""),
                   help="可选的 OpenAI 兼容 chat 接口地址；填了就把中文提示词先翻成英文"
                        "（例：https://api.deepseek.com/v1）")
    p.add_argument("--translate-model", default=os.environ.get("CB_TRANSLATE_MODEL", "deepseek-chat"),
                   help="翻译所用模型名")
    p.add_argument("--translate-key", default=os.environ.get("CB_TRANSLATE_KEY", ""),
                   help="翻译接口 API Key（建议用环境变量 CB_TRANSLATE_KEY 传）")
    p.add_argument("--translate-from-diceframe",
                   default=os.environ.get("CB_TRANSLATE_FROM_DICEFRAME", ""),
                   metavar="DICE_FRAME_DATA_DIR",
                   help="直接复用 DiceFrame 自己配的 LLM 服务商做翻译，免填 URL/Key。"
                        "传它的 data 目录，例如 "
                        "<你的 DiceFrame 目录>\\data（绿色版通常是 "
                        "D:\\某处\\DiceFrame-x.y.z-windows-portable\\data）；"
                        "显式给了 --translate-url / --translate-key 时不生效")
    p.add_argument("--job-timeout", type=float, default=280.0,
                   help="单次 ComfyUI 生成超时秒数（要小于 DiceFrame 的 imagegen_timeout_seconds）")
    p.add_argument("--on-translate-failure", choices=("error", "warn"), default="error",
                   help="中文提示词翻译失败时：error=直接返回 502 报错（默认，避免交出白图）；"
                        "warn=照旧把中文送进模型（一定会出白图/乱码）")
    p.add_argument("--positive-prefix", default=os.environ.get("CB_POSITIVE_PREFIX", ""),
                   help="固定提示词，加在每次生图提示词的最前面（用逗号连接，不参与翻译）")
    p.add_argument("--positive-suffix", default=os.environ.get("CB_POSITIVE_SUFFIX", ""),
                   help="固定提示词，加在每次生图提示词的最后面（用逗号连接，不参与翻译）")
    p.add_argument("--filename-prefix", default="DiceFrame/DF", help="ComfyUI 输出文件名前缀")
    p.add_argument("--url-store", default="", help="启用 response_format=url 时图片落盘目录")
    p.add_argument("--dry-run", action="store_true", help="不真的调用 ComfyUI，返回 1x1 占位图")
    p.add_argument("-v", "--verbose", action="store_true", help="调试日志")

    a = p.parse_args(argv)

    translate_url, translate_key, translate_model = a.translate_url, a.translate_key, a.translate_model
    translate_from_df = ""
    if a.translate_from_diceframe and not translate_url:
        translate_from_df = a.translate_from_diceframe
        reused = diceframe_translate_credentials(translate_from_df)
        if reused:
            translate_url = reused["url"]
            translate_key = translate_key or reused["key"]
            if reused["model"]:
                translate_model = reused["model"]
            _DF_CREDS_CACHE["sig"] = _diceframe_data_sig(translate_from_df)
            LOG.info("提示词翻译复用 DiceFrame 服务商「%s」：%s（模型 %s，Key %s）",
                     reused["name"], translate_url, translate_model,
                     "已读取" if translate_key else "缺失")

    store = a.url_store or os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
    return Config(host=a.host, port=a.port, comfy=a.comfy, checkpoint=a.checkpoint,
                  steps=a.steps, cfg=a.cfg, sampler=a.sampler, scheduler=a.scheduler,
                  negative=a.negative_prompt, base_pixels=a.base_pixels,
                  hires=not a.no_hires, hires_max=a.hires_max, hires_steps=a.hires_steps,
                  hires_denoise=a.hires_denoise, edits_denoise=a.edits_denoise,
                  edits_fit=a.edits_fit, edits_mode=a.edits_mode, max_size=a.max_size,
                  engine=a.engine, anima_unet=a.anima_unet, anima_clip=a.anima_clip,
                  anima_clip_type=a.anima_clip_type, anima_vae=a.anima_vae,
                  anima_negative=a.anima_negative, anima_base_pixels=a.anima_base_pixels,
                  negative_extra=a.negative_extra, anima_negative_extra=a.anima_negative_extra,
                  anima_base_max_long=a.anima_base_max_long,
                  anima_steps=a.anima_steps, anima_cfg=a.anima_cfg,
                  anima_sampler=a.anima_sampler, anima_scheduler=a.anima_scheduler,
                  anima_hires=not a.anima_no_hires, anima_hires_max=a.anima_hires_max,
                  anima_hires_steps=a.anima_hires_steps,
                  anima_lora=a.anima_lora, anima_lora_strength=a.anima_lora_strength,
                  job_timeout=a.job_timeout,
                  translate_url=translate_url, translate_model=translate_model,
                  translate_key=translate_key, on_translate_failure=a.on_translate_failure,
                  translate_from_diceframe=translate_from_df,
                  positive_prefix=a.positive_prefix, positive_suffix=a.positive_suffix,
                  filename_prefix=a.filename_prefix, url_store=store, dry_run=a.dry_run,
                  verbose=a.verbose)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    args = list(sys.argv[1:] if argv is None else argv)
    # 先配好日志再解析参数：parse_args 里解析 DiceFrame 服务商时要打日志。
    logging.basicConfig(
        level=logging.DEBUG if ("-v" in args or "--verbose" in args) else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout)
    cfg = parse_args(args)

    bridge = Bridge(cfg)
    ping = bridge.comfy.ping()
    if ping["ok"]:
        stats = ping["stats"]
        dev = (stats.get("devices") or [{}])[0]
        LOG.info("ComfyUI 连接正常：%s | %s | 显存空闲 %.1f GB",
                 cfg.comfy, dev.get("name"),
                 (dev.get("vram_free") or 0) / 1073741824)
        try:
            ckpts = bridge.comfy.checkpoints()
            LOG.info("可用底模（SD1.5 管线）：%s", ", ".join(ckpts) or "(无)")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("读取底模列表失败：%s", exc)
        try:
            unets = bridge.comfy.diffusion_models()
            LOG.info("可用扩散模型（Anima 管线）：%s", ", ".join(unets) or "(无)")
            if unets:
                LOG.info("Anima 装载：UNET=%s | CLIP=%s(type=%s) | VAE=%s%s",
                         cfg.anima_unet, cfg.anima_clip, cfg.anima_clip_type, cfg.anima_vae,
                         f" | LoRA={cfg.anima_lora}" if cfg.anima_lora else "")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("读取 diffusion_models 列表失败：%s", exc)
    else:
        LOG.warning("暂时连不上 ComfyUI（%s）：%s", cfg.comfy, ping.get("error"))
        LOG.warning("服务仍会启动，等 ComfyUI 起来后自动恢复。")

    srv = Server((cfg.host, cfg.port), cfg, bridge)
    LOG.info("桥接已启动：http://%s:%d   （OpenAI 兼容 base_url 填这个）", cfg.host, cfg.port)
    LOG.info("DiceFrame 侧：base_url = http://%s:%d/v1", cfg.host, cfg.port)
    LOG.info("自检： curl http://%s:%d/health", cfg.host, cfg.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        LOG.info("收到 Ctrl+C，正在退出 …")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
