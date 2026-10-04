# ComfyUI ⇄ DiceFrame 桥接服务

把本地 **ComfyUI** 包装成一个 **OpenAI 兼容的图像生成 API**（只用 Python 标准库 + `requests`），
让 **DiceFrame** 把本地 ComfyUI 当成一个「OpenAI 兼容服务商」来用：

```
DiceFrame ──HTTP(OpenAI 风格)──▶ comfy-bridge :8192 ──HTTP──▶ ComfyUI :8188 ──▶ 本地显卡
   ▲                                                                              │
   └──────────────────── PNG 图片（base64 或 URL） ◀──────────────────────────────┘
```

* 入口文件：`comfy_bridge.py`（无第三方 Web 框架，无 `comfy-sdk`）
* 依赖：**只有 `requests`**（其余全部是标准库：`http.server`、`json`、`base64`、`argparse`、`zlib`…）
* 不依赖 Flask / FastAPI / Pillow

### 前置条件

| 需要 | 说明 |
| --- | --- |
| Python | **3.9+**（3.10–3.13 实测通过）。只要装了 `requests` 即可，`start-bridge.bat` 会自动挑一个可用的解释器 |
| ComfyUI | 默认连 `http://127.0.0.1:8188`，可用 `--comfy` 改 |
| 模型 | 见第 0 节。没有 Anima 也能跑，只用 SD1.5 的 checkpoint 就行 |
| DiceFrame | 已在 **v2.6.1 windows-portable** 上实测。它的「图像生成」走 OpenAI 兼容协议，理论上其它版本也行 |

```bat
pip install -r requirements.txt
```

---

## 0. 用哪个引擎？Anima（动漫）还是 SD1.5（写实）

桥接内置**两套管线**，按请求里的 `model` 字段自动切换：

| 引擎 | 触发条件 | 装载方式 | 实测耗时 |
| --- | --- | --- | --- |
| **Anima**（推荐） | `model` 命中 `models/diffusion_models` 里的文件名（如 `anima-base-v1.0.safetensors`） | `UNETLoader` + `CLIPLoader` + `VAELoader` + `EmptySD3LatentImage` | 二次元插画风格，1024×1024 约 **36s**，1792×1024 约 **80s** |
| SD1.5 | 其它（如 `DreamShaper_8_pruned.safetensors`） | `CheckpointLoaderSimple` | 写实/通用，1024×1024 约 **14s** |

本机实测用的 Anima 三件套（文件名可能不同，可用 `--anima-unet` / `--anima-clip` / `--anima-vae` 覆盖）：

```
models/diffusion_models/anima-base-v1.0.safetensors              ← 默认，画质最好
models/diffusion_models/miaomiaoHarem_aniAnimeColoring10.safetensors  ← 高饱和平涂风格（颜色很冲）
models/text_encoders/qwen_3_06b_base.safetensors                 ← Anima 的文本编码器
models/vae/qwen_image_vae.safetensors                            ← Anima 的 VAE
```

Anima 的装载参数照搬作者的官方工作流：
`CLIPLoader(type=stable_diffusion)`、`KSampler(steps=20, cfg=8.0, euler/simple, denoise=1.0)`。

两个引擎的**共同点**：文本编码器都只认英文（见第 3 节）。
**Anima 的额外好处**：`/v1/images/edits`（图生图）保人物一致性的能力明显强于 SD1.5，
DiceFrame 的 `scene` 用途会带最多 8 张角色头像当参考图，用 Anima 效果好得多。

想强制只用某一种引擎：启动时加 `--engine anima` 或 `--engine sd15`。

---

## 1. 快速开始

1. 先启动 **ComfyUI**，确认 `http://127.0.0.1:8188` 能打开。
2. 双击本目录下的 **`start-bridge.bat`**，或命令行运行 `python comfy_bridge.py --port 8192`。
   启动器会自动找 Python（3.9+ 且装了 `requests`）和 DiceFrame 的 `data` 目录，
   看到下面这几行就算成功：

   ```
   ComfyUI 连接正常：http://127.0.0.1:8188 | <你的显卡> | 显存空闲 4.9 GB
   可用扩散模型（Anima 管线）：anima-base-v1.0.safetensors
   桥接已启动：http://127.0.0.1:8192   （OpenAI 兼容 base_url 填这个）
   DiceFrame 侧：base_url = http://127.0.0.1:8192/v1
   ```

   > 端口用 `set "CB_PORT=8192"` 覆盖（改完 DiceFrame 里那个服务商的 Base URL 也要跟着改）。
   > 如果桥接报「Connection refused」，先看是不是别的进程占着这个端口：
   > `netstat -ano | findstr :8192`。

3. 自检：浏览器打开 <http://127.0.0.1:8192/health>，应返回 `"status": "ok"`。
   想跑更完整的端到端（真的出两张图 + 错误路径）：`python e2e_test.py`
   （默认打 8192，用 `CB_BASE` 改地址）。

> 桥接窗口要一直开着。关掉窗口 = 停止服务。

---

## 2. DiceFrame 侧接线（关键一步）

DiceFrame 里「图像生成」的 base_url 和 API Key **不是**可以直接填的配置项，
它必须从一个 **AI 服务商条目**里解析出来，所以要在 DiceFrame 里**新建一个服务商**。

### 方式一：脚本自动接（需要 DiceFrame 访问密码）

```bat
python wire_diceframe.py --password 你的访问密码            :: 干跑，只打印将提交的改动
python wire_diceframe.py --password 你的访问密码 --apply    :: 真的写进去
```

不加 `--apply` 就是干跑，会先把 DiceFrame 的公开配置备份到 `_diceframe_backup/` 再打印 diff。
密码只用于这一次请求，不会写进任何文件；也可以走环境变量 `DF_PASSWORD`。
要换回 SD1.5：`python wire_diceframe.py --password … --model DreamShaper_8_pruned.safetensors --apply`。

### 方式二：在 DiceFrame 网页界面里手填

打开 DiceFrame → **设置** → **AI 服务商 / 模型服务** 一栏：

| 字段 | 填写内容 |
| --- | --- |
| 名称 | `ComfyUI 本地`（随便起） |
| Base URL / 接口地址 | `http://127.0.0.1:8192/v1`（端口要和桥接实际监听的一致，用 `CB_PORT` 改） |
| API 格式 | **OpenAI 兼容**（必须选这个，否则保存会报「图像生成仅支持 OpenAI 兼容服务商」） |
| API Key | **留空即可**（桥接不校验；DiceFrame 官方也允许本地服务商不带 Key） |
| 模型列表 / 模型名 | `anima-base-v1.0.safetensors`（走 Anima 引擎；填 `DreamShaper_8_pruned.safetensors` 就用 SD1.5） |

保存后，再到 **图像生成** 设置里：

| 字段 | 填写内容 |
| --- | --- |
| 启用图像生成 | ✅ 打开 |
| 服务商 / Provider | 选刚才新建的 **`ComfyUI 本地`** |
| 模型 | `anima-base-v1.0.safetensors`（**不能为空**，否则图像生成会显示「不可用」） |
| 正方形尺寸 | `1024x1024`（默认即可，正好是 Anima 原生分辨率） |
| 横版尺寸 | `1792x1024`（默认即可） |
| 超时秒数 | `300`（**必须调大**：Anima 在 1792×1024 上要约 80 秒，默认 120 秒会中途超时） |

在游戏里点「生成场景图」跑一张，就能看到桥接窗口打印 `已排队 prompt_id=…`。

### 为什么不填 `imagegen_base_url`？

因为 DiceFrame 把 `imagegen_base_url` / `imagegen_api_key` 列为**不可公开配置项**，
运行时才从 `imagegen_provider_ref` 指向的服务商解析出来（源码：`app/src/webui/composition.py`）。
所以「新建服务商 + 让图像生成引用它」是官方唯一支持的接法。

### 三个必须满足的条件（否则图像生成一直显示不可用）

1. `imagegen_enabled` = 开
2. 服务商 Base URL **非空**、**API 格式 = openai**
3. `imagegen_model` **非空**

---

## 3. 中文提示词怎么办？（重要）

**两套引擎都不认中文**，这一条是实测结论，不是猜测：

| 引擎 | 文本编码器 | 中文提示词的实测结果 |
| --- | --- | --- |
| SD1.5（DreamShaper 8） | CLIP（SD1.5） | 出一张**完全不相关**的图（提示词「漂浮在云海上的古老城堡」→ 青绿色池塘和睡莲） |
| Anima（anima-base-v1.0） | Qwen3-0.6B，`CLIPLoader type=stable_diffusion` | 出一张**几乎空白**的图（只有淡淡线稿），或随机色块 |

> 验证方式：直接把你本机那张 `跑团.json` 转成 API 格式跑中文提示词，结果同样是一张黄色贴纸小人
> —— 说明问题在文本编码器/分词器，不在桥接。
> 把 `CLIPLoader` 的 `type` 换成 `qwen_image` 也一样，中文同样无效。

桥接内置了一个**中文→英文翻译**步骤，在生图前先把中文提示词丢给一个 OpenAI 兼容的 chat 接口翻译。
翻译失败会**重试一次**；还是失败就**直接返回 `502` 报错**（而不是硬画一张白图给你）——
因为「中文 + 没翻译」在 Anima 上必然是一张白图，画 35 秒交一张废图不如把原因说清楚。报错长这样：

```json
{"error": {
  "message": "中文提示词没能翻译成英文：HTTP 401: {\"error\":{\"message\":\"Authentication Fails, Your api key: ****0729 is invalid\"…}}。Anima/SD1.5 的文本编码器都只认英文…",
  "type": "translation_error", "code": "translation_failed", "param": null}}
```

> 想回到「翻译失败也照画」的老行为，加 `--on-translate-failure warn`
> （或环境变量 `CB_ON_TRANSLATE_FAILURE=warn`）；默认是 `error`。

启用方式（任选其一）：

**A. 复用 DiceFrame 自己的 LLM 服务商（推荐，默认开启）**

`start-bridge.bat` 会**自动找到** DiceFrame 的 `data` 目录：先看自己所在的盘，再浅层扫各盘根目录
和常见的下载/桌面位置，用「有没有 `data\config.json`，且里面提到 `ai_providers`」来确认是自己人。
找到了就打印 `translate: reuse DiceFrame LLM provider`。想手动指定就加一行：

```bat
set "CB_DICEFRAME_DATA=D:\你的目录\DiceFrame-x.y.z-windows-portable\data"
```

桥接启动时会读这个目录下的 `config.json` + `secrets.json`，按
`llm_provider_ref` → `fallback1` → `fallback2` → 第一个非本机的 OpenAI 兼容服务商
的顺序挑一个，拿它的 `base_url`、`models[0]` 和 `ai_provider_key_<id>` 当翻译后端。
日志会打 `提示词翻译复用 DiceFrame 服务商「Deepseek」：https://api.deepseek.com/v1（模型 deepseek-flash，Key 已读取）`。

> 它**刻意不用** `imagegen_provider_ref`——接线之后那玩意儿指向桥接自己，会形成死循环。
> 本机地址（127.0.0.1 / localhost）的服务商一律跳过。
> 你换了 DiceFrame 里的 LLM 服务商或 API Key，桥接会**按文件 mtime 自动跟着换，不用重启**
> （每次翻译前检查一遍 `config.json` / `secrets.json` 有没有动过，没动就是零开销）。

命令行等价写法：

```bat
python comfy_bridge.py --translate-from-diceframe "<DiceFrame 目录>\data"
```

**B. 指定别的翻译服务（环境变量，改 `start-bridge.bat`）**

```bat
set "CB_TRANSLATE_URL=https://api.deepseek.com/v1"
set "CB_TRANSLATE_MODEL=deepseek-chat"
set "CB_TRANSLATE_KEY=sk-你的key"
```

**C. 命令行参数**

```bat
python comfy_bridge.py --translate-url https://api.deepseek.com/v1 ^
                       --translate-model deepseek-chat ^
                       --translate-key sk-你的key
```

启用后 `/` 接口的 `translate.enabled` 会变成 `true`，日志里会打印
`中文提示词已翻译：一座漂浮在云海之上… → An ancient castle floating above a sea of clouds…`。

**D. 不用翻译**：在 DiceFrame 里直接用英文写画面描述（效果最好）。

---

## 4. 固定提示词：让每张图都带上同一段话

想做「统一画风」「统一质量词」「每个场景都要有某个元素」这类效果，有**两个地方**可以加，按需选一个（也可以同时用）。

### 位置 A：DiceFrame 侧（推荐，改完立刻生效，不用重启桥接）

DiceFrame 自己就有这个功能。每个提示词是按固定顺序拼出来的（源码 `app/src/imagegen/service.py:198-205`）：

```
style_prefix → request_style → rules → template → scene → purpose 后缀
```

各段用空行 `\n\n` 连接。打开 **设置 → 图像生成**，填这几项：

| 界面里的字段 | 配置键 | 效果 |
| --- | --- | --- |
| **统一风格前缀** | `imagegen_style_prefix` | 永远排在最前面，每张图都带 ✅ 这就是你要的「固定提示词」 |
| 手动 / 自动提示词规则 | `imagegen_manual_rules` / `imagegen_auto_rules` | 排在场景描述**之前**，支持 `{scene}` `{narration}` `{actions}` `{panels}` 占位符 |
| 手动 / 自动提示词模板 | `imagegen_manual_prompt` / `imagegen_auto_prompt` | 排在场景描述**之后**（离画面描述最近，通常权重最高） |

想让固定词出现在**末尾**（一般比放开头更有效），就留空「统一风格前缀」，改成在模板里写：

```
{scene}

anime style, highly detailed, cinematic lighting
```

> 旁边那个「AI 优化」按钮（`POST /image-prompts/optimize`）可以用你已配好的 LLM 帮你润色这些字段。

**当前本机的值**：`imagegen_style_prefix` 是**空**的，所以现在已经生效的是 DiceFrame 自动加的用途后缀，例如 scene 用途会带
`Wide cinematic environment scene, no text, no interface elements.`。

### 位置 B：桥接侧（对任何调用方都生效，包括 curl 和别的程序）

```bat
REM 在 start-bridge.bat 里取消注释并改成你要的
set "CB_POSITIVE_PREFIX=masterpiece, best quality, cinematic lighting"
set "CB_POSITIVE_SUFFIX=anime style, highly detailed"
```

或命令行：

```bash
python comfy_bridge.py --positive-prefix "masterpiece, best quality" --positive-suffix "anime style"
```

拼出来就是（`revised_prompt` 字段会原样回显，方便确认）：

```
masterpiece, best quality, cinematic lighting, <DiceFrame 发来的提示词>, anime style, highly detailed
```

规则：
- 用 `, ` 连接，两段都为空时**完全不做任何处理**（默认行为不变）。
- **刻意放在翻译之后拼接** —— 你自己写的固定词不会被翻译器改写。
- 两个引擎、`/generations` 和 `/edits` 都生效。

### A 和 B 怎么选？

| 你想要的 | 用哪个 |
| --- | --- |
| 只想改 DiceFrame 出图的风格，不想重启桥接 | **A** |
| 固定词要排在提示词**末尾** | **A**（模板里写）或 B 的 `--positive-suffix` |
| 让 curl / 其他程序走桥接时也带上 | **B** |
| 想「DiceFrame 的通用风格 + 桥接的底层兜底」两层都要 | 两个都配，会叠加 |

---

## 4b. 负面提示词（不想要什么）加在哪？

**DiceFrame 自己没有负面提示词这个功能** —— 它给服务商发的只有 `{model, prompt, n, size}`（`app/src/imagegen/providers.py`），配置项里也没有对应字段，界面上更找不到。所以负面词**只能加在桥接这一侧**。

### 唯一推荐的做法：`--anima-negative-extra`（追加）

打开 `start-bridge.bat`，把注释里的这一行取消注释、改成你要的词，然后关掉桥接窗口重新双击：

```bat
REM set "CB_ANIMA_NEGATIVE_EXTRA=extra fingers, mutated hands, long neck, off-model"
```

- 它**追加**在内置词表末尾，内置的 `lowres / bad anatomy / watermark / censored …` 一个都不会丢。
- 走 SD1.5 管线（`DreamShaper_8`）时用 `CB_NEGATIVE_EXTRA`，格式一样。
- 只写**英文或 danbooru 标签**，中文会被当成噪声。
- 起效确认：`GET http://127.0.0.1:8192/` 里的 `negative.anima_extra` 应该显示你的词；`negative.anima_full` 是最终完整词表。

命令行等价写法：`python comfy_bridge.py --anima-negative-extra "extra fingers, mutated hands"`。

### 完整替换：`--anima-negative` / `--negative-prompt`

想彻底换掉内置词表就用这个（`CB_ANIMA_NEGATIVE` / `CB_NEGATIVE_PROMPT`）。**慎用**：内置的质量词会被一起丢掉，多数情况下你只是"想再加几个词"，那就用上面的 `-extra`。

### 单次请求带负面词

`POST /v1/images/generations` 的 JSON body、`POST /v1/images/edits` 的 multipart 表单，都认 `negative_prompt`（也接受简写 `negative`）：

```bash
curl -s http://127.0.0.1:8192/v1/images/generations ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"anima-base-v1.0.safetensors\",\"prompt\":\"a castle\",\"size\":\"1024x1024\",\"negative_prompt\":\"extra fingers\"}"
```

DiceFrame 不会带这个字段，所以这条路只对你自己用 curl / 别的程序调桥接时有用。

### ⚠️ 千万别加构图类词汇

**不要**往负面词里写 `frame` / `borders` / `letterboxed` / `black bars` / `film strip`（`out of frame` 也最好避开）。

原因：DiceFrame 给**头像**用途的正向提示词自动追加的后缀是

```
Single character portrait, centered composition, clear face, no text, no frame.
```

正向说 `no frame`、负向说 `frame`，两边打架，Anima 会把人物缩成画布正中一小块、四周大片纯白（v1.2.0 就是这么坏掉的，v1.3.0 已把这几个词从内置词表里删掉，并加了自检护栏）。**同理也别用负面词去压影院黑边** —— 换个 seed 重出就行。

### 负面词加多了会不会变差？

不会。同一 seed 下的对照实测（头像提示词，负向追加 `extra fingers, mutated hands, off-model`）：

| | 纯白占比 | 非白包围盒 |
| --- | --- | --- |
| 内置词表 | 77.4% | 423×869 |
| 内置 + 3 个追加词 | 76.3% | 423×865 |

几乎一样。不同 seed 之间构图差异（全身 / 半身）是模型本身的行为，跟负面词无关。

---

## 5. 接口说明

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 服务信息、当前默认参数、翻译是否启用 |
| GET | `/health` | 桥接 + ComfyUI 健康状态、显存、队列、可用底模 |
| GET | `/v1/models` | 返回 OpenAI 风格的模型列表（内容 = ComfyUI 里的底模文件） |
| POST | `/v1/images/generations` | **文生图**（JSON） |
| POST | `/v1/images/edits` | **图生图 / 参考图**（`multipart/form-data`） |
| GET | `/files/<name>.png` | 当 `response_format=url` 时下载生成的图片 |

### `POST /v1/images/generations`

```json
{
  "model": "DreamShaper_8_pruned.safetensors",
  "prompt": "an ancient castle floating above a sea of clouds, sunset, concept art",
  "negative_prompt": "lowres, bad anatomy, watermark",
  "size": "1024x1024",
  "n": 1,
  "seed": 12345,
  "response_format": "b64_json",
  "steps": 30, "cfg_scale": 7.0,
  "sampler_name": "dpmpp_2m", "scheduler": "karras"
}
```

* `size` 支持 `1024x1024`、`1792x1024`、`1024x1792`、`512x768`…；也可写 `"square"` / `"landscape"` / `"portrait"`。
* 除 `model` / `prompt` / `size` 外**全部可选**，`model` 不认识时会自动回退到本机唯一底模。
* `response_format` 支持 `b64_json`（默认，DiceFrame 用这个）和 `url`。

响应：

```json
{
  "created": 1790866494,
  "model": "DreamShaper_8_pruned.safetensors",
  "data": [{ "b64_json": "iVBORw0KGgo…", "revised_prompt": "…", "seed": 1637002982 }],
  "usage": { "bridge_seconds": 15.8, "size": "1792x1024",
             "seed": 1637002982, "checkpoint": "DreamShaper_8_pruned.safetensors" }
}
```

### `POST /v1/images/edits`

`multipart/form-data`，文件字段名 **`image`**（单张）或 **`image[]`**（多张，只用第一张），
其余字段与 `/generations` 相同。DiceFrame 给 `scene` 用途附带角色头像参考图时走这个接口。

默认行为是 **img2img**：把第一张参考图上传给 ComfyUI 当底图，`denoise` 默认 `0.65`。
想忽略参考图、只按提示词生成，用 `--edits-mode txt2img`。

### 错误格式

```json
{ "error": { "message": "prompt 不能为空", "type": "invalid_request_error",
             "code": null, "param": null } }
```

`type` 取值：`invalid_request_error` / `server_error` / `timeout_error`。
HTTP 状态码：`400` 参数错 / `404` 未知接口 / `413` 请求体过大 / `502` ComfyUI 执行失败 / `504` 超时。

---

## 6. 常用参数

### SD1.5 管线

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8190` | 监听地址。只给本机 DiceFrame 用就别改（本机因端口被僵尸进程占用，实际跑在 **8192**，见第 1 节） |
| `--comfy` | `http://127.0.0.1:8188` | ComfyUI 地址 |
| `--engine` | `auto` | `auto` 按 `model` 自动选；`sd15` / `anima` 强制 |
| `--checkpoint` | 空（自动取第一个） | 底模文件名 |
| `--steps` | `28` | 基础采样步数 |
| `--cfg` | `7.0` | CFG，DreamShaper 8 推荐 7 |
| `--sampler` / `--scheduler` | `dpmpp_2m` / `karras` | 采样器 / 调度器 |
| `--base-pixels` | `393216` | 基础出图像素预算（≈768×512）。**SD1.5 不要超过 589824** |
| `--no-hires` | 关 | 加这个关闭 hires-fix 二次采样 |
| `--hires-max` | `1280` | hires-fix 允许的最大长边 |
| `--hires-steps` / `--hires-denoise` | `12` / `0.45` | hires-fix 步数 / 重绘幅度 |

### Anima 管线

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--anima-unet` | `anima-base-v1.0.safetensors` | `models/diffusion_models` 下的扩散模型 |
| `--anima-clip` / `--anima-clip-type` | `qwen_3_06b_base.safetensors` / `stable_diffusion` | 文本编码器（照搬你本机工作流） |
| `--anima-vae` | `qwen_image_vae.safetensors` | VAE |
| `--anima-steps` / `--anima-cfg` | `20` / `8.0` | Anima 官方推荐值 |
| `--anima-sampler` / `--anima-scheduler` | `euler` / `simple` | 同上 |
| `--anima-negative` | 二次元负向词表 | **整体替换**内置词表，慎用（会连质量词一起丢掉） |
| `--anima-negative-extra` | 空 | **追加**到内置负向词末尾，想"再加几个词"用这个 |
| `--anima-base-max-long` | `1024` | 基础分辨率长边上限；横竖版靠这个收住，再交给 hires 抬 |
| `--anima-hires-max` / `--anima-hires-steps` | `1536` / `8` | Anima 的 hires 上限与步数 |
| `--anima-no-hires` | 关 | 关闭 Anima 的 hires |
| `--anima-lora` / `--anima-lora-strength` | 空 / `0.9` | 可选 LoRA（`LoraLoaderModelOnly`） |

### 两边通用

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--edits-denoise` | `0.65` | `/edits` 的重绘幅度，越高越偏离参考图 |
| `--edits-mode` | `img2img` | `/edits` 行为：`img2img` 用参考图，`txt2img` 忽略参考图 |
| `--max-size` | `2048` | 输出尺寸上限，超过会等比缩小 |
| `--job-timeout` | `280` | 单次生成超时秒数（**必须小于 DiceFrame 的 `imagegen_timeout_seconds`**） |
| `--positive-prefix` | 空 | 固定提示词，加在每次生图提示词**最前面**（逗号连接，不参与翻译） |
| `--positive-suffix` | 空 | 固定提示词，加在每次生图提示词**最后面**（逗号连接，不参与翻译） |
| `--negative-extra` | 空 | 追加到 **SD1.5** 负向词末尾（保留内置词表） |
| `--negative-prompt` | SD1.5 词表 | **整体替换** SD1.5 负向词，慎用 |
| `--translate-url` / `--translate-model` / `--translate-key` | 空 / `deepseek-chat` / 空 | 中文提示词翻译服务 |
| `--on-translate-failure` | `error` | 翻译失败时：`error`=返回 502 报错（默认，避免白图）；`warn`=照旧把中文送进模型 |
| `--url-store` | `./generated` | `response_format=url` 时的落盘目录（1 小时自动清理） |
| `--dry-run` | 关 | 不调用 ComfyUI，直接返回 1×1 占位图，用来测通路 |
| `-v` / `--verbose` | 关 | 调试日志 |

### 为什么 1792×1024 也能出好图？

**SD1.5** 直接画 1792×1024 会出双头 / 重复构图；**Anima** 直接画会又慢又容易重复。
两套管线都自动做了三级处理，只是参数不同：

```
SD1.5  目标 1792x1024
  → 按宽高比压到 base_pixels 393216 → EmptyLatentImage 832x472
  → KSampler 基础采样（28 步）
  → LatentUpscale(bicubic) 放大到 1280x728
  → KSampler hires-fix（12 步，denoise 0.45）
  → VAEDecode → ImageScale(lanczos) 精确缩到 1792x1024
  → SaveImage

Anima  目标 1792x1024
  → 按宽高比算 1352x776，再把长边收回到 1024 → EmptySD3LatentImage 1024x584
  → KSampler 基础采样（20 步，euler/simple，cfg 8.0）
  → LatentUpscale(bicubic) 放大到 1536x880
  → KSampler hires-fix（8 步，denoise 0.45）
  → VAEDecode → ImageScale(lanczos) 精确缩到 1792x1024
  → SaveImage
```

所以任何尺寸都能返回**精确的**目标分辨率。Anima 横向实测 **80 秒**、方形 **36 秒**（RTX 5060 Laptop）。

---

## 7. 排错

| 现象 | 原因 / 处理 |
| --- | --- |
| DiceFrame 图像生成显示**不可用** | 三条件没凑齐：服务商 Base URL 非空 + 格式 openai + `imagegen_model` 非空 |
| 日志 `暂时连不上 ComfyUI` | ComfyUI 没起或不是 `127.0.0.1:8188`；桥接仍会启动，ComfyUI 起来后自动恢复 |
| `HTTP 504 … 超过 280 秒` | 图太大或显卡忙；调大 `--job-timeout` **并且**把 DiceFrame 的 `imagegen_timeout_seconds` 调得更大。<br>**v1.4.1 修了一个相关 bug**：以前只要 ComfyUI 已经写进 history、状态却迟迟不 `completed`，桥接就会一直循环「还没好」而**永远不超时**（实测有一次硬等到 **699 秒**才回 504，客户端早该断了）。现在超时判断挪到循环最顶部，一定会在 `job-timeout` 附近返回 |
| DiceFrame 报「图像生成服务返回 HTTP 400」 | 桥接返回了参数错误，看桥接窗口日志里的具体 message |
| 出图很慢 | 首个请求要加载底模；之后 Anima 方形约 36 秒、横版约 50–80 秒，SD1.5 约 14 秒（RTX 5060 Laptop） |
| 出图是**正中一小块人像、四周大片纯白**（尤其头像） | 曾经的 v1.2.0 及更早版本把 `letterboxed, black bars, borders, frame, film strip` 写进了 Anima 的负向词，和 DiceFrame 头像用途后缀里的 `no frame` 打架，模型会把人物缩成画布正中一小块。**v1.3.0 已删掉这 5 个词**，重启桥接即可恢复。确认方法：`GET /` 里 `version` 至少要是 `1.3.0`。若你手动用 `--anima-negative` 覆盖过，也要把这几个词去掉 |
| 出图上下有**黑边** | 提示词里的 `Wide cinematic` 偶尔被理解成影院画幅。**不要用负向词去压**（见上一行，会给头像带来更严重的白边）；换个 seed 重出即可 |
| 中文提示词出白图/乱码 | **先升级到当前版本**：现在会直接报 `502 translation_failed` 而不是出白图。看到 502 就说明翻译没成功——最常见的原因是 DiceFrame「Deepseek」服务商里的 **API Key 失效/填错**（拿 `curl https://api.deepseek.com/v1/chat/completions -H "Authorization: Bearer <key>"` 试一下）；也可以改 `start-bridge.bat` 里的 `CB_TRANSLATE_URL/MODEL/KEY` 换个翻译后端 |
| DiceFrame 的 LLM 功能（对话/润色）也报错 | 同一个 Key 的问题。翻译是复用它的，所以两边会一起坏 |
| 想确认 ComfyUI 收到任务 | 看 ComfyUI 的 `output\DiceFrame\DF_*.png`（`e2e_test.py` 里设 `COMFY_OUTPUT_DIR` 可直接列出最近几张） |
| 固定提示词没生效 | 看接口回显的 `revised_prompt` 里有没有你的词；桥接侧用 `GET /` 的 `fixed_prompt` 块确认已加载。DiceFrame 侧要按「保存」 |
| 负面提示词没生效 / 想确认加了什么 | `GET /` 里的 `negative` 块：`anima_extra` 是你追加的词，`anima_full` 是最终完整词表。改 `start-bridge.bat` 后必须重启桥接 |
| 端口被占 / 改端口 | `--port 8193`，DiceFrame 的服务商 Base URL 同步改（本机已经在用 8192） |

---

## 8. 文件清单

| 文件 | 说明 |
| --- | --- |
| `comfy_bridge.py` | **主程序**。桥接服务本体（双引擎） |
| `start-bridge.bat` | Windows 一键启动。自动找 Python 和 DiceFrame 目录，崩溃后 5 秒自动重启 |
| `selftest.py` | 纯函数 / 尺寸策略 / 工作流结构 / 引擎路由 / 固定提示词 / 负面词 / **wait() 超时回归** / multipart / 图片头自检（当前 **PASS=80**） |
| `test_translate.py` | 提示词翻译链路测试，自带本地 stub 服务（当前 **PASS=29**） |
| `e2e_test.py` | 真实 HTTP 端到端测试（需要 ComfyUI + 桥接都在跑） |
| `wire_diceframe.py` | 通过 DiceFrame 自己的 `/api/config` 自动接线（需要访问密码） |
| `requirements.txt` | 就一个 `requests` |
| `LICENSE` | MIT |
| `.gitignore` | 已排除 `__pycache__/`、`generated/`、`_test_out/`、`_diceframe_backup/` 等 |

### 关于 `wire_diceframe.py`

DiceFrame 设了访问密码，`POST /api/config` 需要 `Authorization: Bearer <访问密码>`（外加 `X-TRPG-Confirm: true`）。
密码是 `pbkdf2_sha256` 哈希后存在 `data/secrets.json` 的，**没法反推**，所以只能你告诉我或者你自己填。

```bat
python wire_diceframe.py --password 你的访问密码            :: 干跑，先备份 + 打印改动
python wire_diceframe.py --password 你的访问密码 --apply    :: 提交
```

脚本只回传 `id/name/base_url/api_format/models/model_capabilities`，**不会**把
DiceFrame 已经屏蔽的 `api_key` 字段覆盖掉，所以原有服务商（DeepSeek）的密钥是安全的。

不想用脚本就在 DiceFrame 界面里按第 2 节手动填，效果完全一样。
