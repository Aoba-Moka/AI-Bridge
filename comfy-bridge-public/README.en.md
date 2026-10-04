# ComfyUI ⇄ DiceFrame Bridge Service

Wrap local **ComfyUI** as an **OpenAI-compatible image generation API** (using only the Python standard library + `requests`), so that **DiceFrame** can treat local ComfyUI as an "OpenAI-compatible provider":

```
DiceFrame ──HTTP(OpenAI style)──▶ comfy-bridge :8192 ──HTTP──▶ ComfyUI :8188 ──▶ local GPU
   ▲                                                                              │
   └──────────────────── PNG image (base64 or URL) ◀──────────────────────────────┘
```

* Entry point: `comfy_bridge.py` (no third-party web framework, no `comfy-sdk`)
* Dependencies: **only `requests`** (everything else is standard library: `http.server`, `json`, `base64`, `argparse`, `zlib`…)
* Does not depend on Flask / FastAPI / Pillow

### Prerequisites

| Requirement | Description |
| --- | --- |
| Python | **3.9+** (tested on 3.10–3.13). You only need `requests` installed; `start-bridge.bat` will automatically choose a usable interpreter |
| ComfyUI | Defaults to `http://127.0.0.1:8188`; change with `--comfy` |
| Models | See Section 0. It can run without Anima; an SD1.5 checkpoint is enough |
| DiceFrame | Tested on **v2.6.1 windows-portable**. Its "image generation" uses an OpenAI-compatible protocol, so other versions should theoretically work too |

```bat
pip install -r requirements.txt
```

---

## 0. Which Engine? Anima (Anime) or SD1.5 (Realistic)

The bridge has **two built-in pipelines**, and automatically switches based on the `model` field in the request:

| Engine | Trigger | Loading Method | Tested Time |
| --- | --- | --- | --- |
| **Anima** (recommended) | `model` matches a filename in `models/diffusion_models` (e.g. `anima-base-v1.0.safetensors`) | `UNETLoader` + `CLIPLoader` + `VAELoader` + `EmptySD3LatentImage` | Anime illustration style, 1024×1024 about **36s**, 1792×1024 about **80s** |
| SD1.5 | Others (e.g. `DreamShaper_8_pruned.safetensors`) | `CheckpointLoaderSimple` | Realistic/general, 1024×1024 about **14s** |

The Anima trio tested on this machine (filenames may differ; override with `--anima-unet` / `--anima-clip` / `--anima-vae`):

```
models/diffusion_models/anima-base-v1.0.safetensors              ← default, best quality
models/diffusion_models/miaomiaoHarem_aniAnimeColoring10.safetensors  ← high-saturation flat-color style (very vivid colors)
models/text_encoders/qwen_3_06b_base.safetensors                 ← Anima text encoder
models/vae/qwen_image_vae.safetensors                            ← Anima VAE
```

Anima loading parameters copy the author's official workflow:
`CLIPLoader(type=stable_diffusion)`, `KSampler(steps=20, cfg=8.0, euler/simple, denoise=1.0)`.

What the two engines have in common: both text encoders only recognize English (see Section 3).
**Anima's extra benefit**: `/v1/images/edits` (image-to-image) is noticeably better than SD1.5 at preserving character consistency. DiceFrame's `scene` purpose can include up to 8 character avatars as reference images, and Anima works much better for that.

To force only one engine: add `--engine anima` or `--engine sd15` at startup.

---

## 1. Quick Start

1. Start **ComfyUI** first, and confirm `http://127.0.0.1:8188` opens.
2. Double-click **`start-bridge.bat`** in this directory, or run `python comfy_bridge.py --port 8192` from the command line.
   The launcher automatically finds Python (3.9+ with `requests` installed) and DiceFrame's `data` directory.
   If you see the following lines, it succeeded:

   ```
   ComfyUI connection OK: http://127.0.0.1:8188 | <your GPU> | free VRAM 4.9 GB
   Available diffusion models (Anima pipeline): anima-base-v1.0.safetensors
   Bridge started: http://127.0.0.1:8192   (fill this into OpenAI-compatible base_url)
   DiceFrame side: base_url = http://127.0.0.1:8192/v1
   ```

   > Override the port with `set "CB_PORT=8192"` (after changing it, also update the provider's Base URL in DiceFrame).
   > If the bridge reports "Connection refused", first check whether another process is occupying the port:
   > `netstat -ano | findstr :8192`.

3. Self-check: open <http://127.0.0.1:8192/health> in a browser; it should return `"status": "ok"`.
   For a fuller end-to-end test (actually generates two images + error paths): `python e2e_test.py`
   (defaults to 8192; use `CB_BASE` to change the address).

> The bridge window must stay open. Closing the window = stopping the service.

---

## 2. DiceFrame-Side Wiring (Key Step)

In DiceFrame, the base_url and API Key for "image generation" **are not** config items you can fill in directly.
They must be resolved from an **AI provider entry**, so you need to **create a provider** in DiceFrame.

### Method 1: Automatic Wiring via Script (Requires DiceFrame Access Password)

```bat
python wire_diceframe.py --password YOUR_ACCESS_PASSWORD            :: dry run, only prints changes to be submitted
python wire_diceframe.py --password YOUR_ACCESS_PASSWORD --apply    :: actually write them
```

Without `--apply`, it is a dry run. It first backs up DiceFrame's public config to `_diceframe_backup/`, then prints the diff.
The password is only used for this one request and is not written to any file; you can also use the environment variable `DF_PASSWORD`.
To switch back to SD1.5: `python wire_diceframe.py --password … --model DreamShaper_8_pruned.safetensors --apply`.

### Method 2: Fill It In Manually in the DiceFrame Web UI

Open DiceFrame → **Settings** → **AI Providers / Model Services**:

| Field | Value |
| --- | --- |
| Name | `ComfyUI Local` (anything you like) |
| Base URL / API Address | `http://127.0.0.1:8192/v1` (the port must match what the bridge actually listens on; change with `CB_PORT`) |
| API Format | **OpenAI-compatible** (you must choose this, otherwise saving will report "Image generation only supports OpenAI-compatible providers") |
| API Key | **Leave empty** (the bridge does not validate it; DiceFrame officially allows local providers without a Key) |
| Model List / Model Name | `anima-base-v1.0.safetensors` (uses the Anima engine; fill `DreamShaper_8_pruned.safetensors` to use SD1.5) |

After saving, go to **Image Generation** settings:

| Field | Value |
| --- | --- |
| Enable Image Generation | ✅ On |
| Provider | Select the **`ComfyUI Local`** provider you just created |
| Model | `anima-base-v1.0.safetensors` (**cannot be empty**, otherwise image generation will show "Unavailable") |
| Square Size | `1024x1024` (default is fine; exactly Anima's native resolution) |
| Landscape Size | `1792x1024` (default is fine) |
| Timeout Seconds | `300` (**must be increased**: Anima takes about 80 seconds at 1792×1024; the default 120 seconds will time out midway) |

In the game, click "Generate Scene Image" and run one; you should see the bridge window print `queued prompt_id=…`.

### Why Not Fill In `imagegen_base_url`?

Because DiceFrame lists `imagegen_base_url` / `imagegen_api_key` as **non-public config items**,
and resolves them at runtime from the provider pointed to by `imagegen_provider_ref` (source: `app/src/webui/composition.py`).
So "create a provider + make image generation reference it" is the only officially supported wiring method.

### Three Conditions That Must Be Met (Otherwise Image Generation Always Shows Unavailable)

1. `imagegen_enabled` = on
2. Provider Base URL **non-empty**, **API format = openai**
3. `imagegen_model` **non-empty**

---

## 3. What About Chinese Prompts? (Important)

**Neither engine recognizes Chinese.** This is a tested conclusion, not a guess:

| Engine | Text Encoder | Actual Result with Chinese Prompt |
| --- | --- | --- |
| SD1.5 (DreamShaper 8) | CLIP (SD1.5) | Produces a **completely unrelated** image (prompt "漂浮在云海上的古老城堡" → teal pond and water lilies) |
| Anima (anima-base-v1.0) | Qwen3-0.6B, `CLIPLoader type=stable_diffusion` | Produces an **almost blank** image (only faint line art), or random color blocks |

> Verification method: directly convert your local `跑团.json` to API format and run a Chinese prompt; the result is likewise a yellow sticker figure
> — this shows the problem is in the text encoder/tokenizer, not the bridge.
> Changing `CLIPLoader`'s `type` to `qwen_image` also makes no difference; Chinese still does not work.

The bridge has a built-in **Chinese→English translation** step. Before image generation, it sends the Chinese prompt to an OpenAI-compatible chat endpoint for translation.
If translation fails, it **retries once**; if it still fails, it **directly returns a `502` error** (instead of forcing out a white image) —
because "Chinese + no translation" on Anima will inevitably be a white image. Spending 35 seconds to deliver a useless image is worse than clearly stating the reason. The error looks like:

```json
{"error": {
  "message": "Chinese prompt could not be translated to English: HTTP 401: {\"error\":{\"message\":\"Authentication Fails, Your api key: ****0729 is invalid\"…}}. Anima/SD1.5 text encoders only recognize English…",
  "type": "translation_error", "code": "translation_failed", "param": null}}
```

> To return to the old behavior "draw even if translation fails", add `--on-translate-failure warn`
> (or environment variable `CB_ON_TRANSLATE_FAILURE=warn`); the default is `error`.

Enable it in one of the following ways:

**A. Reuse DiceFrame's own LLM provider (recommended, enabled by default)**

`start-bridge.bat` will **automatically find** DiceFrame's `data` directory: first check its own drive, then shallow-scan each drive root and common download/desktop locations, using "does it have `data\config.json` and does it mention `ai_providers`" to confirm it is the right one.
If found, it prints `translate: reuse DiceFrame LLM provider`. To specify manually, add:

```bat
set "CB_DICEFRAME_DATA=D:\your\dir\DiceFrame-x.y.z-windows-portable\data"
```

When the bridge starts, it reads `config.json` + `secrets.json` under this directory, and picks a provider in the order
`llm_provider_ref` → `fallback1` → `fallback2` → first non-local OpenAI-compatible provider,
using its `base_url`, `models[0]`, and `ai_provider_key_<id>` as the translation backend.
The log prints `Prompt translation reusing DiceFrame provider "Deepseek": https://api.deepseek.com/v1 (model deepseek-flash, Key read)`.

> It **deliberately does not use** `imagegen_provider_ref` — after wiring, that points to the bridge itself and would create an infinite loop.
> Providers at local addresses (127.0.0.1 / localhost) are always skipped.
> If you change the LLM provider or API Key in DiceFrame, the bridge will **automatically follow by file mtime, no restart required**
> (before each translation it checks whether `config.json` / `secrets.json` have changed; if not, zero overhead).

Command-line equivalent:

```bat
python comfy_bridge.py --translate-from-diceframe "<DiceFrame dir>\data"
```

**B. Specify another translation service (environment variables, edit `start-bridge.bat`)**

```bat
set "CB_TRANSLATE_URL=https://api.deepseek.com/v1"
set "CB_TRANSLATE_MODEL=deepseek-chat"
set "CB_TRANSLATE_KEY=sk-your-key"
```

**C. Command-line arguments**

```bat
python comfy_bridge.py --translate-url https://api.deepseek.com/v1 ^
                       --translate-model deepseek-chat ^
                       --translate-key sk-your-key
```

After enabling, `translate.enabled` in `/` becomes `true`, and the log prints:
`Chinese prompt translated: 一座漂浮在云海之上… → An ancient castle floating above a sea of clouds…`.

**D. No translation**: write scene descriptions directly in English in DiceFrame (best results).

---

## 4. Fixed Prompts: Make Every Image Carry the Same Text

If you want effects like "unified style", "unified quality words", or "every scene must have a certain element", there are **two places** to add it. Choose one as needed (you can also use both).

### Location A: DiceFrame Side (Recommended; Takes Effect Immediately After Saving, No Bridge Restart Needed)

DiceFrame itself has this feature. Each prompt is assembled in a fixed order (source `app/src/imagegen/service.py:198-205`):

```
style_prefix → request_style → rules → template → scene → purpose suffix
```

Segments are joined with blank lines `\n\n`. Open **Settings → Image Generation** and fill in:

| Field in UI | Config Key | Effect |
| --- | --- | --- |
| **Unified Style Prefix** | `imagegen_style_prefix` | Always at the very front; included in every image ✅ This is the "fixed prompt" you want |
| Manual / Auto Prompt Rules | `imagegen_manual_rules` / `imagegen_auto_rules` | Placed **before** the scene description; supports `{scene}` `{narration}` `{actions}` `{panels}` placeholders |
| Manual / Auto Prompt Template | `imagegen_manual_prompt` / `imagegen_auto_prompt` | Placed **after** the scene description (closest to the image description, usually highest weight) |

To make fixed words appear at **the end** (generally more effective than at the beginning), leave "Unified Style Prefix" empty and write in the template:

```
{scene}

anime style, highly detailed, cinematic lighting
```

> The "AI Optimize" button next to it (`POST /image-prompts/optimize`) can use your configured LLM to polish these fields.

**Current local values**: `imagegen_style_prefix` is **empty**, so what is already in effect is the purpose suffix automatically added by DiceFrame. For example, scene purpose includes:
`Wide cinematic environment scene, no text, no interface elements.`

### Location B: Bridge Side (Effective for Any Caller, Including curl and Other Programs)

```bat
REM In start-bridge.bat, uncomment and change to what you want
set "CB_POSITIVE_PREFIX=masterpiece, best quality, cinematic lighting"
set "CB_POSITIVE_SUFFIX=anime style, highly detailed"
```

Or command line:

```bash
python comfy_bridge.py --positive-prefix "masterpiece, best quality" --positive-suffix "anime style"
```

The assembled result is (`revised_prompt` field echoes it back exactly, for confirmation):

```
masterpiece, best quality, cinematic lighting, <prompt sent by DiceFrame>, anime style, highly detailed
```

Rules:
- Joined with `, `; if both are empty, **no processing at all** (default behavior unchanged).
- **Deliberately concatenated after translation** — your own fixed words will not be rewritten by the translator.
- Works for both engines and for both `/generations` and `/edits`.

### How to Choose Between A and B?

| What You Want | Which to Use |
| --- | --- |
| Only change DiceFrame's output style, without restarting the bridge | **A** |
| Fixed words must be at the **end** of the prompt | **A** (write in template) or B's `--positive-suffix` |
| Make curl / other programs also carry them when calling the bridge | **B** |
| Want both "DiceFrame general style + bridge low-level fallback" | Configure both; they stack |

---

## 4b. Negative Prompts (What You Don't Want): Where to Add Them?

**DiceFrame itself does not have a negative prompt feature** — it only sends `{model, prompt, n, size}` to providers (`app/src/imagegen/providers.py`), there is no corresponding config field, and you cannot find it in the UI. So negative words **can only be added on the bridge side**.

### Only Recommended Method: `--anima-negative-extra` (Append)

Open `start-bridge.bat`, uncomment the line in the comments, change it to your words, then close the bridge window and double-click again:

```bat
REM set "CB_ANIMA_NEGATIVE_EXTRA=extra fingers, mutated hands, long neck, off-model"
```

- It **appends** to the end of the built-in word list; built-in `lowres / bad anatomy / watermark / censored …` are not lost.
- For the SD1.5 pipeline (`DreamShaper_8`), use `CB_NEGATIVE_EXTRA`, same format.
- Write only **English or danbooru tags**; Chinese will be treated as noise.
- Confirm effect: `negative.anima_extra` in `GET http://127.0.0.1:8192/` should show your words; `negative.anima_full` is the final complete word list.

Command-line equivalent: `python comfy_bridge.py --anima-negative-extra "extra fingers, mutated hands"`.

### Full Replacement: `--anima-negative` / `--negative-prompt`

Use this to completely replace the built-in word list (`CB_ANIMA_NEGATIVE` / `CB_NEGATIVE_PROMPT`). **Use with caution**: built-in quality words will be lost with it. In most cases you just "want to add a few words", so use the above `-extra`.

### Per-Request Negative Prompt

`POST /v1/images/generations` JSON body and `POST /v1/images/edits` multipart form both accept `negative_prompt` (also accepts shorthand `negative`):

```bash
curl -s http://127.0.0.1:8192/v1/images/generations ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"anima-base-v1.0.safetensors\",\"prompt\":\"a castle\",\"size\":\"1024x1024\",\"negative_prompt\":\"extra fingers\"}"
```

DiceFrame will not send this field, so this route is only useful when you call the bridge yourself with curl / other programs.

### ⚠️ Never Add Composition Words

**Do not** put `frame` / `borders` / `letterboxed` / `black bars` / `film strip` (and preferably avoid `out of frame`) into negative words.

Reason: DiceFrame automatically appends this suffix to positive prompts for **avatar** purposes:

```
Single character portrait, centered composition, clear face, no text, no frame.
```

Positive says `no frame`, negative says `frame`; the two conflict, and Anima shrinks the character to a small block in the center of the canvas with large pure-white areas around it (v1.2.0 broke this way; v1.3.0 removed these words from the built-in list and added a self-check guard). **Likewise, do not use negative words to suppress cinematic black bars** — just change the seed and regenerate.

### Does Adding Too Many Negative Words Hurt?

No. Controlled test under the same seed (avatar prompt, negative appended `extra fingers, mutated hands, off-model`):

| | Pure White Ratio | Non-White Bounding Box |
| --- | --- | --- |
| Built-in word list | 77.4% | 423×869 |
| Built-in + 3 appended words | 76.3% | 423×865 |

Almost identical. Composition differences between seeds (full body / half body) are model behavior, unrelated to negative words.

---

## 5. API Description

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Service information, current default parameters, whether translation is enabled |
| GET | `/health` | Bridge + ComfyUI health status, VRAM, queue, available base models |
| GET | `/v1/models` | Returns an OpenAI-style model list (contents = checkpoint files in ComfyUI) |
| POST | `/v1/images/generations` | **Text-to-image** (JSON) |
| POST | `/v1/images/edits` | **Image-to-image / reference image** (`multipart/form-data`) |
| GET | `/files/<name>.png` | Download generated image when `response_format=url` |

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

* `size` supports `1024x1024`, `1792x1024`, `1024x1792`, `512x768`…; you can also write `"square"` / `"landscape"` / `"portrait"`.
* Except for `model` / `prompt` / `size`, **all fields are optional**. If `model` is unrecognized, it automatically falls back to the only base model on this machine.
* `response_format` supports `b64_json` (default; DiceFrame uses this) and `url`.

Response:

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

`multipart/form-data`, file field name **`image`** (single) or **`image[]`** (multiple, only the first is used).
Other fields are the same as `/generations`. DiceFrame uses this endpoint when attaching character avatar reference images for the `scene` purpose.

Default behavior is **img2img**: uploads the first reference image to ComfyUI as the base image, `denoise` default `0.65`.
To ignore the reference image and generate only from the prompt, use `--edits-mode txt2img`.

### Error Format

```json
{ "error": { "message": "prompt cannot be empty", "type": "invalid_request_error",
             "code": null, "param": null } }
```

`type` values: `invalid_request_error` / `server_error` / `timeout_error`.
HTTP status codes: `400` invalid parameter / `404` unknown endpoint / `413` request body too large / `502` ComfyUI execution failed / `504` timeout.

---

## 6. Common Parameters

### SD1.5 Pipeline

| Parameter | Default | Description |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8190` | Listen address. If only for local DiceFrame, do not change (on this machine, because the port was occupied by a zombie process, it actually runs on **8192**, see Section 1) |
| `--comfy` | `http://127.0.0.1:8188` | ComfyUI address |
| `--engine` | `auto` | `auto` selects automatically based on `model`; `sd15` / `anima` force it |
| `--checkpoint` | empty (automatically takes the first) | Base model filename |
| `--steps` | `28` | Base sampling steps |
| `--cfg` | `7.0` | CFG; DreamShaper 8 recommends 7 |
| `--sampler` / `--scheduler` | `dpmpp_2m` / `karras` | Sampler / scheduler |
| `--base-pixels` | `393216` | Base output pixel budget (≈768×512). **For SD1.5 do not exceed 589824** |
| `--no-hires` | off | Add this to disable hires-fix second pass |
| `--hires-max` | `1280` | Maximum long side allowed by hires-fix |
| `--hires-steps` / `--hires-denoise` | `12` / `0.45` | hires-fix steps / denoise strength |

### Anima Pipeline

| Parameter | Default | Description |
| --- | --- | --- |
| `--anima-unet` | `anima-base-v1.0.safetensors` | Diffusion model under `models/diffusion_models` |
| `--anima-clip` / `--anima-clip-type` | `qwen_3_06b_base.safetensors` / `stable_diffusion` | Text encoder (copied from your local workflow) |
| `--anima-vae` | `qwen_image_vae.safetensors` | VAE |
| `--anima-steps` / `--anima-cfg` | `20` / `8.0` | Anima official recommended values |
| `--anima-sampler` / `--anima-scheduler` | `euler` / `simple` | Same as above |
| `--anima-negative` | Anime negative word list | **Completely replaces** the built-in list; use with caution (quality words will be lost too) |
| `--anima-negative-extra` | empty | **Appends** to the end of the built-in negative words; use this to "add a few words" |
| `--anima-base-max-long` | `1024` | Base resolution long-side limit; landscape/portrait are constrained by this, then raised by hires |
| `--anima-hires-max` / `--anima-hires-steps` | `1536` / `8` | Anima hires limit and steps |
| `--anima-no-hires` | off | Disable Anima hires |
| `--anima-lora` / `--anima-lora-strength` | empty / `0.9` | Optional LoRA (`LoraLoaderModelOnly`) |

### Common to Both

| Parameter | Default | Description |
| --- | --- | --- |
| `--edits-denoise` | `0.65` | `/edits` denoise strength; higher means more deviation from the reference image |
| `--edits-mode` | `img2img` | `/edits` behavior: `img2img` uses reference image, `txt2img` ignores it |
| `--max-size` | `2048` | Output size limit; larger sizes are proportionally scaled down |
| `--job-timeout` | `280` | Single-generation timeout in seconds (**must be less than DiceFrame's `imagegen_timeout_seconds`**) |
| `--positive-prefix` | empty | Fixed prompt added at the **very front** of every image generation prompt (joined by commas; not translated) |
| `--positive-suffix` | empty | Fixed prompt added at the **very end** of every image generation prompt (joined by commas; not translated) |
| `--negative-extra` | empty | Appends to the end of the **SD1.5** negative words (keeps built-in list) |
| `--negative-prompt` | SD1.5 word list | **Completely replaces** SD1.5 negative words; use with caution |
| `--translate-url` / `--translate-model` / `--translate-key` | empty / `deepseek-chat` / empty | Chinese prompt translation service |
| `--on-translate-failure` | `error` | On translation failure: `error` = return 502 error (default, avoids white images); `warn` = send Chinese into the model as before |
| `--url-store` | `./generated` | Directory for `response_format=url` (automatically cleaned after 1 hour) |
| `--dry-run` | off | Do not call ComfyUI; directly return a 1×1 placeholder image, for testing connectivity |
| `-v` / `--verbose` | off | Debug logging |

### Why Can 1792×1024 Still Produce Good Images?

**SD1.5** directly drawing 1792×1024 produces two heads / repeated composition; **Anima** directly drawing is slow and prone to repetition.
Both pipelines automatically do three-stage processing, just with different parameters:

```
SD1.5  target 1792x1024
  → scale down to base_pixels 393216 by aspect ratio → EmptyLatentImage 832x472
  → KSampler base sampling (28 steps)
  → LatentUpscale(bicubic) upscale to 1280x728
  → KSampler hires-fix (12 steps, denoise 0.45)
  → VAEDecode → ImageScale(lanczos) precisely scale to 1792x1024
  → SaveImage

Anima  target 1792x1024
  → calculate 1352x776 by aspect ratio, then bring long side back to 1024 → EmptySD3LatentImage 1024x584
  → KSampler base sampling (20 steps, euler/simple, cfg 8.0)
  → LatentUpscale(bicubic) upscale to 1536x880
  → KSampler hires-fix (8 steps, denoise 0.45)
  → VAEDecode → ImageScale(lanczos) precisely scale to 1792x1024
  → SaveImage
```

So any size can return the **exact** target resolution. Anima landscape tested at **80 seconds**, square at **36 seconds** (RTX 5060 Laptop).

---

## 7. Troubleshooting

| Symptom | Cause / Solution |
| --- | --- |
| DiceFrame image generation shows **Unavailable** | The three conditions are not all met: provider Base URL non-empty + format openai + `imagegen_model` non-empty |
| Log says `temporarily cannot connect to ComfyUI` | ComfyUI is not running or is not at `127.0.0.1:8188`; the bridge still starts, and recovers automatically after ComfyUI starts |
| `HTTP 504 … exceeded 280 seconds` | Image too large or GPU busy; increase `--job-timeout` **and** increase DiceFrame's `imagegen_timeout_seconds` even more.<br>**v1.4.1 fixed a related bug**: previously, as long as ComfyUI had written to history but the status had not yet become `completed`, the bridge would keep looping "not ready yet" and **never time out** (one test waited hard until **699 seconds** before returning 504, long after the client should have disconnected). Now the timeout check is moved to the top of the loop, so it will definitely return around `job-timeout` |
| DiceFrame reports "Image generation service returned HTTP 400" | The bridge returned a parameter error; check the specific message in the bridge window log |
| Image generation is very slow | The first request must load the base model; afterward Anima square is about 36 seconds, landscape about 50–80 seconds, SD1.5 about 14 seconds (RTX 5060 Laptop) |
| Image is **a small portrait in the exact center, surrounded by large pure-white areas** (especially avatars) | Versions v1.2.0 and earlier put `letterboxed, black bars, borders, frame, film strip` into Anima's negative words, conflicting with `no frame` in DiceFrame's avatar-purpose suffix. The model shrinks the character to a small block in the center of the canvas. **v1.3.0 removed these 5 words**; restart the bridge to recover. To confirm: `version` in `GET /` must be at least `1.3.0`. If you manually overrode with `--anima-negative`, also remove these words |
| Image has **black bars** on top and bottom | `Wide cinematic` in the prompt is occasionally interpreted as cinema aspect ratio. **Do not use negative words to suppress it** (see previous row; it causes worse white borders for avatars); change the seed and regenerate |
| Chinese prompt produces white image/garbled image | **First upgrade to the current version**: now it directly reports `502 translation_failed` instead of producing a white image. Seeing 502 means translation failed — the most common cause is an **invalid/wrong API Key** in DiceFrame's "Deepseek" provider (test with `curl https://api.deepseek.com/v1/chat/completions -H "Authorization: Bearer <key>"`); you can also change `CB_TRANSLATE_URL/MODEL/KEY` in `start-bridge.bat` to use another translation backend |
| DiceFrame's LLM features (chat/polish) also error | The same Key problem. Translation reuses it, so both break together |
| Want to confirm ComfyUI received the task | Check ComfyUI's `output\DiceFrame\DF_*.png` (`COMFY_OUTPUT_DIR` in `e2e_test.py` can directly list recent images) |
| Fixed prompt not taking effect | Check whether your words appear in the `revised_prompt` echoed by the API; on the bridge side, use `GET /`'s `fixed_prompt` block to confirm it is loaded. On DiceFrame side, click "Save" |
| Negative prompt not taking effect / want to confirm what was added | `negative` block in `GET /`: `anima_extra` is your appended words, `anima_full` is the final complete word list. After changing `start-bridge.bat`, you must restart the bridge |
| Port occupied / change port | `--port 8193`, and update the provider Base URL in DiceFrame accordingly (this machine is already using 8192) |

---

## 8. File List

| File | Description |
| --- | --- |
| `comfy_bridge.py` | **Main program**. The bridge service itself (dual-engine) |
| `start-bridge.bat` | Windows one-click startup. Automatically finds Python and DiceFrame directory; automatically restarts 5 seconds after a crash |
| `selftest.py` | Pure functions / size strategy / workflow structure / engine routing / fixed prompts / negative words / **wait() timeout regression** / multipart / image header self-check (currently **PASS=80**) |
| `test_translate.py` | Prompt translation pipeline test, with built-in local stub service (currently **PASS=29**) |
| `e2e_test.py` | Real HTTP end-to-end test (requires ComfyUI + bridge both running) |
| `wire_diceframe.py` | Automatically wire through DiceFrame's own `/api/config` (requires access password) |
| `requirements.txt` | Just one `requests` |
| `LICENSE` | MIT |
| `.gitignore` | Excludes `__pycache__/`, `generated/`, `_test_out/`, `_diceframe_backup/`, etc. |

### About `wire_diceframe.py`

DiceFrame has an access password set, and `POST /api/config` requires `Authorization: Bearer <access password>` (plus `X-TRPG-Confirm: true`).
The password is stored as a `pbkdf2_sha256` hash in `data/secrets.json` and **cannot be reverse-derived**, so you must either tell me or fill it in yourself.

```bat
python wire_diceframe.py --password YOUR_ACCESS_PASSWORD            :: dry run, backs up first + prints changes
python wire_diceframe.py --password YOUR_ACCESS_PASSWORD --apply    :: submit
```

The script only sends back `id/name/base_url/api_format/models/model_capabilities`, and **will not** overwrite the
`api_key` field that DiceFrame has already masked, so the keys of existing providers (DeepSeek) are safe.

If you do not want to use the script, fill it in manually in the DiceFrame UI according to Section 2; the effect is exactly the same.