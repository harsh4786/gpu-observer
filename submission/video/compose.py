"""Compose the captioned walkthrough from captured stills: letterboxed content,
caption band, animated pointer on real clicks, crossfades. Writes frames + an
ffmpeg concat list; static holds are a single frame with a duration."""
import json, math, pathlib, sys
from PIL import Image, ImageDraw, ImageFont

V = pathlib.Path(sys.argv[1]); RAW = V / "raw"; FR = V / "frames"
W, H, FPS = 1920, 1080, 30
CW, CH = 1653, 930                      # 16:9 content area
OX, OY = (W - CW) // 2, 0               # content offset
S = CW / 1920                           # capture px -> content px
BG = (7, 16, 15)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(FONT, 34) if pathlib.Path(FONT).exists() else ImageFont.load_default()

def base(name):
    im = Image.open(RAW / name).convert("RGB")
    if im.size != (1920, 1080):         # viewer frames are 1920x1078: pad, never rescale
        pad = Image.new("RGB", (1920, 1080), BG); pad.paste(im, (0, 0)); im = pad
    return im.resize((CW, CH), Image.LANCZOS)

def wrap(text, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.getlength(t) <= maxw: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]

def canvas(content, caption):
    im = Image.new("RGB", (W, H), BG); im.paste(content, (OX, OY))
    d = ImageDraw.Draw(im)
    d.rectangle([0, CH, W, H], fill=(10, 22, 20)); d.rectangle([OX, CH + 22, OX + 6, H - 22], fill=(114, 227, 177))
    lines = wrap(caption, CW - 60)[:2]
    y = CH + (H - CH - len(lines) * 46) // 2
    for ln in lines: d.text((OX + 30, y), ln, font=font, fill=(230, 242, 236)); y += 46
    return im

def pointer(im, x, y, ripple=0.0):
    d = ImageDraw.Draw(im, "RGBA")
    px, py = OX + x * S, OY + y * S
    if ripple > 0:
        r = 14 + 34 * ripple; a = int(220 * (1 - ripple))
        d.ellipse([px - r, py - r, px + r, py + r], outline=(114, 227, 177, a), width=4)
    pts = [(0, 0), (0, 34), (9, 26), (16, 41), (22, 38), (15, 24), (27, 24)]
    poly = [(px + a, py + b) for a, b in pts]
    d.polygon([(p[0] + 2, p[1] + 2) for p in poly], fill=(0, 0, 0, 120))
    d.polygon(poly, fill=(255, 255, 255, 255), outline=(10, 22, 20, 255))
    return im

meta = json.loads((RAW / "rects.json").read_text())["rects"]
scenes = [
  ("slide-1.png", 6,  "GPU Observer: see exactly what one LLM request does on the GPU.", None),
  ("slide-2.png", 9,  "Between request in and tokens out, an inference server is a black box. Requests are batched and reordered, and profilers show kernels but not whose work they are.", None),
  ("slide-3.png", 10, "GPU Observer keeps one request's identity from the prompt to its tokens, the scheduler step, the GPU rows it was packed into, the CUDA kernels that ran, and the KV-cache blocks it owns.", None),
  ("slide-4.png", 10, "Live on an NVIDIA DGX Spark: type a prompt and watch it become tokens, engine steps, packed rows and the kernel stages of each transformer layer, lit by real CUPTI launches.", None),
  ("slide-5.png", 9,  "Three things break naive attribution: rows get reordered, CUDA Graphs hide kernel launches, and CUPTI and Compute Sanitizer can't share a process.", None),
  ("viewer-landing.png", 6, "No GPU needed: open the hosted viewer and load a real capture recorded on the DGX Spark.",
     {"from": (960, 620), "to": (meta["loadSample"]["x"], meta["loadSample"]["y"]), "click": True}),
  ("viewer-multilane.png", 9, "Step 5: our focused request's 94-token prefill shares the GPU with seven decoding requests, each followed from its scheduler slice to its packed rows.",
     {"from": (1600, 300), "to": (meta["select_multilane"]["x"], 54), "click": True}),
  ("viewer-reordered.png", 9, "Step 67: the scheduler's order and the GPU's row order disagree for six requests. The crossing edges show it, which is why row order can't be assumed.", None),
  ("viewer-gpu-before-click.png", 5, "Click the GPU node to see whose work this kernel did.",
     {"from": (900, 250), "to": (meta["gpuNode"]["x"], meta["gpuNode"]["y"]), "click": True}),
  ("viewer-ownership.png", 11, "The kernel's real GPU interval, and the cache blocks owned by each request, reconstructed with the row-to-block rule validated against device events.", None),
  ("viewer-evidence.png", 6, "Every value says how it was obtained: measured, reconstructed or unavailable.", None),
  ("slide-7.png", 10, "Result: per-request KV-cache block ownership recovered from 7,560 device events under CUDA Graph replay, with zero mismatches, in single-stream decode.", None),
  ("slide-8.png", 9,  "The same requests joined to real CUPTI GPU intervals: 1,400 activities, zero correlation misses, 64 ns calibration uncertainty.", None),
  ("slide-9.png", 11, "Across 15 fresh-server runs, no instrumentation setting, including capturing every kernel launch, was distinguishable from unmodified vLLM within half a percent.", None),
  ("slide-11.png", 9, "Exact ownership is proven for one kernel family today, and the live kernel view is tuned to Qwen3-14B. Next: any model, more kernels, multi-stream batches.", None),
  ("slide-12.png", 8, "Try it at harsh4786.github.io/gpu-observer, and read the code at github.com/harsh4786/gpu-observer.", None),
]

entries, n = [], 0
def emit(im, dur):
    global n
    p = FR / f"f{n:05d}.png"; im.save(p, compress_level=1); entries.append((p, dur)); n += 1
ease = lambda t: 0.5 - 0.5 * math.cos(math.pi * t)
FADE = 12
prev_last = None
for img, hold, cap, ptr in scenes:
    frame0 = canvas(base(img), cap)
    first = frame0.copy()
    if ptr: first = pointer(first, *ptr["from"])
    if prev_last is not None:
        for i in range(1, FADE + 1): emit(Image.blend(prev_last, first, i / (FADE + 1)), 1 / FPS)
    used = FADE / FPS if prev_last is not None else 0
    last = first
    if ptr:
        emit(first, 0.4); used += 0.4
        (x0, y0), (x1, y1) = ptr["from"], ptr["to"]
        for i in range(1, 25):
            t = ease(i / 24); emit(pointer(frame0.copy(), x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), 1 / FPS)
        used += 24 / FPS
        if ptr.get("click"):
            for i in range(1, 11): emit(pointer(frame0.copy(), x1, y1, ripple=i / 10), 1 / FPS)
            used += 10 / FPS
        last = pointer(frame0.copy(), x1, y1)
    emit(last, max(0.5, hold - used))
    prev_last = last
emit(prev_last, 1.0)
with open(V / "list.txt", "w") as f:
    for p, d in entries: f.write(f"file '{p}'\nduration {d:.5f}\n")
    f.write(f"file '{entries[-1][0]}'\n")
print(f"frames={n} planned_duration={sum(d for _, d in entries):.1f}s")
