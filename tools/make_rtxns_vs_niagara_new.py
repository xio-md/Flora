"""Compose RTXNS current render next to newly captured Niagara GT."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"D:\RTXNS\output\bistro_test")
LEFT = OUT / "bistro_rt_shadow_steady.png"       # RTXNS current
RIGHT = OUT / "GT2.png"                          # Niagara manual capture
RESULT = OUT / "rtxns_vs_niagara_new.png"

W, H = 1024, 768

def _resize_keep_aspect(img, w, h, fill=(0, 0, 0)):
    iw, ih = img.size
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), fill)
    canvas.paste(img, ((w - nw) // 2, (h - nh) // 2))
    return canvas

left = _resize_keep_aspect(Image.open(LEFT).convert("RGB"), W, H)
right = _resize_keep_aspect(Image.open(RIGHT).convert("RGB"), W, H)

PAD = 20
LABEL_H = 40
total_w = W * 2 + PAD * 3
total_h = H + LABEL_H + PAD * 2

canvas = Image.new("RGB", (total_w, total_h), (16, 16, 16))
d = ImageDraw.Draw(canvas)

try:
    f = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)
except OSError:
    f = ImageFont.load_default()

d.text((PAD + 8, PAD // 2), "RTXNS (1-sample + bilateral blur)",
       font=f, fill=(255, 255, 255))
d.text((PAD * 2 + W + 8, PAD // 2), "Niagara GT (ray traced + blur)",
       font=f, fill=(255, 255, 255))

canvas.paste(left,  (PAD, LABEL_H))
canvas.paste(right, (PAD * 2 + W, LABEL_H))

canvas.save(RESULT)
print(f"wrote {RESULT} ({canvas.size[0]}x{canvas.size[1]})")
