#!/usr/bin/env python3
"""Monta docs/apresentacao.html a partir do template, embutindo as figuras (base64) e o QR code do site."""
import base64, io, re
from pathlib import Path
from PIL import Image
import segno

DOCS = Path(__file__).resolve().parent
SITE = "https://deep-learning-terremoto.web.app"

def png_b64(path: Path, max_w: int = 1400) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

qr = segno.make(SITE, error="m")
buf = io.BytesIO(); qr.save(buf, kind="png", scale=6, border=1, dark="#0E1A2B", light="#ffffff")
qr_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

html = (DOCS / "apresentacao_template.html").read_text(encoding="utf-8")
imgs = {"fig_calibration": png_b64(DOCS / "fig" / "fig_calibration.png"),
        "fig_rbm_weights": png_b64(DOCS / "fig" / "fig_rbm_weights.png", 1000),
        "fig_hopfield_prototypes": png_b64(DOCS / "fig" / "fig_hopfield_prototypes.png"),
        "banda_semana": png_b64(DOCS / "fig" / "banda_semana.png", 1300),
        "qr": qr_b64}
for k, v in imgs.items():
    html = html.replace("{{IMG:%s}}" % k, v)
missing = re.findall(r"\{\{IMG:[^}]+\}\}", html)
assert not missing, missing
(DOCS / "apresentacao.html").write_text(html, encoding="utf-8")
print("apresentacao.html:", round(len(html.encode()) / 1024), "KB")
