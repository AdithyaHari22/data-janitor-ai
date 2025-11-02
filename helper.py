from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pandas as pd 
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def win_iqr_bounds(s: pd.Series):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return None
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

def safe_json(payload: dict):
    return JSONResponse(content=jsonable_encoder(payload))

def mk_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x, y = 40, height - 40
    for line in text.splitlines():
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(x, y, line[:115])  # simple wrap
        y -= 14
    c.save()
    buf.seek(0)
    return buf.read()

