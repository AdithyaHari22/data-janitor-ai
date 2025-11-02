from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd 
import numpy as np 
import magic
from pathlib import Path
import uuid, tempfile, os, io
from helper import win_iqr_bounds,  safe_json, mk_pdf
import json

STORE = {} 
CLEAN = {}
REPORT_TXT = {}  
app = FastAPI()


@app.get("/health")
def health():
    '''
    Confirms if the connection is working.
    returns:
        OK - True
    '''
    return {"OK":True}


@app.post("/ingest")
def read_file(file: UploadFile = File(...)):
    '''
    Reads the file ingested for data cleaning. Detects the type of file.
    returns:
        JSON with file description
    '''
    try:
        head_bytes = file.file.read(4096)
        mime_type = magic.from_buffer(head_bytes, mime=True) or "application/octet-stream"
        file.file.seek(0)
        
        suffix = Path(file.filename or "").suffix.lower()
        
        if suffix == "":
            if "csv" in mime_type: suffix = ".csv"
            elif "json" in mime_type: suffix = ".json"
            elif "xml" in mime_type or "html" in mime_type: suffix = ".xml"
            elif "excel" in mime_type or "spreadsheet" in mime_type: suffix = ".xlsx"
            elif "parquet" in mime_type: suffix = ".parquet"
            else: suffix = ".bin"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            chunk = file.file.read(1024 * 1024)
            while chunk:
                tmp.write(chunk)
                chunk = file.file.read(1024 * 1024)
        
        fmt = None
        if ('text/csv' in mime_type) or (suffix == '.csv'):
            fmt = 'csv'
            df = pd.read_csv(temp_path)
        elif ('application/json' in mime_type) or ('x-ndjson' in mime_type) or ('jsonlines' in mime_type) or (suffix in ('.json', '.ndjson', '.jsonl')):
            fmt = 'json'
            if (suffix in ('.ndjson', '.jsonl')) or ('x-ndjson' in mime_type) or ('jsonlines' in mime_type):
                df = pd.read_json(temp_path, lines=True)
            else:
                try:
                    df = pd.read_json(temp_path)
                except ValueError:
                    df = pd.read_json(temp_path, lines=True)
        elif (('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in mime_type)
              or ('application/vnd.ms-excel' in mime_type)
              or (suffix in ('.xlsx', '.xls'))):
            fmt = 'excel'
            df = pd.read_excel(temp_path)
        elif ('text/xml' in mime_type) or ('application/xml' in mime_type) or (suffix in ('.xml', '.html')):
            fmt = 'xml'
            df = pd.read_xml(temp_path)
        elif (suffix == '.parquet') or ('parquet' in mime_type):
            fmt = 'parquet'
            df = pd.read_parquet(temp_path, engine='pyarrow')
        else:
            os.unlink(temp_path)
            raise ValueError(f"Unsupported file type: mime={mime_type}, suffix={suffix}")
        
        df_preview = df.head(5).replace({np.nan: None})
        ingest_id = str(uuid.uuid4())
        STORE[ingest_id] = df
        
        response =  {
            "source_format": fmt,
            "mime_type": mime_type,
            "filename": file.filename,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
            "head": df_preview.to_dict(orient="records"),
            "ingest_id": ingest_id
        }
        return jsonable_encoder(response)
    
    except Exception as e:
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

def retreive_df(ingest_id: str):
    df = STORE.get(ingest_id)
    if df is None:
        raise HTTPException(status_code=404, detail="ingest_id not found (server may have restarted)")
    return df
        
@app.post("/analyze")
def issues_report(ingest_id: str = Query(...)):
    '''
    load the dataframe and create a plan for the app to use in data cleaning.
    '''
    if ingest_id not in STORE:
        raise HTTPException(status_code=404, detail="ingest_id not found")
    df = STORE[ingest_id]

    report_cols = {}
    for c in df.columns:
        s = df[c]
        na = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        is_num = pd.api.types.is_numeric_dtype(s)
        suggested_cast = None
        outliers_iqr = None

        if not is_num:
            try:
                sn = pd.to_numeric(s, errors="coerce")
                if sn.notna().mean() >= 0.95:
                    suggested_cast = "numeric"
            except Exception:
                pass
        if suggested_cast is None and not pd.api.types.is_datetime64_any_dtype(s):
            try:
                sd = pd.to_datetime(s, errors="coerce", utc=True)
                if sd.notna().mean() >= 0.95:
                    suggested_cast = "datetime"
            except Exception:
                pass

        if is_num:
            bounds = win_iqr_bounds(s.dropna().astype(float))
            if bounds:
                lo, hi = bounds
                outliers_iqr = int(((s < lo) | (s > hi)).sum())
            else:
                outliers_iqr = 0

        report_cols[c] = {
            "na": na,
            "unique": unique,
            "is_numeric": bool(is_num),
            "suggested_cast": suggested_cast,
            "outliers_iqr": outliers_iqr
        }

    dupes = int(df.duplicated().sum())

    return safe_json({
        "ingest_id": ingest_id,
        "duplicates": dupes,
        "columns": report_cols
    })
    

@app.post("/dry_run")
def dry_run(
    ingest_id: str = Query(...),
    winsorize_numeric: bool = Query(True),
    fill_numeric: str = Query("median"),
    fill_text: str = Query("mode"),
    constant_fill_value: str = Query("Unknown"),
    drop_duplicates: bool = Query(True),
):
    if ingest_id not in STORE:
        raise HTTPException(status_code=404, detail="ingest_id not found")
    df0 = STORE[ingest_id]
    df = df0.copy()
    rows_before = len(df0)
    na_before = int(df0.isna().sum().sum())
    dtypes_before = {c: str(dt) for c, dt in df0.dtypes.items()}

    plan_cols = {}
    outliers_clipped = {}
    fills_applied = {}

    for c in df.columns:
        s = df[c]
        casted = False
        try:
            sn = pd.to_numeric(s, errors="coerce")
            if sn.notna().mean() >= 0.95:
                df[c] = sn
                plan_cols.setdefault(c, {})["cast"] = "numeric"
                casted = True
        except Exception:
            pass
        if not casted:
            try:
                sd = pd.to_datetime(s, errors="coerce", utc=True)
                if sd.notna().mean() >= 0.95:
                    df[c] = sd
                    plan_cols.setdefault(c, {})["cast"] = "datetime"
                    casted = True
            except Exception:
                pass

        na_col_before = int(df[c].isna().sum())
        if na_col_before > 0:
            if pd.api.types.is_numeric_dtype(df[c]):
                if fill_numeric == "median":
                    df[c] = df[c].fillna(df[c].median())
                elif fill_numeric == "mean":
                    df[c] = df[c].fillna(df[c].mean())
                if fill_numeric != "none":
                    fills_applied[c] = na_col_before
                    plan_cols.setdefault(c, {})["fillna"] = fill_numeric
            else:
                if fill_text == "mode":
                    mode_vals = df[c].mode(dropna=True)
                    fill_val = mode_vals.iloc[0] if len(mode_vals) else constant_fill_value
                    df[c] = df[c].fillna(fill_val)
                    fills_applied[c] = na_col_before
                    plan_cols.setdefault(c, {})["fillna"] = "mode"
                elif fill_text == "constant":
                    df[c] = df[c].fillna(constant_fill_value)
                    fills_applied[c] = na_col_before
                    plan_cols.setdefault(c, {})["fillna"] = "constant"
                
        if winsorize_numeric and pd.api.types.is_numeric_dtype(df[c]):
            bounds = win_iqr_bounds(df[c].dropna().astype(float))
            if bounds:
                lo, hi = bounds
                before_clip = int(((df[c] < lo) | (df[c] > hi)).sum())
                if before_clip > 0:
                    df[c] = df[c].clip(lower=lo, upper=hi)
                    outliers_clipped[c] = before_clip
                    plan_cols.setdefault(c, {})["winsorize"] = True

    duplicates_removed = 0
    if drop_duplicates:
        duplicates_removed = int(df.duplicated().sum())
        if duplicates_removed > 0:
            df = df.drop_duplicates()
    rows_after = len(df)
    na_after = int(df.isna().sum().sum())

    dtypes_after = {c: str(dt) for c, dt in df.dtypes.items()}
    dtype_changes = {
        c: {"from": dtypes_before[c], "to": dtypes_after[c]}
        for c in df.columns if c in dtypes_before and dtypes_before[c] != dtypes_after[c]
    }
    columns_changed = []
    try:
        for c in df.columns:
            b = df0[c]
            a = df[c]
            if len(b) != len(a):
                columns_changed.append(c)
                continue
            bs = b.astype("string")
            as_ = a.astype("string")
            diff_any = bool(((bs != as_) & ~(bs.isna() & as_.isna())).any())
            if diff_any:
                columns_changed.append(c)
    except Exception:
        pass

    plan = {
        "drop_duplicates": drop_duplicates,
        "columns": plan_cols
    }
    diff_summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "na_before_total": na_before,
        "na_after_total": na_after,
        "dtype_changes": dtype_changes,
        "columns_changed": columns_changed,
        "outliers_clipped": outliers_clipped,
        "fills_applied": fills_applied
    }

    return safe_json({
        "ingest_id": ingest_id,
        "plan": plan,
        "diff_summary": diff_summary
    })
    
    
@app.post("/clean")
def clean(
    ingest_id: str = Query(...),
    winsorize_numeric: bool = Query(True),
    fill_numeric: str = Query("median"),
    fill_text: str = Query("mode"),
    constant_fill_value: str = Query("Unknown"),
    drop_duplicates: bool = Query(True),
):
    dr = dry_run(
        ingest_id=ingest_id,
        winsorize_numeric=winsorize_numeric,
        fill_numeric=fill_numeric,
        fill_text=fill_text,
        constant_fill_value=constant_fill_value,
        drop_duplicates=drop_duplicates
    ).body
    dr_dict = json.loads(dr)

    df0 = STORE.get(ingest_id)
    if df0 is None:
        raise HTTPException(status_code=404, detail="ingest_id not found")
    df = df0.copy()
    plan_cols = dr_dict["plan"]["columns"]

    for c in df.columns:
        if c in plan_cols and plan_cols[c].get("cast") == "numeric":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif c in plan_cols and plan_cols[c].get("cast") == "datetime":
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

        if c in plan_cols and "fillna" in plan_cols[c]:
            how = plan_cols[c]["fillna"]
            if pd.api.types.is_numeric_dtype(df[c]):
                if how == "median":
                    df[c] = df[c].fillna(df[c].median())
                elif how == "mean":
                    df[c] = df[c].fillna(df[c].mean())
            else:
                if how == "mode":
                    mode_vals = df[c].mode(dropna=True)
                    fill_val = mode_vals.iloc[0] if len(mode_vals) else constant_fill_value
                    df[c] = df[c].fillna(fill_val)
                elif how == "constant":
                    df[c] = df[c].fillna(constant_fill_value)

        if c in plan_cols and plan_cols[c].get("winsorize"):
            bounds = win_iqr_bounds(df[c].dropna().astype(float))
            if bounds:
                lo, hi = bounds
                df[c] = df[c].clip(lower=lo, upper=hi)

    if dr_dict["plan"]["drop_duplicates"]:
        df = df.drop_duplicates()

    clean_id = ingest_id 
    CLEAN[clean_id] = df

    ds = dr_dict["diff_summary"]
    lines = []
    lines.append("Data Janitor — Cleaning Report")
    lines.append(f"clean_id: {clean_id}")
    lines.append("")
    lines.append("Summary:")
    lines.append(f"- Rows: {ds['rows_before']} -> {ds['rows_after']} (removed {ds['duplicates_removed']} duplicates)")
    lines.append(f"- Missing values: {ds['na_before_total']} -> {ds['na_after_total']}")
    if ds["dtype_changes"]:
        lines.append("- Dtype changes:")
        for k, v in ds["dtype_changes"].items():
            lines.append(f"  * {k}: {v['from']} -> {v['to']}")
    if ds["outliers_clipped"]:
        lines.append("- Outliers clipped:")
        for k, v in ds["outliers_clipped"].items():
            lines.append(f"  * {k}: {v}")
    if ds["fills_applied"]:
        lines.append("- Fills applied (NaNs resolved):")
        for k, v in ds["fills_applied"].items():
            lines.append(f"  * {k}: {v}")
    if ds["columns_changed"]:
        lines.append("- Columns changed:")
        lines.append("  " + ", ".join(ds["columns_changed"]))
    report_text = "\n".join(lines)
    REPORT_TXT[clean_id] = report_text

    return safe_json({
        "clean_id": clean_id,
        "message": "Cleaned dataset ready. Use /download and /report to retrieve artifacts.",
        "diff_summary": ds
    })
    

@app.get("/download/{clean_id}")
def download(clean_id: str, format: str = Query("csv")):
    df = CLEAN.get(clean_id)
    if df is None:
        raise HTTPException(status_code=404, detail="clean_id not found")
    format = format.lower().strip()
    if format not in ("csv", "parquet", "json"):
        raise HTTPException(status_code=400, detail="format must be csv|parquet|json")

    if format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        data = buf.getvalue().encode("utf-8")
        media = "text/csv"
        fn = f"cleaned_{clean_id}.csv"
    elif format == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        data = buf.getvalue()
        media = "application/octet-stream"
        fn = f"cleaned_{clean_id}.parquet"
    else:  # json
        data = df.to_json(orient="records").encode("utf-8")
        media = "application/json"
        fn = f"cleaned_{clean_id}.json"

    return StreamingResponse(io.BytesIO(data), media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{fn}"'
    })
    
@app.get("/report/{clean_id}")
def report(clean_id: str, format: str = Query("pdf")):
    text = REPORT_TXT.get(clean_id)
    if text is None:
        raise HTTPException(status_code=404, detail="No report found. Run /clean first.")
    fmt = format.lower().strip()
    if fmt == "txt":
        data = text.encode("utf-8")
        media = "text/plain"
        fn = f"report_{clean_id}.txt"
        return StreamingResponse(io.BytesIO(data), media_type=media, headers={
            "Content-Disposition": f'attachment; filename="{fn}"'
        })
    elif fmt == "pdf":
        pdf_bytes = mk_pdf(text)
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={
            "Content-Disposition": f'attachment; filename="report_{clean_id}.pdf"'
        })
    else:
        raise HTTPException(status_code=400, detail="format must be txt|pdf")

