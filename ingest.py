import argparse, hashlib, json, re, shutil
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# ---------- config ----------
TREATMENT_MAP = {
    "Raw mechanical decorticate Hemp": "raw",
    "Partial Delignified Hemp": "partial_delignified",
    "Delignified Hemp": "delignified",
    "Degummed Hemp": "degummed",
    "Cottonized Hemp": "cottonized",
}
OFDA_DAY_RE = re.compile(r"^\s*Day\s*(\d+)", re.I)
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns
        .map(lambda c: str(c).strip())
        .map(lambda c: c.replace(" ", "_"))
        .map(lambda c: c.replace("%", "pct"))
        .map(lambda c: c.replace("(", "").replace(")", ""))
        .map(lambda c: c.replace("__", "_"))
        .map(lambda c: c.lower())
    )
    return out

def upsert_sample(engine, sample_id: str, treatment: str | None = None):
    with engine.begin() as cx:
        cx.execute(text("""
            INSERT INTO samples (sample_id, material_type, treatment)
            VALUES (:sid, 'hemp', :treat)
            ON CONFLICT (sample_id) DO UPDATE
            SET treatment = COALESCE(EXCLUDED.treatment, samples.treatment)
        """), {"sid": sample_id, "treat": treatment})

def log_ingest(engine, file_name: str, sha: str, sheet: str, nrows: int):
    with engine.begin() as cx:
        cx.execute(text("""
            INSERT INTO file_ingest_log (file_name, sha256, sheet_name, row_count)
            VALUES (:f, :h, :s, :n)
            ON CONFLICT (file_name, sha256, sheet_name) DO NOTHING
        """), {"f": file_name, "h": sha, "s": sheet, "n": int(nrows)})

def parse_ofda(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_cols(df_raw).rename(columns={
        "sampleid": "sample_id",
        "description": "description",
        "meand": "mean_d_um",
        "numd": "n_measurements",
        "temp_c": "temp_c",
        "rh%": "rh_pct",
        "rhpct": "rh_pct",
    })
    rows = []
    current_day = None
    for _, r in df.iterrows():
        sid = str(r.get("sample_id", "")).strip()
        if not sid:
            continue
        m = OFDA_DAY_RE.match(sid)
        if m:
            current_day = int(m.group(1))
            continue
        mean_d = pd.to_numeric(r.get("mean_d_um"), errors="coerce")
        n_meas = pd.to_numeric(r.get("n_measurements"), errors="coerce")
        if pd.isna(mean_d) or pd.isna(n_meas):
            continue
        sample_id = sid.split()[0]
        rows.append({
            "sample_id": sample_id,
            "run_label": sid,
            "run_day": current_day,
            "mode": "slide",
            "mean_d_um": float(mean_d),
            "n_measurements": int(n_meas),
            "temp_c": float(pd.to_numeric(r.get("temp_c"), errors="coerce")) if "temp_c" in df.columns else None,
            "rh_pct": float(pd.to_numeric(r.get("rh_pct"), errors="coerce")) if "rh_pct" in df.columns else None,
        })
    return pd.DataFrame(rows)

def load_ofda_file(engine, path: Path, sheet: str):
    df_raw = pd.read_excel(path, sheet_name=sheet)
    out = parse_ofda(df_raw)
    with engine.begin() as cx:
        for _, r in out.iterrows():
            upsert_sample(engine, r["sample_id"])
            cx.execute(text("""
                INSERT INTO ofda_runs
                (sample_id, run_label, run_day, mode, mean_d_um, n_measurements,
                temp_c, rh_pct, source_sheet)
                VALUES (:sample_id, :run_label, :run_day, :mode, :mean_d_um, :n_measurements,
                        :temp_c, :rh_pct, :source_sheet)
                ON CONFLICT (sample_id, run_label, source_sheet) DO UPDATE
                SET run_day        = EXCLUDED.run_day,
                    mode           = EXCLUDED.mode,
                    mean_d_um      = EXCLUDED.mean_d_um,
                    n_measurements = EXCLUDED.n_measurements,
                    temp_c         = EXCLUDED.temp_c,
                    rh_pct         = EXCLUDED.rh_pct
            """), {**r.to_dict(), "source_sheet": sheet})
    return len(out)

def maybe_load_cottonscope(engine, path: Path):
    try:
        df_raw = pd.read_excel(path, sheet_name="Cottonscope")
    except Exception:
        return 0
    if df_raw.empty:
        return 0
    df = clean_cols(df_raw).rename(columns={"sampleid": "sample_id", "wdum": "wd_um"})
    if not {"sample_id", "wd_um"}.issubset(df.columns):
        return 0
    rows = 0
    with engine.begin() as cx:
        for _, r in df.iterrows():
            sid = str(r["sample_id"]).strip()
            if not sid:
                continue
            wd = pd.to_numeric(r["wd_um"], errors="coerce")
            if pd.isna(wd):
                continue
            upsert_sample(engine, sid)
            cx.execute(text("""
                INSERT INTO cottonscope_runs (sample_id, wd_um, instrument_id, notes)
                VALUES (:sid, :wd, 'Cottonscope', NULL)
                ON CONFLICT DO NOTHING
            """), {"sid": sid, "wd": float(wd)})
            rows += 1
    return rows

def _num(x):
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else None

def load_favimat(engine, path: Path):
    xl = pd.ExcelFile(path)
    total = 0
    for sheet in xl.sheet_names:
        df_raw = xl.parse(sheet)
        if df_raw.empty:
            continue
        df = clean_cols(df_raw).rename(columns={
            "sample_": "sample_id",
            "reps": "replicate",
            "worktb": "work_tb",
            "lin_den_": "linear_density",
        })
        if not {"sample_id", "replicate"}.issubset(df.columns):
            continue
        treatment = TREATMENT_MAP.get(sheet, sheet.strip().lower().replace(" ", "_"))
        units = {"emax":"percent?","ebreak":"percent?","fmax":"cN or N?","work_tb":"cN*mm?","tenacity":"cN/tex?","linear_density":"tex?"}
        with engine.begin() as cx:
            for _, r in df.iterrows():
                sid = str(r["sample_id"]).strip()
                if not sid:
                    continue
                upsert_sample(engine, sid, treatment=treatment)
                payload = {
                    "sample_id": sid,
                    "treatment": treatment,
                    "replicate": int(pd.to_numeric(r["replicate"], errors="coerce")) if pd.notna(r["replicate"]) else None,
                    "emax": _num(r.get("emax")),
                    "ebreak": _num(r.get("ebreak")),
                    "fmax": _num(r.get("fmax")),
                    "work_tb": _num(r.get("work_tb")),
                    "tenacity": _num(r.get("tenacity")),
                    "linear_density": _num(r.get("linear_density")),
                    "units_json": json.dumps(units),
                }
                if payload["replicate"] is None:
                    continue
                cx.execute(text("""
                    INSERT INTO favimat_single_fiber
                    (sample_id, treatment, replicate, emax, ebreak, fmax, work_tb, tenacity, linear_density, units_json)
                    VALUES (:sample_id, :treatment, :replicate, :emax, :ebreak, :fmax, :work_tb, :tenacity, :linear_density, CAST(:units_json AS JSONB))
                    ON CONFLICT (sample_id, treatment, replicate) DO UPDATE
                    SET emax = EXCLUDED.emax,
                        ebreak = EXCLUDED.ebreak,
                        fmax = EXCLUDED.fmax,
                        work_tb = EXCLUDED.work_tb,
                        tenacity = EXCLUDED.tenacity,
                        linear_density = EXCLUDED.linear_density,
                        units_json = EXCLUDED.units_json
                """), payload)
                total += 1
    return total


def detect_workbook_type(path: Path) -> dict:
    xl = pd.ExcelFile(path)
    sheets = {sheet.strip().lower(): sheet for sheet in xl.sheet_names}
    detected = {
        "workbook_type": "unknown",
        "ofda_sheets": [],
        "has_cottonscope": False,
        "has_favimat": False,
    }

    for lowered, original in sheets.items():
        if "ofda" in lowered and "diameter" in lowered:
            detected["ofda_sheets"].append(original)
        if lowered == "cottonscope":
            detected["has_cottonscope"] = True

    favimat_hits = 0
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, nrows=5)
        cols = set(clean_cols(df).columns)
        if {"sample_id", "sample_", "replicate", "reps"} & cols and {"fmax", "worktb", "work_tb"} & cols:
            favimat_hits += 1

    detected["has_favimat"] = favimat_hits > 0

    if detected["has_favimat"]:
        detected["workbook_type"] = "favimat"
    elif detected["ofda_sheets"] and detected["has_cottonscope"]:
        detected["workbook_type"] = "ofda_cottonscope"
    elif detected["ofda_sheets"]:
        detected["workbook_type"] = "ofda"
    elif detected["has_cottonscope"]:
        detected["workbook_type"] = "cottonscope"

    return detected


def ingest_workbook(engine, path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    meta = detect_workbook_type(path)
    sha = file_sha(path)
    result = {
        "file_name": path.name,
        "sha256": sha,
        "workbook_type": meta["workbook_type"],
        "rows_loaded": {"ofda": 0, "cottonscope": 0, "favimat": 0},
        "sheets_loaded": [],
    }

    for sheet in meta["ofda_sheets"]:
        n = load_ofda_file(engine, path, sheet)
        log_ingest(engine, path.name, sha, sheet, n)
        result["rows_loaded"]["ofda"] += n
        result["sheets_loaded"].append(sheet)

    if meta["has_cottonscope"]:
        c = maybe_load_cottonscope(engine, path)
        log_ingest(engine, path.name, sha, "Cottonscope", c)
        result["rows_loaded"]["cottonscope"] += c
        result["sheets_loaded"].append("Cottonscope")

    if meta["has_favimat"]:
        n = load_favimat(engine, path)
        log_ingest(engine, path.name, sha, "ALL_SHEETS", n)
        result["rows_loaded"]["favimat"] += n
        result["sheets_loaded"].append("ALL_SHEETS")

    result["total_rows_loaded"] = sum(result["rows_loaded"].values())
    return result


def archive_uploaded_file(path: Path, target_dir: Path = PROCESSED_DIR) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / path.name
    if destination.exists():
        destination = target_dir / f"{path.stem}_{file_sha(path)[:8]}{path.suffix}"
    shutil.move(str(path), str(destination))
    return destination

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    engine = create_engine(args.dsn, future=True)
    data_dir = Path(args.path).expanduser().resolve()

    for path in sorted(data_dir.glob("*.xlsx")):
        result = ingest_workbook(engine, path)
        print(f"[{path.name}] type={result['workbook_type']} rows={result['total_rows_loaded']} detail={result['rows_loaded']}")

if __name__ == "__main__":
    from sqlalchemy import create_engine
    main()
