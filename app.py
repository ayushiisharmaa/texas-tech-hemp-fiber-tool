import io
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text

from ingest import archive_uploaded_file, detect_workbook_type, ingest_workbook

DB_DSN = "postgresql+psycopg://hemp_user:hemp_pass@localhost:5433/hemp_db"
engine = create_engine(DB_DSN, future=True)

app = FastAPI(title="Hemp Fiber Data Portal")
app.mount("/web", StaticFiles(directory="web"), name="web")

TABLES = {
    "ofda": "ofda_runs",
    "favimat": "favimat_single_fiber",
    "cottonscope": "cottonscope_runs",
}

SORTABLE_COLUMNS = {
    "ofda": {"sample_id", "mean_d_um", "n_measurements", "temp_c", "rh_pct", "source_sheet"},
    "favimat": {"sample_id", "treatment", "replicate", "tenacity", "fmax", "linear_density", "emax", "ebreak", "work_tb"},
    "cottonscope": {"sample_id", "wd_um", "instrument_id"},
}

ANALYTIC_METRICS = {
    "ofda": ["mean_d_um", "n_measurements", "temp_c", "rh_pct"],
    "favimat": ["tenacity", "fmax", "linear_density", "emax", "ebreak", "work_tb"],
    "cottonscope": ["wd_um"],
}


@app.get("/", response_class=HTMLResponse)
def home():
    return Path("web/index.html").read_text()


def round_value(value, digits: int = 2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def build_query(
    table_key: str,
    sample_id: str | None,
    treatment: str | None,
    source_sheet: str | None,
    sort_by: str | None,
    sort_dir: str,
    limit: str,
    offset: int,
):
    if table_key not in TABLES:
        raise ValueError("invalid table")

    table_name = TABLES[table_key]
    sql = f"SELECT * FROM {table_name}"
    count_sql = f"SELECT COUNT(*) FROM {table_name}"
    where_clauses = []
    params = {}

    if sample_id:
        where_clauses.append("sample_id ILIKE :sample_id")
        params["sample_id"] = f"%{sample_id}%"

    if treatment and table_key == "favimat":
        where_clauses.append("treatment = :treatment")
        params["treatment"] = treatment

    if source_sheet and table_key == "ofda":
        where_clauses.append("source_sheet = :source_sheet")
        params["source_sheet"] = source_sheet

    if where_clauses:
        where_sql = " WHERE " + " AND ".join(where_clauses)
        sql += where_sql
        count_sql += where_sql

    if sort_by and sort_by in SORTABLE_COLUMNS.get(table_key, set()):
        direction = "DESC" if sort_dir.lower() == "desc" else "ASC"
        sql += f" ORDER BY {sort_by} {direction}"

    if limit != "all":
        sql += " LIMIT :limit OFFSET :offset"
        params["limit"] = int(limit)
        params["offset"] = offset

    return sql, count_sql, params


def get_table_df(table: str, sample_id: str | None = None, treatment: str | None = None, source_sheet: str | None = None):
    sql, _, params = build_query(table, sample_id, treatment, source_sheet, None, "asc", "all", 0)
    return pd.read_sql(text(sql), engine, params=params)


def summarize_series(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std_dev": None,
            "variance": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "cv_pct": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    mean = clean.mean()
    std_dev = clean.std(ddof=1) if len(clean) > 1 else 0.0
    se = (std_dev / (len(clean) ** 0.5)) if len(clean) > 1 else 0.0
    margin = 1.96 * se
    variance = clean.var(ddof=1) if len(clean) > 1 else 0.0

    return {
        "count": int(clean.count()),
        "mean": round_value(mean),
        "median": round_value(clean.median()),
        "std_dev": round_value(std_dev),
        "variance": round_value(variance),
        "min": round_value(clean.min()),
        "max": round_value(clean.max()),
        "q1": round_value(clean.quantile(0.25)),
        "q3": round_value(clean.quantile(0.75)),
        "iqr": round_value(clean.quantile(0.75) - clean.quantile(0.25)),
        "cv_pct": round_value((std_dev / mean) * 100) if mean else None,
        "ci95_low": round_value(mean - margin),
        "ci95_high": round_value(mean + margin),
    }


def build_metric_cards(df: pd.DataFrame, table: str) -> list[dict]:
    metric = ANALYTIC_METRICS[table][0]
    stats = summarize_series(df[metric]) if metric in df.columns else summarize_series(pd.Series(dtype=float))
    total_rows = int(len(df.index))
    unique_samples = int(df["sample_id"].nunique()) if "sample_id" in df.columns else 0
    return [
        {"label": "Rows", "value": total_rows},
        {"label": "Unique Samples", "value": unique_samples},
        {"label": f"Mean {metric}", "value": stats["mean"]},
        {"label": f"Std Dev {metric}", "value": stats["std_dev"]},
    ]


def build_top_chart(df: pd.DataFrame, table: str) -> dict:
    if df.empty:
        return {"label": "No data", "description": "No rows are available for this selection.", "x": [], "y": []}

    if table == "favimat":
        grouped = (
            df.groupby("treatment", dropna=True)[["tenacity", "fmax"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("tenacity" if df["tenacity"].notna().any() else "fmax", ascending=False)
        )
        metric = "tenacity" if grouped.get("tenacity", pd.Series(dtype=float)).notna().any() else "fmax"
        return {
            "label": f"Average {metric} by treatment",
            "description": "Mechanical performance grouped by FAVIMAT treatment class.",
            "x": grouped["treatment"].fillna("Unknown").head(10).tolist(),
            "y": grouped[metric].fillna(0).head(10).round(2).tolist(),
        }

    metric = "mean_d_um" if table == "ofda" else "wd_um"
    grouped = (
        df.groupby("sample_id", dropna=True)[metric]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    return {
        "label": f"Top samples by {metric}",
        "description": "Highest average values in the current selection.",
        "x": grouped["sample_id"].tolist(),
        "y": grouped[metric].round(2).tolist(),
    }


def build_filters() -> dict:
    with engine.begin() as conn:
        treatments = pd.read_sql(
            text("SELECT DISTINCT treatment FROM favimat_single_fiber WHERE treatment IS NOT NULL ORDER BY treatment"),
            conn,
        )["treatment"].tolist()
        source_sheets = pd.read_sql(
            text("SELECT DISTINCT source_sheet FROM ofda_runs WHERE source_sheet IS NOT NULL ORDER BY source_sheet"),
            conn,
        )["source_sheet"].tolist()
    return {"treatments": treatments, "source_sheets": source_sheets}


def build_analytics(df: pd.DataFrame, table: str) -> dict:
    metrics = []
    for metric in ANALYTIC_METRICS[table]:
        if metric in df.columns:
            metrics.append({"metric": metric, **summarize_series(df[metric])})

    if table == "favimat" and not df.empty:
        breakdown = (
            df.groupby("treatment", dropna=False)[["tenacity", "fmax", "linear_density"]]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        breakdown.columns = ["_".join([part for part in col if part]).strip("_") for col in breakdown.columns.to_flat_index()]
        breakdown = breakdown.fillna("")
        groups = breakdown.to_dict(orient="records")
    else:
        metric = "mean_d_um" if table == "ofda" else "wd_um"
        if df.empty or "sample_id" not in df.columns or metric not in df.columns:
            groups = []
        else:
            grouped = (
                df.groupby("sample_id")[metric]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
                .sort_values("mean", ascending=False)
                .head(20)
            )
            grouped = grouped.round(2).fillna("")
            groups = grouped.to_dict(orient="records")

    primary_metric = ANALYTIC_METRICS[table][0]
    numeric_primary = pd.to_numeric(df[primary_metric], errors="coerce").dropna() if primary_metric in df.columns else pd.Series(dtype=float)
    q1 = numeric_primary.quantile(0.25) if not numeric_primary.empty else None
    q3 = numeric_primary.quantile(0.75) if not numeric_primary.empty else None
    iqr = (q3 - q1) if q1 is not None and q3 is not None else None

    if iqr is not None:
        outlier_mask = (numeric_primary < (q1 - 1.5 * iqr)) | (numeric_primary > (q3 + 1.5 * iqr))
        outlier_count = int(outlier_mask.sum())
    else:
        outlier_count = 0

    completeness = []
    total_rows = int(len(df.index))
    for metric in ANALYTIC_METRICS[table]:
        if metric in df.columns:
            missing = int(df[metric].isna().sum())
            completeness.append(
                {
                    "metric": metric,
                    "missing": missing,
                    "missing_pct": round_value((missing / total_rows) * 100 if total_rows else 0),
                }
            )

    missing_total = int(sum(item["missing"] for item in completeness))
    flags = []
    if total_rows == 0:
        flags.append("No rows match the current filter selection.")
    if completeness and any(item["missing_pct"] and item["missing_pct"] >= 10 for item in completeness):
        flags.append("Some key metrics have more than 10% missing values.")
    if total_rows and outlier_count:
        flags.append(f"{outlier_count} records fall outside the expected {primary_metric} range.")
    if metrics and metrics[0]["cv_pct"] and metrics[0]["cv_pct"] > 20:
        flags.append(f"{primary_metric} shows elevated variability, so sample-level review is recommended.")

    health = {
        "rows": total_rows,
        "unique_samples": int(df["sample_id"].nunique()) if "sample_id" in df.columns else 0,
        "missing_total": missing_total,
        "outlier_count": outlier_count,
        "primary_metric": primary_metric,
    }

    return {"metrics": metrics, "groups": groups, "health": health, "completeness": completeness, "flags": flags}


def get_ingest_snapshot() -> dict:
    snapshot = {}
    with engine.begin() as conn:
        for key, table in TABLES.items():
            row = conn.execute(
                text(f"SELECT COUNT(*) AS rows, COUNT(DISTINCT sample_id) AS samples FROM {table}")
            ).mappings().one()
            snapshot[key] = {"rows": int(row["rows"]), "samples": int(row["samples"] or 0)}
    return snapshot


@app.get("/api/query")
def query_data(
    table: str = Query(...),
    sample_id: str | None = None,
    treatment: str | None = None,
    source_sheet: str | None = None,
    sort_by: str | None = None,
    sort_dir: str = "asc",
    limit: str = "25",
    offset: int = 0,
):
    if table not in TABLES:
        return {"error": "invalid table"}

    sql, count_sql, params = build_query(table, sample_id, treatment, source_sheet, sort_by, sort_dir, limit, offset)
    df = pd.read_sql(text(sql), engine, params=params)
    count_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}

    with engine.begin() as conn:
        total = conn.execute(text(count_sql), count_params).scalar()

    return {"rows": df.to_dict(orient="records"), "total": total, "limit": limit, "offset": offset}


@app.get("/api/download")
def download_csv(
    table: str = Query(...),
    sample_id: str | None = None,
    treatment: str | None = None,
    source_sheet: str | None = None,
    sort_by: str | None = None,
    sort_dir: str = "asc",
    limit: str = "all",
    offset: int = 0,
):
    if table not in TABLES:
        return {"error": "invalid table"}

    sql, _, params = build_query(table, sample_id, treatment, source_sheet, sort_by, sort_dir, limit, offset)
    df = pd.read_sql(text(sql), engine, params=params)

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={table}_query.csv"},
    )


@app.get("/api/summary")
def summary_data(table: str = Query(...)):
    if table not in TABLES:
        return {"error": "invalid table"}

    df = get_table_df(table)
    return {
        "cards": build_metric_cards(df, table),
        "chart": build_top_chart(df, table),
        "filters": build_filters(),
    }


@app.get("/api/analytics")
def analytics_data(
    table: str = Query(...),
    sample_id: str | None = None,
    treatment: str | None = None,
    source_sheet: str | None = None,
):
    if table not in TABLES:
        return {"error": "invalid table"}

    df = get_table_df(table, sample_id=sample_id, treatment=treatment, source_sheet=source_sheet)
    return build_analytics(df, table)


@app.get("/api/compare/ofda-cottonscope")
def compare_ofda_cottonscope(sample_id: str | None = None):
    ofda_filters = []
    cottonscope_filters = []
    params = {}
    if sample_id:
        params["sample_id"] = f"%{sample_id}%"
        ofda_filters.append("REGEXP_REPLACE(sample_id, '_(\\d{2})$', '') ILIKE :sample_id")
        cottonscope_filters.append("sample_id ILIKE :sample_id")

    ofda_where_sql = f"WHERE {' AND '.join(ofda_filters)}" if ofda_filters else ""
    cottonscope_where_sql = f"WHERE {' AND '.join(cottonscope_filters)}" if cottonscope_filters else ""
    compare_sql = f"""
        WITH ofda AS (
            SELECT
                REGEXP_REPLACE(sample_id, '_(\\d{{2}})$', '') AS normalized_sample_id,
                AVG(mean_d_um) AS ofda_mean,
                STDDEV_SAMP(mean_d_um) AS ofda_std,
                COUNT(*) AS ofda_n
            FROM ofda_runs
            {ofda_where_sql}
            GROUP BY normalized_sample_id
        ),
        cottonscope AS (
            SELECT
                sample_id AS normalized_sample_id,
                AVG(wd_um) AS cottonscope_mean,
                STDDEV_SAMP(wd_um) AS cottonscope_std,
                COUNT(*) AS cottonscope_n
            FROM cottonscope_runs
            {cottonscope_where_sql}
            GROUP BY normalized_sample_id
        )
        SELECT
            ofda.normalized_sample_id AS sample_id,
            ROUND(ofda.ofda_mean::numeric, 2) AS ofda_mean,
            ROUND(cottonscope.cottonscope_mean::numeric, 2) AS cottonscope_mean,
            ROUND(ofda.ofda_std::numeric, 2) AS ofda_std,
            ROUND(cottonscope.cottonscope_std::numeric, 2) AS cottonscope_std,
            ofda.ofda_n,
            cottonscope.cottonscope_n
        FROM ofda
        INNER JOIN cottonscope ON cottonscope.normalized_sample_id = ofda.normalized_sample_id
        ORDER BY ofda.normalized_sample_id
    """
    compare_df = pd.read_sql(text(compare_sql), engine, params=params)

    if compare_df.empty:
        return {
            "summary": [],
            "regression": {},
            "rows": [],
        }

    compare_df["ofda_mean"] = pd.to_numeric(compare_df["ofda_mean"], errors="coerce")
    compare_df["cottonscope_mean"] = pd.to_numeric(compare_df["cottonscope_mean"], errors="coerce")
    compare_df["ofda_std"] = pd.to_numeric(compare_df["ofda_std"], errors="coerce")
    compare_df["cottonscope_std"] = pd.to_numeric(compare_df["cottonscope_std"], errors="coerce")
    compare_df["ofda_n"] = pd.to_numeric(compare_df["ofda_n"], errors="coerce").fillna(0).astype(int)
    compare_df["cottonscope_n"] = pd.to_numeric(compare_df["cottonscope_n"], errors="coerce").fillna(0).astype(int)

    valid_pairs = compare_df.dropna(subset=["ofda_mean", "cottonscope_mean"]).copy()
    slope = None
    intercept = None
    correlation = None
    r_squared = None

    if len(valid_pairs.index) > 1:
        x = valid_pairs["ofda_mean"]
        y = valid_pairs["cottonscope_mean"]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        x_variance = float((x_centered**2).sum())
        if x_variance > 0:
            slope = float((x_centered * y_centered).sum() / x_variance)
            intercept = float(y.mean() - slope * x.mean())
        correlation = x.corr(y)
        if pd.notna(correlation):
            correlation = float(correlation)
            r_squared = float(correlation**2)

    if slope is not None and intercept is not None:
        compare_df["predicted_cottonscope"] = compare_df["ofda_mean"] * slope + intercept
        compare_df["fit_residual"] = compare_df["cottonscope_mean"] - compare_df["predicted_cottonscope"]
        compare_df["abs_fit_residual"] = compare_df["fit_residual"].abs()
        compare_df["fit_position"] = compare_df["fit_residual"].apply(
            lambda value: "On fit line"
            if abs(float(value)) < 0.01
            else ("Above fit line" if float(value) > 0 else "Below fit line")
        )
    else:
        compare_df["predicted_cottonscope"] = None
        compare_df["fit_residual"] = None
        compare_df["abs_fit_residual"] = None
        compare_df["fit_position"] = "Fit unavailable"

    compare_df["support_total"] = compare_df["ofda_n"] + compare_df["cottonscope_n"]

    for column in ("predicted_cottonscope", "fit_residual", "abs_fit_residual"):
        if column in compare_df.columns:
            compare_df[column] = compare_df[column].apply(lambda value: round_value(value) if pd.notna(value) else None)

    summary = [
        {"label": "Matched Samples", "value": int(compare_df["sample_id"].nunique())},
        {"label": "Avg OFDA Mean", "value": round_value(compare_df["ofda_mean"].mean())},
        {"label": "Avg Cottonscope Mean", "value": round_value(compare_df["cottonscope_mean"].mean())},
        {"label": "R^2", "value": round_value(r_squared, 3)},
    ]

    return {
        "summary": summary,
        "regression": {
            "pair_count": int(len(valid_pairs.index)),
            "slope": round_value(slope, 4),
            "intercept": round_value(intercept, 4),
            "correlation": round_value(correlation, 4),
            "r_squared": round_value(r_squared, 4),
        },
        "rows": compare_df.fillna("").to_dict(orient="records"),
    }


@app.get("/api/uploads/history")
def upload_history():
    try:
        with engine.begin() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT file_name, sha256, sheet_name, row_count, ingested_at
                    FROM file_ingest_log
                    ORDER BY ingested_at DESC
                    LIMIT 25
                    """
                ),
                conn,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read upload history: {exc}") from exc

    return {"rows": df.fillna("").to_dict(orient="records")}


@app.post("/api/upload")
async def upload_workbook(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only Excel workbooks are supported.")

    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(delete=False, suffix=suffix, dir=upload_dir) as tmp:
        payload = await file.read()
        tmp.write(payload)
        temp_path = Path(tmp.name)

    final_path = upload_dir / Path(file.filename).name
    if final_path.exists():
        final_path = upload_dir / f"{temp_path.stem}_{Path(file.filename).name}"
    temp_path.replace(final_path)

    try:
        before = get_ingest_snapshot()
        preview = detect_workbook_type(final_path)
        result = ingest_workbook(engine, final_path)
        archived_path = archive_uploaded_file(final_path)
        after = get_ingest_snapshot()
    except Exception as exc:
        if final_path.exists():
            final_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    impact = {}
    detected_instruments = []
    for instrument in TABLES:
        rows_added = after[instrument]["rows"] - before[instrument]["rows"]
        samples_added = after[instrument]["samples"] - before[instrument]["samples"]
        impact[instrument] = {
            "rows_added": rows_added,
            "samples_added": samples_added,
        }
        if rows_added > 0:
            detected_instruments.append(instrument)

    return JSONResponse(
        {
            "message": "Workbook ingested successfully.",
            "preview": preview,
            "result": result,
            "impact": impact,
            "detected_instruments": detected_instruments,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "archived_to": str(archived_path),
        }
    )
