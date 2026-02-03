# SQLite telemetry logger
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "telemetry" / "telemetry.db"
SCHEMA_PATH = ROOT / "telemetry" / "schema.sql"

def get_conn():
    # sqlite connection
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    try:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema) # run CREATE TABLE statements
        conn.commit()
    finally:
        conn.close()


def log_run(input_mode: str, claim: str, topk: int, evidence: list, judge_obj: dict, flags: list[str]):
    # Insert one pipeline run into sqlite
    init_db()  # ensure DB exists and tables created
    created_at = datetime.now(timezone.utc).isoformat()

    evidence_json = json.dumps(evidence, ensure_ascii=False)
    judge_json = json.dumps(judge_obj, ensure_ascii=False)
    flags_json = json.dumps(flags, ensure_ascii=False)

    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO runs (created_at, input_mode, claim, topk, evidence_json, judge_json, flags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at, input_mode, claim, topk, evidence_json, judge_json, flags_json),
        )
        conn.commit()
    finally:
        conn.close()
