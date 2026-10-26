"""SQLite storage for inspection history."""
import sqlite3
from datetime import datetime

DB_PATH = "inspections.db"


def get_conn(path: str = DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str = DB_PATH):
    with get_conn(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name  TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                verdict     TEXT NOT NULL,
                num_defects INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS defects (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER NOT NULL REFERENCES inspections(id),
                class_name    TEXT NOT NULL,
                confidence    REAL NOT NULL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                area_pct      REAL,
                severity      TEXT
            );
            """
        )


def save_inspection(image_name: str, verdict: str, defects: list, path: str = DB_PATH) -> int:
    """defects: list of dicts with class_name, confidence, box (x1,y1,x2,y2), area_pct, severity"""
    with get_conn(path) as conn:
        cur = conn.execute(
            "INSERT INTO inspections (image_name, created_at, verdict, num_defects) VALUES (?,?,?,?)",
            (image_name, datetime.now().isoformat(timespec="seconds"), verdict, len(defects)),
        )
        insp_id = cur.lastrowid
        for d in defects:
            x1, y1, x2, y2 = d["box"]
            conn.execute(
                "INSERT INTO defects (inspection_id, class_name, confidence, x1, y1, x2, y2, area_pct, severity) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (insp_id, d["class_name"], d["confidence"], x1, y1, x2, y2, d["area_pct"], d["severity"]),
            )
        return insp_id


def fetch_inspections(path: str = DB_PATH):
    with get_conn(path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM inspections ORDER BY id DESC")]


def fetch_defects(path: str = DB_PATH):
    with get_conn(path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM defects ORDER BY id DESC")]
