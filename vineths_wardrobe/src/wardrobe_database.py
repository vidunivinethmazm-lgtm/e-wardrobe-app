"""
eWardrobeAI — SQLite Wardrobe Persistence Layer (Phase 10)

Replaces the in-memory RaveehaOrganisationalDB with a SQLite-backed store.
The database file is created automatically at first run.

Usage
-----
Pass db_path to RaveehaOrganisationalDB to persist state across restarts:

    from src.wardrobe_database import SQLiteWardrobeDB
    db = SQLiteWardrobeDB(db_path='wardrobe.db')

Or use in-memory SQLite for tests:
    db = SQLiteWardrobeDB(db_path=':memory:')
"""

from __future__ import annotations

import sqlite3
import os
import logging
from typing import Optional

from src.outfit_recommender import (
    GarmentRecord, CleaningStatus, Availability,
    GarmentCategory, Style, RaveehaOrganisationalDB,
)

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'wardrobe.db')

# ── Schema ─────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS garments (
    garment_id       TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    category         TEXT NOT NULL,
    style            TEXT NOT NULL,
    sizes            TEXT NOT NULL,
    colours          TEXT NOT NULL,
    body_types       TEXT NOT NULL,
    asset_path       TEXT NOT NULL,
    thumbnail_path   TEXT NOT NULL,
    cleaning_status  TEXT NOT NULL DEFAULT 'Clean',
    availability     TEXT NOT NULL DEFAULT 'owned',
    tags             TEXT NOT NULL DEFAULT ''
);
"""

_CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS status_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    garment_id   TEXT NOT NULL,
    old_status   TEXT NOT NULL,
    new_status   TEXT NOT NULL,
    changed_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class SQLiteWardrobeDB(RaveehaOrganisationalDB):
    """
    SQLite-backed wardrobe database.
    Inherits from RaveehaOrganisationalDB — the in-memory _records dict
    is loaded from SQLite on init and kept in sync on every write.

    Data Persistence
    ----------------
    - Garment catalogue is written to SQLite on first run (seed).
    - Subsequent runs load the persisted state (preserving any status changes).
    - Status history is logged to a separate `status_history` table.

    Thread Safety
    -------------
    Each call creates a new connection (check_same_thread=False).
    Suitable for single-worker FastAPI (uvicorn workers=1).
    For multi-worker, use a connection pool or PostgreSQL.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._init_db()

        # Call parent to populate in-memory _records (seed catalogue)
        super().__init__()

        # Sync: load persisted records from SQLite into memory
        self._load_from_db()
        logger.info(f"[SQLiteWardrobeDB] Loaded {len(self._records)} garments "
                    f"from {db_path}")

    # ── DB Initialisation ─────────────────────────────────────────────────────

    def _init_db(self):
        con = self._connect()
        con.execute(_CREATE_TABLE)
        con.execute(_CREATE_HISTORY_TABLE)
        con.commit()
        con.close()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    # ── Sync: memory → SQLite ─────────────────────────────────────────────────

    def _persist_record(self, rec: GarmentRecord):
        con = self._connect()
        con.execute("""
            INSERT OR REPLACE INTO garments
              (garment_id, name, category, style, sizes, colours,
               body_types, asset_path, thumbnail_path,
               cleaning_status, availability, tags)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            rec.garment_id, rec.name,
            rec.category.value, rec.style.value,
            ','.join(rec.sizes), ','.join(rec.colours),
            ','.join(rec.body_types),
            rec.asset_path, rec.thumbnail_path,
            rec.cleaning_status.value, rec.availability.value,
            ','.join(rec.tags),
        ))
        con.commit()
        con.close()

    def _seed_catalogue(self):
        """Override parent seed: write to SQLite only if table is empty."""
        super()._seed_catalogue()
        con = self._connect()
        count = con.execute("SELECT COUNT(*) FROM garments").fetchone()[0]
        con.close()
        if count == 0:
            for rec in self._records.values():
                self._persist_record(rec)
            logger.info("[SQLiteWardrobeDB] Catalogue seeded to SQLite.")

    # ── Sync: SQLite → memory ─────────────────────────────────────────────────

    def _load_from_db(self):
        """Load persisted garment states from SQLite into the in-memory dict."""
        con  = self._connect()
        rows = con.execute("SELECT * FROM garments").fetchall()
        con.close()

        if not rows:
            # DB empty — seed it now
            for rec in self._records.values():
                self._persist_record(rec)
            return

        for row in rows:
            gid = row['garment_id']
            if gid not in self._records:
                # Reconstruct GarmentRecord from SQLite row
                self._records[gid] = GarmentRecord(
                    garment_id     = row['garment_id'],
                    name           = row['name'],
                    category       = GarmentCategory(row['category']),
                    style          = Style(row['style']),
                    sizes          = row['sizes'].split(','),
                    colours        = row['colours'].split(','),
                    body_types     = row['body_types'].split(','),
                    asset_path     = row['asset_path'],
                    thumbnail_path = row['thumbnail_path'],
                    cleaning_status= CleaningStatus(row['cleaning_status']),
                    availability   = Availability(row['availability']),
                    tags           = row['tags'].split(',') if row['tags'] else [],
                )
            else:
                # Update existing in-memory record with persisted status
                self._records[gid].cleaning_status = CleaningStatus(row['cleaning_status'])
                self._records[gid].availability    = Availability(row['availability'])

    # ── Override update to persist ────────────────────────────────────────────

    def update_cleaning_status(self, garment_id: str,
                                status: CleaningStatus) -> bool:
        rec = self._records.get(garment_id)
        if rec is None:
            return False

        old_status = rec.cleaning_status.value
        result     = super().update_cleaning_status(garment_id, status)

        if result:
            con = self._connect()
            con.execute(
                "UPDATE garments SET cleaning_status=? WHERE garment_id=?",
                (status.value, garment_id)
            )
            con.execute(
                "INSERT INTO status_history (garment_id, old_status, new_status) "
                "VALUES (?,?,?)",
                (garment_id, old_status, status.value)
            )
            con.commit()
            con.close()
            logger.info(f"[SQLiteWardrobeDB] {garment_id}: "
                        f"{old_status} → {status.value} (persisted)")
        return result

    # ── History ───────────────────────────────────────────────────────────────

    def get_status_history(self, garment_id: Optional[str] = None,
                           limit: int = 50) -> list[dict]:
        """
        Retrieve cleaning status change history.
        Optionally filter by garment_id.
        """
        con = self._connect()
        if garment_id:
            rows = con.execute(
                "SELECT * FROM status_history WHERE garment_id=? "
                "ORDER BY changed_at DESC LIMIT ?",
                (garment_id, limit)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM status_history ORDER BY changed_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def get_wear_count(self, garment_id: str) -> int:
        """Number of times a garment has been marked as Dirty (worn)."""
        con = self._connect()
        count = con.execute(
            "SELECT COUNT(*) FROM status_history "
            "WHERE garment_id=? AND new_status='Dirty'",
            (garment_id,)
        ).fetchone()[0]
        con.close()
        return count

    def get_most_worn(self, top_n: int = 5) -> list[dict]:
        """Returns the top-N most worn garments by wear count."""
        con  = self._connect()
        rows = con.execute("""
            SELECT garment_id, COUNT(*) as wear_count
            FROM status_history
            WHERE new_status = 'Dirty'
            GROUP BY garment_id
            ORDER BY wear_count DESC
            LIMIT ?
        """, (top_n,)).fetchall()
        con.close()
        result = []
        for row in rows:
            rec = self._records.get(row['garment_id'])
            result.append({
                'garmentId':   row['garment_id'],
                'name':        rec.name if rec else 'Unknown',
                'wearCount':   row['wear_count'],
            })
        return result
