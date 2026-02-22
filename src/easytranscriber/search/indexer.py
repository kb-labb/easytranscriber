import logging
import sqlite3
from pathlib import Path

import msgspec

from easytranscriber.data.datamodel import AudioMetadata

logger = logging.getLogger(__name__)


def index_file(conn: sqlite3.Connection, json_path: Path) -> bool:
    """Index a single alignment JSON file. Returns True if the file was (re)indexed."""
    mtime = json_path.stat().st_mtime

    # Check if already indexed with same mtime
    existing = conn.execute(
        "SELECT id, mtime FROM documents WHERE json_path = ?", (str(json_path),)
    ).fetchone()

    if existing and existing["mtime"] == mtime:
        return False

    # Remove stale entry if mtime changed
    if existing:
        conn.execute("DELETE FROM documents WHERE id = ?", (existing["id"],))

    # Parse JSON using the project's own data model
    raw = json_path.read_bytes()
    metadata = msgspec.json.decode(raw, type=AudioMetadata)

    num_speeches = len(metadata.speeches) if metadata.speeches else 0
    num_alignments = 0
    if metadata.speeches:
        for speech in metadata.speeches:
            num_alignments += len(speech.alignments)

    # Insert document
    cur = conn.execute(
        """INSERT INTO documents (audio_path, json_path, duration, sample_rate,
                                  num_speeches, num_alignments, mtime)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            metadata.audio_path,
            str(json_path),
            metadata.duration,
            metadata.sample_rate,
            num_speeches,
            num_alignments,
            mtime,
        ),
    )
    doc_id = cur.lastrowid

    # Insert alignments
    if metadata.speeches:
        rows = []
        for speech_idx, speech in enumerate(metadata.speeches):
            for align_idx, alignment in enumerate(speech.alignments):
                rows.append(
                    (
                        doc_id,
                        speech_idx,
                        align_idx,
                        alignment.text,
                        alignment.start,
                        alignment.end,
                        alignment.duration,
                        alignment.score,
                    )
                )
        conn.executemany(
            """INSERT INTO alignments
               (document_id, speech_idx, alignment_idx, text, start_time, end_time, duration, score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    return True


def index_directory(
    alignments_dir: Path, conn: sqlite3.Connection, force: bool = False
) -> tuple[int, int]:
    """
    Index all JSON files in the alignments directory.

    Returns (indexed_count, skipped_count).
    """
    if force:
        conn.execute("DELETE FROM alignments")
        conn.execute("DELETE FROM documents")
        # Rebuild FTS index
        conn.execute("INSERT INTO alignments_fts(alignments_fts) VALUES('rebuild')")
        conn.commit()

    json_files = sorted(alignments_dir.glob("*.json"))
    total_files = len(json_files)
    if not json_files:
        logger.warning("No JSON files found in %s", alignments_dir)
        return 0, 0

    indexed = 0
    skipped = 0
    for file_num, json_path in enumerate(json_files, 1):
        try:
            was_indexed = index_file(conn, json_path)
            if was_indexed:
                indexed += 1
            else:
                skipped += 1
            status = "indexed" if was_indexed else "skipped (unchanged)"
            logger.info("[%d/%d] %s — %s", file_num, total_files, json_path.name, status)
        except Exception:
            logger.exception("[%d/%d] Failed to index %s", file_num, total_files, json_path)

    # Remove documents whose JSON files no longer exist
    existing_paths = {str(p) for p in json_files}
    all_db_paths = conn.execute("SELECT id, json_path FROM documents").fetchall()
    stale_ids = [r["id"] for r in all_db_paths if r["json_path"] not in existing_paths]
    if stale_ids:
        placeholders = ",".join("?" * len(stale_ids))
        conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", stale_ids)
        logger.info("Removed %d stale documents from index", len(stale_ids))

    conn.commit()
    return indexed, skipped
