import logging
import sqlite3
from pathlib import Path

import msgspec

from easytranscriber.data.datamodel import AudioMetadata

logger = logging.getLogger(__name__)


def _detect_index_mode(metadata: "AudioMetadata") -> str:
    """Detect whether to index by chunks or alignments.

    Checks if the first chunk of the first speech has text. If not, the file
    was produced by easyaligner (ground-truth alignment) and should be indexed
    by alignments. ASR pipeline output always populates chunk text.
    """
    if metadata.speeches:
        first_speech = metadata.speeches[0]
        if first_speech.chunks and first_speech.chunks[0].text is None:
            return "alignments"
    return "chunks"


def index_file(conn: sqlite3.Connection, json_path: Path, index_mode: str | None = None) -> bool:
    """Index a single alignment JSON file. Returns True if the file was (re)indexed.

    Parameters
    ----------
    index_mode : str or None
        ``"chunks"`` indexes VAD chunks produced by ASR pipelines.
        ``"alignments"`` indexes sentence-level AlignmentSegments, as produced by
        ``easyaligner`` when ground-truth text is aligned to audio (chunks have no text).
        If ``None`` (default), the mode is detected automatically from the file contents.
    """
    mtime = json_path.stat().st_mtime

    # Parse JSON using the project's own data model (needed for auto-detection)
    raw = json_path.read_bytes()
    metadata = msgspec.json.decode(raw, type=AudioMetadata)

    resolved_mode = index_mode if index_mode is not None else _detect_index_mode(metadata)

    # Check if already indexed with same mtime and same mode
    existing = conn.execute(
        "SELECT id, mtime, index_mode FROM documents WHERE json_path = ?", (str(json_path),)
    ).fetchone()

    if existing and existing["mtime"] == mtime and existing["index_mode"] == resolved_mode:
        return False

    # Remove stale entry if mtime or mode changed
    if existing:
        conn.execute("DELETE FROM documents WHERE id = ?", (existing["id"],))

    num_speeches = len(metadata.speeches) if metadata.speeches else 0
    num_segments = 0
    if metadata.speeches:
        for speech in metadata.speeches:
            if resolved_mode == "alignments":
                num_segments += len(speech.alignments)
            else:
                num_segments += len(speech.chunks)

    # Insert document
    cur = conn.execute(
        """INSERT INTO documents (audio_path, json_path, duration, sample_rate,
                                  num_speeches, num_chunks, index_mode, mtime)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            metadata.audio_path,
            str(json_path),
            metadata.duration,
            metadata.sample_rate,
            num_speeches,
            num_segments,
            resolved_mode,
            mtime,
        ),
    )
    doc_id = cur.lastrowid

    if metadata.speeches:
        rows = []
        if resolved_mode == "alignments":
            for speech_idx, speech in enumerate(metadata.speeches):
                for seg_idx, seg in enumerate(speech.alignments):
                    rows.append(
                        (
                            doc_id,
                            speech_idx,
                            seg_idx,
                            seg.text,
                            seg.start,
                            seg.end,
                            seg.duration,
                        )
                    )
        else:
            for speech_idx, speech in enumerate(metadata.speeches):
                for chunk_idx, chunk in enumerate(speech.chunks):
                    if not chunk.text:
                        continue
                    rows.append(
                        (
                            doc_id,
                            speech_idx,
                            chunk_idx,
                            chunk.text,
                            chunk.start,
                            chunk.end,
                            chunk.duration,
                        )
                    )
        conn.executemany(
            """INSERT INTO chunks
               (document_id, speech_idx, chunk_idx, text, start_time, end_time, duration)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    return True


def index_directory(
    alignments_dir: Path,
    conn: sqlite3.Connection,
    force: bool = False,
    index_mode: str | None = None,
) -> tuple[int, int]:
    """
    Index all JSON files in the alignments directory.

    Parameters
    ----------
    index_mode : str or None
        ``"chunks"``, ``"alignments"``, or ``None`` to auto-detect per file.
        See :func:`index_file`.

    Returns (indexed_count, skipped_count).
    """
    if force:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        # Rebuild FTS index
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
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
            was_indexed = index_file(conn, json_path, index_mode=index_mode)
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
