import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_path      TEXT NOT NULL,
    json_path       TEXT NOT NULL UNIQUE,
    duration        REAL NOT NULL,
    sample_rate     INTEGER NOT NULL,
    num_speeches    INTEGER NOT NULL,
    num_alignments  INTEGER NOT NULL,
    mtime           REAL NOT NULL,
    indexed_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS alignments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    speech_idx      INTEGER NOT NULL,
    alignment_idx   INTEGER NOT NULL,
    text            TEXT NOT NULL,
    start_time      REAL NOT NULL,
    end_time        REAL NOT NULL,
    duration        REAL,
    score           REAL
);

CREATE VIRTUAL TABLE IF NOT EXISTS alignments_fts USING fts5(
    text,
    content='alignments',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 0'
);

CREATE TRIGGER IF NOT EXISTS alignments_ai AFTER INSERT ON alignments BEGIN
    INSERT INTO alignments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS alignments_ad AFTER DELETE ON alignments BEGIN
    INSERT INTO alignments_fts(alignments_fts, rowid, text)
        VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS alignments_au AFTER UPDATE ON alignments BEGIN
    INSERT INTO alignments_fts(alignments_fts, rowid, text)
        VALUES('delete', old.id, old.text);
    INSERT INTO alignments_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def search_alignments(
    conn: sqlite3.Connection,
    query: str,
    page: int = 1,
    per_page: int = 20,
    snippets_per_doc: int = 5,
):
    """Search alignments using FTS5, returning results grouped by document."""
    offset = (page - 1) * per_page

    # Count total matching documents
    count_sql = """
        SELECT COUNT(DISTINCT a.document_id)
        FROM alignments_fts fts
        JOIN alignments a ON a.id = fts.rowid
        WHERE alignments_fts MATCH ?
    """
    total_docs = conn.execute(count_sql, (query,)).fetchone()[0]

    # Get matching document IDs (paginated)
    doc_ids_sql = """
        SELECT a.document_id, MIN(rank) as best_rank
        FROM alignments_fts fts
        JOIN alignments a ON a.id = fts.rowid
        WHERE alignments_fts MATCH ?
        GROUP BY a.document_id
        ORDER BY best_rank
        LIMIT ? OFFSET ?
    """
    doc_rows = conn.execute(doc_ids_sql, (query, per_page, offset)).fetchall()
    doc_ids = [r["document_id"] for r in doc_rows]

    if not doc_ids:
        return [], total_docs

    # Fetch document metadata
    placeholders = ",".join("?" * len(doc_ids))
    docs_sql = f"SELECT * FROM documents WHERE id IN ({placeholders})"
    docs = {r["id"]: dict(r) for r in conn.execute(docs_sql, doc_ids).fetchall()}

    # Fetch top matching snippets per document
    results = []
    for doc_id in doc_ids:
        doc = docs[doc_id]
        snippets_sql = """
            SELECT a.id, a.speech_idx, a.alignment_idx, a.start_time, a.end_time,
                   a.score,
                   snippet(alignments_fts, 0, '<mark>', '</mark>', '...', 48) as snippet_text
            FROM alignments_fts fts
            JOIN alignments a ON a.id = fts.rowid
            WHERE alignments_fts MATCH ? AND a.document_id = ?
            ORDER BY rank
            LIMIT ?
        """
        snippets = [
            dict(r)
            for r in conn.execute(snippets_sql, (query, doc_id, snippets_per_doc)).fetchall()
        ]
        doc["snippets"] = snippets
        results.append(doc)

    return results, total_docs


def get_all_documents(conn: sqlite3.Connection, page: int = 1, per_page: int = 20):
    """List all indexed documents, paginated."""
    offset = (page - 1) * per_page
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY indexed_at DESC LIMIT ? OFFSET ?",
        (per_page, offset),
    ).fetchall()
    return [dict(r) for r in rows], total


def get_document(conn: sqlite3.Connection, document_id: int):
    """Get a single document by ID."""
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row) if row else None


def get_document_alignments(conn: sqlite3.Connection, document_id: int):
    """Get full alignment data for a document, suitable for the transcript player."""
    row = conn.execute("SELECT json_path FROM documents WHERE id = ?", (document_id,)).fetchone()
    if not row:
        return None
    return row["json_path"]
