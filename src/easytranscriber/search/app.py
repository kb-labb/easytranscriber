import math
import mimetypes
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from easytranscriber.search.db import (
    get_all_documents,
    get_document,
    get_document_alignments,
    search_alignments,
)

SEARCH_DIR = Path(__file__).parent


def format_duration(seconds: float) -> str:
    """Format seconds as h:mm:ss or m:ss."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_timestamp(seconds: float) -> str:
    """Format seconds as h:mm:ss or m:ss."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def create_app(
    db_conn: sqlite3.Connection,
    audio_dir: Path,
    per_page: int = 20,
    snippets_per_doc: int = 5,
) -> FastAPI:
    app = FastAPI(title="easysearch")

    templates = Jinja2Templates(directory=str(SEARCH_DIR / "templates"))
    templates.env.filters["duration"] = format_duration
    templates.env.filters["timestamp"] = format_timestamp

    app.mount("/static", StaticFiles(directory=str(SEARCH_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def search_page(
        request: Request,
        q: str = Query(default="", description="Search query"),
        page: int = Query(default=1, ge=1),
    ):
        if q.strip():
            try:
                results, total = search_alignments(
                    db_conn, q.strip(), page, per_page, snippets_per_doc
                )
            except Exception:
                # FTS5 query syntax errors fall through as empty results
                results, total = [], 0
        else:
            results = []
            total = db_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

        total_pages = max(1, math.ceil(total / per_page))

        return templates.TemplateResponse(
            "search.html",
            {
                "request": request,
                "query": q,
                "results": results,
                "total": total,
                "page": page,
                "total_pages": total_pages,
                "has_query": bool(q.strip()),
            },
        )

    @app.get("/documents", response_class=HTMLResponse)
    async def documents_page(
        request: Request,
        page: int = Query(default=1, ge=1),
    ):
        results, total = get_all_documents(db_conn, page, per_page)
        total_pages = max(1, math.ceil(total / per_page))

        return templates.TemplateResponse(
            "documents.html",
            {
                "request": request,
                "results": results,
                "total": total,
                "page": page,
                "total_pages": total_pages,
            },
        )

    @app.get("/document/{document_id}", response_class=HTMLResponse)
    async def document_page(
        request: Request,
        document_id: int,
        t: float = Query(default=0, description="Seek to time"),
        q: str = Query(default="", description="Highlight query"),
    ):
        doc = get_document(db_conn, document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return templates.TemplateResponse(
            "document.html",
            {
                "request": request,
                "document": doc,
                "seek_time": t,
                "query": q,
            },
        )

    @app.get("/api/document/{document_id}")
    async def document_json(document_id: int):
        json_path_str = get_document_alignments(db_conn, document_id)
        if not json_path_str:
            raise HTTPException(status_code=404, detail="Document not found")

        json_path = Path(json_path_str)
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="JSON file not found on disk")

        # Return the raw JSON file directly
        return FileResponse(json_path, media_type="application/json")

    @app.get("/audio/{filename:path}")
    async def serve_audio(filename: str):
        file_path = audio_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")

        media_type, _ = mimetypes.guess_type(str(file_path))
        if media_type is None:
            media_type = "application/octet-stream"

        return FileResponse(file_path, media_type=media_type)

    return app
