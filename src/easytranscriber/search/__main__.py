"""
Launch the easysearch app.

Usage:
    easysearch [options]

Examples:
    easysearch --alignments-dir output/alignments --audio-dir data/audio
    easysearch --reindex
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("easytranscriber.search")


def main():
    parser = argparse.ArgumentParser(
        description="Launch the easysearch app.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alignments-dir",
        type=Path,
        default=Path("output/alignments"),
        help="Directory containing alignment JSON files.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing source audio files.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("search.db"),
        help="Path to the SQLite database file.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=8642, help="Port to listen on.")
    parser.add_argument("--per-page", type=int, default=20, help="Results per page.")
    parser.add_argument(
        "--snippets-per-doc",
        type=int,
        default=5,
        help="Max matching snippets shown per document in search results.",
    )
    parser.add_argument(
        "--reindex", action="store_true", help="Force full re-index of all JSON files."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.alignments_dir.is_dir():
        parser.error(f"Alignments directory not found: {args.alignments_dir}")
    if not args.audio_dir.is_dir():
        parser.error(f"Audio directory not found: {args.audio_dir}")

    # Import here to avoid import errors if optional deps are missing
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        logger.error(
            "Missing dependencies. Install with: pip install easytranscriber[search]"
        )
        raise SystemExit(1)

    from easytranscriber.search.db import init_db
    from easytranscriber.search.indexer import index_directory

    # Initialize database and index
    conn = init_db(args.db)

    logger.info("Indexing %s ...", args.alignments_dir)
    indexed, skipped = index_directory(args.alignments_dir, conn, force=args.reindex)
    logger.info("Indexed %d file(s), skipped %d unchanged.", indexed, skipped)

    # Create and run the app
    from easytranscriber.search.app import create_app

    app = create_app(
        db_conn=conn,
        audio_dir=args.audio_dir,
        per_page=args.per_page,
        snippets_per_doc=args.snippets_per_doc,
    )

    logger.info("Starting server at http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
