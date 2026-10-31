import os
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
_load_env_file(PROJECT_DIR / ".env")

DOCUMENTS_DIR = PROJECT_DIR / "documents"
CHROMA_DIR = PROJECT_DIR / ".chroma_db"


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

COLLECTION_NAME = "rag_documents"
