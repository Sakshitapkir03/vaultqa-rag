# from pathlib import Path
# import sqlite3

# DB_PATH = Path("data/app.db")

# def get_conn():
#     DB_PATH.parent.mkdir(parents=True, exist_ok=True)
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     conn.execute("PRAGMA foreign_keys = ON;")
#     return conn

# def init_db():
#     conn = get_conn()
#     cur = conn.cursor()

#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS conversations (
#         id TEXT PRIMARY KEY,
#         title TEXT NOT NULL,
#         mode TEXT NOT NULL,
#         source TEXT,
#         created_at INTEGER NOT NULL,
#         updated_at INTEGER NOT NULL
#     )
#     """)

#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS messages (
#         id TEXT PRIMARY KEY,
#         conversation_id TEXT NOT NULL,
#         role TEXT NOT NULL,
#         content TEXT NOT NULL,
#         citations_json TEXT,
#         research_json TEXT,
#         created_at INTEGER NOT NULL,
#         FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
#     )
#     """)

#     conn.commit()
#     conn.close()

from pathlib import Path
import sqlite3

DB_PATH = Path("data/app.db")


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        mode TEXT NOT NULL,
        source TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        citations_json TEXT,
        research_json TEXT,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        collection TEXT,
        doc_type TEXT,
        indexed_chunks INTEGER NOT NULL,
        indexed_at INTEGER NOT NULL
    )
    """)

    conn.commit()
    conn.close()