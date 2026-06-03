import json
import time
import uuid
from typing import Any, Dict, List, Optional

from .db import get_conn


def create_conversation(
    user_id: str,
    title: str,
    mode: str,
    source: Optional[str] = None
) -> Dict[str, Any]:
    now = int(time.time())
    conversation_id = str(uuid.uuid4())

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO conversations (id, user_id, title, mode, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (conversation_id, user_id, title, mode, source, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "id": conversation_id,
        "user_id": user_id,
        "title": title,
        "mode": mode,
        "source": source,
        "created_at": now,
        "updated_at": now,
    }


def list_conversations(user_id: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT * FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        """
        SELECT * FROM conversations
        WHERE id = ? AND user_id = ?
        """,
        (conversation_id, user_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_conversation(conversation_id: str, user_id: str) -> None:
    conn = get_conn()
    conn.execute(
        """
        DELETE FROM conversations
        WHERE id = ? AND user_id = ?
        """,
        (conversation_id, user_id),
    )
    conn.commit()
    conn.close()


def add_message(
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    research: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = int(time.time())
    message_id = str(uuid.uuid4())

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO messages (
            id, conversation_id, user_id, role, content,
            citations_json, research_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            conversation_id,
            user_id,
            role,
            content,
            json.dumps(citations or []),
            json.dumps(research or {}),
            now,
        ),
    )
    conn.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE id = ? AND user_id = ?
        """,
        (now, conversation_id, user_id),
    )
    conn.commit()
    conn.close()

    return {
        "id": message_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "citations": citations or [],
        "research": research or {},
        "created_at": now,
    }


def list_messages(conversation_id: str, user_id: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT * FROM messages
        WHERE conversation_id = ? AND user_id = ?
        ORDER BY created_at ASC
        """,
        (conversation_id, user_id),
    ).fetchall()
    conn.close()

    items = []
    for r in rows:
        item = dict(r)
        item["citations"] = json.loads(item.pop("citations_json") or "[]")
        item["research"] = json.loads(item.pop("research_json") or "{}")
        items.append(item)
    return items