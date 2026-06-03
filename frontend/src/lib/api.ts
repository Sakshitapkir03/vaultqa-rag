import { createClient } from "@/lib/supabase/client";
const BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
async function authHeaders(json = true) {
  const supabase = createClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const headers: Record<string, string> = {};

  if (json) {
    headers["Content-Type"] = "application/json";
  }

  if (session?.access_token) {
    headers["Authorization"] = `Bearer ${session.access_token}`;
  }

  return headers;
}
export async function getLibrary() {
  const res = await fetch(`${BASE}/library`, {
    cache: "no-store",
    headers: await authHeaders(false),
  });

  if (!res.ok) throw new Error("Failed to load library");
  return res.json();
}

export async function uploadFile(file: File) {
  const supabase = createClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    headers: session?.access_token
      ? { Authorization: `Bearer ${session.access_token}` }
      : {},
    body: form,
  });

  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}

export async function indexFile(
  filename: string,
  collection: string,
  doc_type?: string
) {
  const res = await fetch(`${BASE}/index`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename,
      collection,
      doc_type: doc_type || null,
    }),
  });

  if (!res.ok) throw new Error("Index failed");
  return res.json();
}

export async function ask(
  question: string,
  scope: { collection?: string; doc_type?: string; source?: string }
) {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      ...scope,
    }),
  });

  if (!res.ok) throw new Error("Ask failed");
  return res.json();
}

export async function askStream(
  question: string,
  scope: { source?: string },
  onChunk: (chunk: string) => void
) {
  const res = await fetch(`${BASE}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      ...scope,
    }),
  });

  if (!res.ok) throw new Error("Streaming ask failed");
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    onChunk(chunk);
  }
}

export async function deepResearch(question: string, source?: string) {
  const res = await fetch(`${BASE}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, source }),
  });

  if (!res.ok) throw new Error("Deep research failed");
  return res.json();
}

export async function deepResearchStream(
  question: string,
  source: string | undefined,
  onEvent: (event: any) => void
) {
  const res = await fetch(`${BASE}/research/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, source }),
  });

  if (!res.ok) throw new Error("Deep research stream failed");
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const parsed = JSON.parse(trimmed);
        onEvent(parsed);
      } catch {
        // ignore malformed partial line
      }
    }
  }

  if (buffer.trim()) {
    try {
      onEvent(JSON.parse(buffer.trim()));
    } catch {
      // ignore trailing malformed chunk
    }
  }
}

export async function getConversations() {
  const res = await fetch(`${BASE}/conversations`, {
    cache: "no-store",
    headers: await authHeaders(false),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to load conversations: ${res.status} ${text}`);
  }

  return res.json();
}

export async function createConversation(payload: {
  title: string;
  mode: string;
  source?: string;
}) {
  const res = await fetch(`${BASE}/conversations`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify(payload),
  });

  if (!res.ok) throw new Error("Failed to create conversation");
  return res.json();
}

export async function getConversation(conversationId: string) {
  const res = await fetch(`${BASE}/conversations/${conversationId}`, {
    cache: "no-store",
    headers: await authHeaders(false),
  });

  if (!res.ok) throw new Error("Failed to load conversation");
  return res.json();
}

export async function deleteConversation(conversationId: string) {
  const res = await fetch(`${BASE}/conversations/${conversationId}`, {
    method: "DELETE",
    headers: await authHeaders(false),
  });

  if (!res.ok) throw new Error("Failed to delete conversation");
  return res.json();
}

export async function addMessageToConversation(
  conversationId: string,
  payload: {
    role: string;
    content: string;
    citations?: any[];
    research?: Record<string, any>;
  }
) {
  const res = await fetch(`${BASE}/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: await authHeaders(),
    body: JSON.stringify(payload),
  });

  if (!res.ok) throw new Error("Failed to save message");
  return res.json();
}

export async function deleteDocument(filename: string) {
  const res = await fetch(`${BASE}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    headers: await authHeaders(false),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to delete document: ${res.status} ${text}`);
  }

  return res.json();
}