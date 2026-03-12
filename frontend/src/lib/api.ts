const BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export async function getLibrary() {
  const res = await fetch(`${BASE}/library`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load library");
  return res.json();
}

export async function uploadFile(file: File) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}

export async function indexFile(filename: string, collection: string, doc_type?: string) {
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