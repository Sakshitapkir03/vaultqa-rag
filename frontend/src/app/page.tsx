"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  askStream,
  deepResearchStream,
  getLibrary,
  indexFile,
  uploadFile,
} from "@/lib/api";

type IndexedDoc = {
  filename: string;
  collection: string;
  doc_type: string;
  indexed_chunks: number;
  indexed_at: number;
};

type Citation = {
  source: string;
  page: number;
  chunk_id: string;
  score: number;
  quote: string;
};

type ChatMsg =
  | { role: "user"; text: string }
  | {
      role: "assistant";
      text: string;
      grounded?: boolean;
      confidence?: number;
      contradiction?: boolean;
      citations?: Citation[];
    };

type ResearchStep = {
  step: string;
  status: string;
};

type ResearchFinding = {
  sub_question: string;
  answer: string;
  citations: Citation[];
};

type ResearchResult = {
  intent?: { intent: string; confidence: number };
  plan?: { main_question: string; intent: string; sub_questions: string[] };
  steps?: ResearchStep[];
  findings?: ResearchFinding[];
  report?: string;
  citations?: Citation[];
  verified?: { verified: boolean; citation_count: number };
  contradiction?: boolean;
};

type Mode = "ask" | "research";

const starterPrompts = [
  "Summarize this document",
  "List the key takeaways",
  "Explain this simply",
  "Create study questions",
];

function timeAgo(ts: number) {
  const now = Math.floor(Date.now() / 1000);
  const diff = Math.max(0, now - ts);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1.5">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="h-2 w-2 rounded-full bg-zinc-400"
          animate={{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }}
          transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.12 }}
        />
      ))}
      <span className="ml-2 text-sm text-zinc-400">Thinking…</span>
    </div>
  );
}

function ModeToggle({
  mode,
  setMode,
}: {
  mode: Mode;
  setMode: (m: Mode) => void;
}) {
  return (
    <div className="inline-flex rounded-2xl border border-white/10 bg-white/[0.05] p-1">
      <button
        onClick={() => setMode("ask")}
        className={`rounded-xl px-4 py-2 text-sm transition ${
          mode === "ask"
            ? "bg-white text-zinc-900"
            : "text-zinc-300 hover:bg-white/[0.06]"
        }`}
      >
        Ask
      </button>
      <button
        onClick={() => setMode("research")}
        className={`rounded-xl px-4 py-2 text-sm transition ${
          mode === "research"
            ? "bg-white text-zinc-900"
            : "text-zinc-300 hover:bg-white/[0.06]"
        }`}
      >
        Deep Research
      </button>
    </div>
  );
}

function ResearchProgress({ steps }: { steps: ResearchStep[] }) {
  if (!steps?.length) return null;

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
      <div className="text-sm font-medium text-zinc-100">Research Progress</div>
      <div className="mt-4 space-y-3">
        {steps.map((s, idx) => (
          <div key={`${s.step}-${idx}`} className="flex items-center gap-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-full border border-emerald-400/30 bg-emerald-500/10 text-xs text-emerald-200">
              ✓
            </div>
            <div className="text-sm text-zinc-300">{s.step}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResearchReport({
  result,
  showProgress,
}: {
  result: ResearchResult | null;
  showProgress: boolean;
}) {
  if (!result) return null;

  return (
    <div className="space-y-4">
      {result.intent && (
        <div className="flex flex-wrap gap-2">
          <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-xs text-zinc-300">
            Intent: {result.intent.intent}
          </span>
          <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-xs text-zinc-300">
            Confidence: {Math.round((result.intent.confidence || 0) * 100)}%
          </span>
          {result.verified?.verified && (
            <span className="rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-200">
              Verified
            </span>
          )}
          {result.contradiction && (
            <span className="rounded-full border border-fuchsia-400/30 bg-fuchsia-500/10 px-3 py-1 text-xs text-fuchsia-200">
              Possible contradiction
            </span>
          )}
        </div>
      )}

      {showProgress && result.steps && <ResearchProgress steps={result.steps} />}

      {result.plan?.sub_questions?.length ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
          <div className="text-sm font-medium text-zinc-100">Research Plan</div>
          <div className="mt-3 space-y-2">
            {result.plan.sub_questions.map((q, i) => (
              <div key={i} className="rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-zinc-300">
                {i + 1}. {q}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {result.report ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.05] p-5">
          <div className="text-sm font-medium text-zinc-100">Research Report</div>
          <div className="mt-4 whitespace-pre-wrap leading-7 text-zinc-200">
            {result.report}
          </div>
        </div>
      ) : null}

      {result.findings?.length ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
          <div className="text-sm font-medium text-zinc-100">Findings</div>
          <div className="mt-4 space-y-4">
            {result.findings.map((f, i) => (
              <div key={i} className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="text-sm font-medium text-zinc-100">{f.sub_question}</div>
                <div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-zinc-300">
                  {f.answer}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {result.citations?.length ? (
        <details className="overflow-hidden rounded-2xl border border-white/10 bg-black/20">
          <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-zinc-200">
            Sources ({result.citations.length})
          </summary>
          <div className="grid gap-3 border-t border-white/10 p-4 md:grid-cols-2">
            {result.citations.map((c, i) => (
              <motion.div
                key={`${c.chunk_id}-${i}`}
                whileHover={{ y: -2 }}
                className="rounded-2xl border border-white/10 bg-white/[0.04] p-4"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="truncate text-xs font-semibold text-zinc-200">
                    {c.source} • p{c.page}
                  </div>
                  <span className="rounded-full bg-white/10 px-2 py-1 text-[10px] text-zinc-200">
                    {Number(c.score || 0).toFixed(3)}
                  </span>
                </div>
                <div className="mt-3 text-sm leading-6 text-zinc-300">
                  {c.quote}
                </div>
              </motion.div>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}

export default function Home() {
  const [uploaded, setUploaded] = useState<string[]>([]);
  const [indexed, setIndexed] = useState<IndexedDoc[]>([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([
    {
      role: "assistant",
      text: "Upload a document from the left panel and ask a question. I will answer only from your uploaded content and show the supporting sources.",
    },
  ]);

  const [mode, setMode] = useState<Mode>("ask");
  const [researchResult, setResearchResult] = useState<ResearchResult | null>(null);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [busyUpload, setBusyUpload] = useState(false);
  const [error, setError] = useState("");
  const [showMobileSidebar, setShowMobileSidebar] = useState(false);

  const chatEndRef = useRef<HTMLDivElement | null>(null);

  async function refresh() {
    const lib = await getLibrary();
    setUploaded(lib.uploaded || []);
    setIndexed(lib.indexed || []);
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, researchResult]);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;

    setError("");
    setBusyUpload(true);

    try {
      const res = await uploadFile(f);
      setSelectedFile(res.filename);
      await indexFile(res.filename, "default");
      await refresh();
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setBusyUpload(false);
      e.target.value = "";
    }
  }

  async function sendAsk(text: string) {
    const q = text.trim();
    if (!q || loading) return;

    setInput("");
    setError("");
    setResearchResult(null);
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        text: "",
        grounded: true,
        confidence: 0.8,
        contradiction: false,
        citations: [],
      },
    ]);

    try {
      await askStream(q, { source: selectedFile || undefined }, (chunk) => {
        setMessages((prev) => {
          const idx = prev.length - 1;
          if (idx < 0) return prev;
          return prev.map((msg, i) =>
            i === idx && msg.role === "assistant"
              ? { ...msg, text: msg.text + chunk }
              : msg
          );
        });
      });
    } catch (err: any) {
      setError(err.message || "Ask failed");
    } finally {
      setLoading(false);
    }
  }

  async function sendResearch(text: string) {
    const q = text.trim();
    if (!q || loading) return;

    setInput("");
    setError("");
    setLoading(true);

    setResearchResult({
      steps: [],
      findings: [],
      citations: [],
      report: "",
    });

    try {
      await deepResearchStream(q, selectedFile || undefined, (event) => {
        const type = event?.type;
        const data = event?.data;

        setResearchResult((prev) => {
          const current: ResearchResult = prev || {
            steps: [],
            findings: [],
            citations: [],
            report: "",
          };

          if (type === "step") {
            return {
              ...current,
              steps: [...(current.steps || []), data],
            };
          }

          if (type === "intent") {
            return {
              ...current,
              intent: data,
            };
          }

          if (type === "plan") {
            return {
              ...current,
              plan: data,
            };
          }

          if (type === "finding") {
            return {
              ...current,
              findings: [...(current.findings || []), data],
            };
          }

          if (type === "report") {
            return {
              ...current,
              report: data.report || "",
            };
          }

          if (type === "done") {
            return {
              ...current,
              ...data,
            };
          }

          return current;
        });
      });
    } catch (err: any) {
      setError(err.message || "Deep research failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmit(text: string) {
    if (mode === "ask") {
      await sendAsk(text);
    } else {
      await sendResearch(text);
    }
  }

  const indexedMap = useMemo(() => {
    const map = new Map<string, IndexedDoc>();
    indexed.forEach((doc) => map.set(doc.filename, doc));
    return map;
  }, [indexed]);

  const selectedDocInfo = selectedFile ? indexedMap.get(selectedFile) : null;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,#18181b_0%,#09090b_42%)] text-zinc-100">
      <div className="mx-auto flex min-h-screen max-w-[1600px]">
        <motion.aside
          initial={false}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.22 }}
          className={`fixed inset-y-0 left-0 z-40 w-[320px] border-r border-white/10 bg-zinc-950/92 backdrop-blur-xl lg:static lg:block lg:w-[320px] ${
            showMobileSidebar ? "block" : "hidden lg:block"
          }`}
        >
          <div className="flex h-full flex-col p-4">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold tracking-tight">VaultQA</div>
                <div className="text-xs text-zinc-400">Private document workspace</div>
              </div>
              <button
                onClick={() => setShowMobileSidebar(false)}
                className="rounded-lg border border-white/10 px-2 py-1 text-xs text-zinc-300 lg:hidden"
              >
                Close
              </button>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 shadow-xl shadow-black/20">
              <div className="text-sm font-medium">Documents</div>
              <div className="mt-1 text-xs text-zinc-400">Upload files and start chatting.</div>

              <input
                type="file"
                accept=".pdf,.txt"
                onChange={onUpload}
                disabled={busyUpload}
                className="mt-4 block w-full text-sm text-zinc-200 file:mr-3 file:rounded-xl file:border-0 file:bg-white file:px-4 file:py-2 file:text-zinc-900 hover:file:bg-zinc-100"
              />

              <div className="mt-4 max-h-[52vh] space-y-2 overflow-auto pr-1">
                {uploaded.length === 0 && (
                  <div className="rounded-xl border border-dashed border-white/10 p-4 text-sm text-zinc-500">
                    No documents yet.
                  </div>
                )}

                {uploaded.map((file) => {
                  const info = indexedMap.get(file);
                  const active = selectedFile === file;

                  return (
                    <motion.button
                      key={file}
                      whileHover={{ y: -1, scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      onClick={() => setSelectedFile(file)}
                      className={`w-full rounded-2xl border p-3 text-left transition ${
                        active
                          ? "border-white/20 bg-white/[0.08]"
                          : "border-white/10 bg-black/20 hover:bg-white/[0.06]"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="truncate text-sm font-semibold">{file}</div>
                          {info ? (
                            <div className="mt-1 text-xs text-zinc-400">
                              {info.indexed_chunks} chunks • {timeAgo(info.indexed_at)}
                            </div>
                          ) : (
                            <div className="mt-1 text-xs text-zinc-500">Uploaded</div>
                          )}
                        </div>

                        <span
                          className={`shrink-0 rounded-full px-2 py-1 text-[10px] ${
                            info
                              ? "border border-emerald-400/30 bg-emerald-500/10 text-emerald-200"
                              : "border border-white/10 bg-white/5 text-zinc-300"
                          }`}
                        >
                          {info ? "ready" : "new"}
                        </span>
                      </div>
                    </motion.button>
                  );
                })}
              </div>
            </div>
          </div>
        </motion.aside>

        <main className="flex min-w-0 flex-1 flex-col lg:ml-0">
          <div className="sticky top-0 z-20 border-b border-white/10 bg-zinc-950/70 backdrop-blur-xl">
            <div className="flex flex-col gap-3 px-4 py-4 lg:px-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setShowMobileSidebar(true)}
                    className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm lg:hidden"
                  >
                    Library
                  </button>

                  <div>
                    <div className="text-base font-semibold tracking-tight">Workspace</div>
                    <div className="text-xs text-zinc-400">
                      {selectedFile ? `Using ${selectedFile}` : "Searching all documents"}
                    </div>
                  </div>
                </div>

                {selectedDocInfo && (
                  <div className="hidden rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-zinc-300 md:block">
                    {selectedDocInfo.indexed_chunks} chunks
                  </div>
                )}
              </div>

              <ModeToggle mode={mode} setMode={setMode} />
            </div>
          </div>

          <div className="flex min-h-0 flex-1 flex-col px-4 pb-4 pt-4 lg:px-6">
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-4 rounded-2xl border border-red-400/20 bg-red-500/10 px-4 py-3 text-sm text-red-200"
              >
                {error}
              </motion.div>
            )}

            <div className="min-h-0 flex-1 overflow-auto">
              <div className="mx-auto flex max-w-4xl flex-col gap-6 pb-6">
                {mode === "ask" ? (
                  <>
                    {messages.length === 1 && !loading && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="mx-auto mt-8 w-full max-w-3xl"
                      >
                        <div className="mb-6 text-center">
                          <div className="text-3xl font-semibold tracking-tight">Ask your documents</div>
                          <div className="mt-2 text-sm text-zinc-400">
                            Grounded answers with evidence from your files only.
                          </div>
                        </div>

                        <div className="grid gap-3 md:grid-cols-2">
                          {starterPrompts.map((prompt) => (
                            <motion.button
                              key={prompt}
                              whileHover={{ scale: 1.01, y: -2 }}
                              whileTap={{ scale: 0.99 }}
                              onClick={() => handleSubmit(prompt)}
                              className="rounded-2xl border border-white/10 bg-white/[0.05] p-4 text-left transition hover:bg-white/[0.08]"
                            >
                              <div className="text-sm font-medium">{prompt}</div>
                            </motion.button>
                          ))}
                        </div>
                      </motion.div>
                    )}

                    {messages.map((msg, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, y: 14 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.22, ease: "easeOut" }}
                        className={msg.role === "user" ? "flex justify-end" : "flex justify-start"}
                      >
                        <div
                          className={`max-w-[88%] rounded-[24px] px-5 py-4 shadow-xl shadow-black/20 ${
                            msg.role === "user"
                              ? "bg-white text-zinc-900"
                              : "border border-white/10 bg-white/[0.05] text-zinc-100 backdrop-blur-sm"
                          }`}
                        >
                          <div className="whitespace-pre-wrap leading-7">{msg.text}</div>
                        </div>
                      </motion.div>
                    ))}

                    {loading && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex justify-start"
                      >
                        <div className="rounded-[24px] border border-white/10 bg-white/[0.05] px-5 py-4 shadow-xl shadow-black/20">
                          <TypingDots />
                        </div>
                      </motion.div>
                    )}
                  </>
                ) : (
                  <>
                    {!researchResult && !loading && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="mx-auto mt-8 w-full max-w-3xl"
                      >
                        <div className="mb-6 text-center">
                          <div className="text-3xl font-semibold tracking-tight">Deep Research</div>
                          <div className="mt-2 text-sm text-zinc-400">
                            Multi-step reasoning, evidence retrieval, and structured research reports.
                          </div>
                        </div>

                        <div className="grid gap-3 md:grid-cols-2">
                          {starterPrompts.map((prompt) => (
                            <motion.button
                              key={prompt}
                              whileHover={{ scale: 1.01, y: -2 }}
                              whileTap={{ scale: 0.99 }}
                              onClick={() => handleSubmit(prompt)}
                              className="rounded-2xl border border-white/10 bg-white/[0.05] p-4 text-left transition hover:bg-white/[0.08]"
                            >
                              <div className="text-sm font-medium">{prompt}</div>
                            </motion.button>
                          ))}
                        </div>
                      </motion.div>
                    )}

                    {loading && mode === "research" && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="rounded-[24px] border border-white/10 bg-white/[0.05] px-5 py-4 shadow-xl shadow-black/20"
                      >
                        <TypingDots />
                      </motion.div>
                    )}

                    {researchResult && (
                      <ResearchReport result={researchResult} showProgress={loading} />
                    )}
                  </>
                )}

                <div ref={chatEndRef} />
              </div>
            </div>

            <div className="mx-auto mt-4 w-full max-w-4xl">
              <div className="mb-3 flex flex-wrap gap-2">
                {starterPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => setInput(prompt)}
                    className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-zinc-300 transition hover:bg-white/[0.08]"
                  >
                    {prompt}
                  </button>
                ))}
              </div>

              <div className="rounded-[28px] border border-white/10 bg-zinc-900/85 p-2 shadow-2xl shadow-black/20 backdrop-blur-xl">
                <div className="flex items-end gap-2">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(input);
                      }
                    }}
                    rows={1}
                    className="max-h-40 min-h-[52px] flex-1 resize-none rounded-[22px] bg-transparent px-4 py-3 text-sm text-zinc-100 outline-none placeholder:text-zinc-500"
                    placeholder={
                      mode === "ask"
                        ? "Ask about your documents..."
                        : "Research your documents deeply..."
                    }
                  />

                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleSubmit(input)}
                    disabled={loading}
                    className="rounded-[22px] bg-white px-5 py-3 text-sm font-semibold text-zinc-900 transition hover:bg-zinc-100 disabled:opacity-50"
                  >
                    {mode === "ask" ? "Send" : "Research"}
                  </motion.button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}