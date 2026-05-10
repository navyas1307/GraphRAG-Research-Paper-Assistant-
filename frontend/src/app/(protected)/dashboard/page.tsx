"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { papersApi, searchApi } from "@/lib/api";

interface Message { role: "user" | "assistant"; content: string; sources?: any[]; }

const SUGGESTIONS = [
  "What are the main contributions?",
  "Which datasets were evaluated?",
  "What methods were proposed?",
  "Summarize the key findings",
];

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  );
}

export default function DashboardPage() {
  const [papers, setPapers] = useState<any[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [papersLoading, setPapersLoading] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // FIX: load once, no loop
  useEffect(() => {
    papersApi
      .list()
      .then((r) => setPapers(r.data))
      .catch(() => {})
      .finally(() => setPapersLoading(false));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // FIX: auto-resize textarea as user types
  useEffect(() => {
    const ta = inputRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  }, [input]);

  const send = useCallback(async (question?: string) => {
    const q = question || input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await searchApi.query(q, selectedPaper || undefined, []);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.data.answer, sources: res.data.sources },
      ]);
    } catch (err: any) {
      const msg =
        err.response?.status === 503
          ? "Vector search is currently unavailable. Please ensure Qdrant is running."
          : "An error occurred. Please try again.";
      setMessages((prev) => [...prev, { role: "assistant", content: msg }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, selectedPaper]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const isEmpty = messages.length === 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", position: "relative" }}>
      {/* Header */}
      <div style={{
        padding: "16px 40px", background: "var(--surface)",
        borderBottom: "1px solid var(--border)",
        display: "flex", alignItems: "center", gap: 16, flexShrink: 0,
      }}>
        <div>
          <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1rem", color: "var(--navy)", letterSpacing: "-0.02em" }}>
            Research Chat
          </p>
          <p style={{ fontSize: "0.74rem", color: "var(--muted)", marginTop: 1 }}>
            {papersLoading ? "Loading…" : `${papers.length} paper${papers.length !== 1 ? "s" : ""} in scope`}
          </p>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 400 }}>Scope:</label>
          <select
            value={selectedPaper}
            onChange={(e) => setSelectedPaper(e.target.value)}
            style={{
              fontSize: "0.8rem", border: "1px solid var(--border-strong)", borderRadius: 5,
              padding: "6px 10px", background: "var(--surface)", color: "var(--text)",
              fontFamily: "'DM Sans', sans-serif", outline: "none",
            }}
          >
            <option value="">All papers</option>
            {papers.map((p) => (
              <option key={p.paper_id} value={p.paper_id}>
                {p.title.slice(0, 48)}{p.title.length > 48 ? "…" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "32px 40px" }}>
        {isEmpty && (
          <div style={{ maxWidth: 560, margin: "60px auto 0", textAlign: "center" }}>
            <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.6rem", color: "var(--navy)", letterSpacing: "-0.03em", marginBottom: 10, lineHeight: 1.2 }}>
              What would you like<br />to know?
            </p>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 32, lineHeight: 1.6 }}>
              {papers.length === 0
                ? "Upload research papers to begin querying your library."
                : "Ask questions across your library or focus on a single paper."}
            </p>
            {papers.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center" }}>
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    style={{
                      padding: "8px 16px", border: "1px solid var(--border-strong)",
                      borderRadius: 24, background: "var(--surface)", cursor: "pointer",
                      fontSize: "0.8rem", color: "var(--text-secondary)", fontFamily: "'DM Sans', sans-serif",
                      transition: "all 0.15s",
                    }}
                    onMouseEnter={(e) => {
                      (e.target as HTMLElement).style.borderColor = "var(--amber)";
                      (e.target as HTMLElement).style.color = "var(--amber)";
                    }}
                    onMouseLeave={(e) => {
                      (e.target as HTMLElement).style.borderColor = "var(--border-strong)";
                      (e.target as HTMLElement).style.color = "var(--text-secondary)";
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        <div style={{ maxWidth: 720, margin: "0 auto", display: "flex", flexDirection: "column", gap: 24 }}>
          {messages.map((msg, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                flexDirection: msg.role === "user" ? "row-reverse" : "row",
                gap: 12,
                animation: "fadeUp 0.25s ease",
              }}
            >
              {msg.role === "assistant" && (
                <div style={{
                  width: 28, height: 28, borderRadius: "50%",
                  background: "var(--navy)", color: "var(--amber)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  flexShrink: 0, marginTop: 2,
                  fontFamily: "'DM Mono', monospace", fontSize: "0.6rem", fontWeight: 500, letterSpacing: "0.05em",
                }}>
                  AI
                </div>
              )}
              <div style={{ maxWidth: "78%" }}>
                <div style={{
                  padding: "12px 16px",
                  background: msg.role === "user" ? "var(--navy)" : "var(--surface)",
                  color: msg.role === "user" ? "var(--cream)" : "var(--text)",
                  borderRadius: msg.role === "user" ? "12px 4px 12px 12px" : "4px 12px 12px 12px",
                  fontSize: "0.85rem",
                  lineHeight: 1.65,
                  border: msg.role === "assistant" ? "1px solid var(--border)" : "none",
                  boxShadow: msg.role === "assistant" ? "0 1px 4px rgba(0,0,0,0.04)" : "none",
                }}>
                  {msg.role === "assistant" ? (
                    <div className="prose prose-sm" style={{ maxWidth: "none" }}>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  ) : msg.content}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 4 }}>
                    {msg.sources.slice(0, 3).map((s: any, j: number) => (
                      <div key={j} style={{
                        padding: "7px 12px", background: "var(--surface-alt)",
                        border: "1px solid var(--border)", borderRadius: 5,
                        display: "flex", alignItems: "center", gap: 8,
                      }}>
                        <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.65rem", color: "var(--amber)", fontWeight: 500 }}>
                          {(s.score * 100).toFixed(0)}%
                        </span>
                        <span style={{ fontSize: "0.74rem", color: "var(--text-secondary)", fontWeight: 500 }}>{s.paper_title}</span>
                        <span style={{ fontSize: "0.72rem", color: "var(--muted)" }}>· {s.section}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div style={{ display: "flex", gap: 12 }}>
              <div style={{
                width: 28, height: 28, borderRadius: "50%", background: "var(--navy)",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "var(--amber)", fontFamily: "'DM Mono', monospace", fontSize: "0.6rem", flexShrink: 0,
              }}>AI</div>
              <div style={{
                padding: "14px 18px", background: "var(--surface)",
                border: "1px solid var(--border)", borderRadius: "4px 12px 12px 12px",
                display: "flex", alignItems: "center", gap: 5,
              }}>
                {[0, 1, 2].map((n) => (
                  <div key={n} style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: "var(--navy-400)",
                    animation: `pulse-dot 1.2s ease ${n * 0.18}s infinite`,
                  }} />
                ))}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div style={{
        borderTop: "1px solid var(--border)", background: "var(--surface)",
        padding: "16px 40px 20px", flexShrink: 0,
      }}>
        <div style={{ maxWidth: 720, margin: "0 auto", position: "relative" }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={
              papersLoading
                ? "Loading papers…"
                : papers.length === 0
                ? "Upload papers to start querying…"
                : "Ask a question about your papers… (Enter to send)"
            }
            disabled={papers.length === 0 || loading || papersLoading}
            rows={1}
            style={{
              width: "100%",
              padding: "12px 52px 12px 16px",
              border: "1px solid var(--border-strong)",
              borderRadius: 8,
              fontSize: "0.85rem",
              background: papers.length === 0 ? "var(--surface-alt)" : "var(--surface)",
              color: "var(--text)",
              outline: "none",
              fontFamily: "'DM Sans', sans-serif",
              resize: "none",
              lineHeight: 1.5,
              overflowY: "hidden",
              transition: "border-color 0.15s",
            }}
            onFocus={(e) => (e.target.style.borderColor = "var(--navy-600)")}
            onBlur={(e) => (e.target.style.borderColor = "var(--border-strong)")}
          />
          <button
            onClick={() => send()}
            disabled={!input.trim() || loading || papers.length === 0}
            style={{
              position: "absolute", right: 10, bottom: 10,
              width: 32, height: 32, borderRadius: 6, border: "none",
              background: input.trim() && !loading && papers.length > 0 ? "var(--navy)" : "var(--border)",
              color: input.trim() && !loading && papers.length > 0 ? "var(--cream)" : "var(--muted)",
              cursor: input.trim() && !loading && papers.length > 0 ? "pointer" : "not-allowed",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.15s",
            }}
          >
            <SendIcon />
          </button>
        </div>
        <p style={{ textAlign: "center", fontSize: "0.7rem", color: "var(--muted)", marginTop: 8 }}>
          Shift+Enter for new line
        </p>
      </div>

      <style jsx>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse-dot {
          0%, 100% { transform: scale(0.8); opacity: 0.4; }
          50% { transform: scale(1.15); opacity: 1; }
        }
      `}</style>
    </div>
  );
}