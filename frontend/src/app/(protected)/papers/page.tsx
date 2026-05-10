"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import { papersApi } from "@/lib/api";

export default function PapersPage() {
  const [papers, setPapers]   = useState<any[]>([]);
  const [loading, setLoading]  = useState(true);
  const [error, setError]      = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [search, setSearch]    = useState("");

  // poll ref — kept alive while any paper is still processing
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError("");
    papersApi
      .list()
      .then((r) => setPapers(r.data))
      .catch(() => setError("Could not load papers. Please check your connection and try again."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // ── BUG FIX: keep polling the list while any paper is "processing"
  // so the library page auto-updates the badge + topics + AI Summary
  // tag without requiring a manual refresh.
  //
  // We poll at 5s intervals. When all papers reach a terminal state
  // ("ready" or "failed") we stop polling automatically.
  useEffect(() => {
    if (pollRef.current) clearTimeout(pollRef.current);

    const anyProcessing = papers.some((p) => p.status === "processing");
    if (!anyProcessing) return;

    pollRef.current = setTimeout(() => {
      papersApi
        .list()
        .then((r) => setPapers(r.data))
        .catch(() => {/* silent — don't blow away existing data */});
    }, 5000);

    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, [papers]);

  const handleDelete = async (paperId: string, e: React.MouseEvent) => {
    e.preventDefault();
    if (!confirm("Permanently delete this paper?")) return;
    setDeleting(paperId);
    try {
      await papersApi.delete(paperId);
      setPapers((prev) => prev.filter((p) => p.paper_id !== paperId));
    } catch {
      alert("Failed to delete paper. Please try again.");
    } finally {
      setDeleting(null);
    }
  };

  const filtered = papers.filter(
    (p) =>
      !search ||
      p.title.toLowerCase().includes(search.toLowerCase()) ||
      p.authors?.join(" ").toLowerCase().includes(search.toLowerCase())
  );

  if (error) {
    return (
      <div style={{ padding: "48px 52px", maxWidth: 860 }}>
        <div style={{
          padding: "24px", borderRadius: 8,
          background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)",
          display: "flex", alignItems: "center", gap: 16,
        }}>
          <span style={{ fontSize: "1.2rem" }}>⚠️</span>
          <div>
            <p style={{ fontSize: "0.9rem", color: "#DC2626", fontWeight: 600 }}>{error}</p>
            <button
              onClick={load}
              style={{
                marginTop: 8, fontSize: "0.8rem", color: "var(--navy)",
                background: "none", border: "1px solid var(--border-strong)",
                borderRadius: 5, padding: "5px 12px", cursor: "pointer",
              }}
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: "48px 52px", maxWidth: 860 }}>
      {/* Page header */}
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 36 }}>
        <div>
          <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.65rem", letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--amber)", marginBottom: 8 }}>
            Library
          </p>
          <h1 style={{ fontSize: "2rem", fontFamily: "'DM Serif Display', serif", fontWeight: 400, color: "var(--navy)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Your Papers
          </h1>
          <p style={{ marginTop: 8, color: "var(--text-secondary)", fontSize: "0.88rem" }}>
            {papers.length} document{papers.length !== 1 ? "s" : ""} in collection
          </p>
        </div>
        <Link
          href="/upload"
          style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "10px 20px", background: "var(--navy)", color: "var(--cream)",
            fontSize: "0.82rem", fontWeight: 500, borderRadius: 6,
            textDecoration: "none", letterSpacing: "0.01em",
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Add Paper
        </Link>
      </div>

      {/* Search */}
      {papers.length > 0 && (
        <div style={{ position: "relative", marginBottom: 24 }}>
          <svg style={{ position: "absolute", left: 14, top: "50%", transform: "translateY(-50%)", color: "var(--muted)" }}
            width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by title or author…"
            style={{
              width: "100%", padding: "10px 14px 10px 40px",
              border: "1px solid var(--border-strong)", borderRadius: 6,
              fontSize: "0.85rem", background: "var(--surface)",
              color: "var(--text)", outline: "none", fontFamily: "'DM Sans', sans-serif",
            }}
          />
        </div>
      )}

      <div className="rule" style={{ marginBottom: 24 }} />

      {/* Loading skeletons */}
      {loading ? (
        <div style={{ display: "flex", flexDirection: "column" }}>
          {[1, 2, 3].map((i) => (
            <div key={i} style={{ padding: "22px 0", borderBottom: "1px solid var(--border)" }}>
              <div className="skeleton" style={{ width: "60%", height: 16, marginBottom: 10 }} />
              <div className="skeleton" style={{ width: "28%", height: 11, marginBottom: 10 }} />
              <div className="skeleton" style={{ width: "85%", height: 11, marginBottom: 6 }} />
              <div className="skeleton" style={{ width: "70%", height: 11 }} />
            </div>
          ))}
        </div>
      ) : papers.length === 0 ? (
        <div style={{ textAlign: "center", padding: "80px 0" }}>
          <div style={{
            width: 56, height: 56, borderRadius: "50%", border: "1px solid var(--border-strong)",
            display: "flex", alignItems: "center", justifyContent: "center",
            margin: "0 auto 20px", color: "var(--muted)",
          }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
            </svg>
          </div>
          <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.2rem", color: "var(--navy)", marginBottom: 8 }}>
            No papers yet
          </p>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: 24 }}>
            Upload your first research paper to begin
          </p>
          <Link href="/upload" style={{
            padding: "10px 24px", background: "var(--navy)", color: "var(--cream)",
            borderRadius: 6, textDecoration: "none", fontSize: "0.85rem", fontWeight: 500,
          }}>
            Upload Paper
          </Link>
        </div>
      ) : (
        <>
          <div style={{ borderTop: "1px solid var(--border)" }}>
            {filtered.map((paper, idx) => (
              <Link
                key={paper.paper_id}
                href={`/papers/${paper.paper_id}`}
                style={{
                  display: "block",
                  padding: "20px 0",
                  borderBottom: "1px solid var(--border)",
                  textDecoration: "none",
                  animation: `fadeUp 0.35s ease ${idx * 0.04}s both`,
                }}
                className="paper-row"
              >
                {/* Title row */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, marginBottom: 5 }}>
                  <p style={{ fontWeight: 600, fontSize: "0.92rem", color: "var(--navy)", lineHeight: 1.4, flex: 1 }}>
                    {paper.title}
                  </p>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
                    {paper.status === "processing" && (
                      <span style={{
                        fontFamily: "'DM Mono', monospace", fontSize: "0.62rem",
                        color: "var(--amber)", background: "var(--amber-dim)",
                        border: "1px solid rgba(200,146,42,0.25)",
                        padding: "2px 7px", borderRadius: 3, letterSpacing: "0.05em",
                      }}>
                        Processing…
                      </span>
                    )}
                    {paper.status === "failed" && (
                      <span style={{
                        fontFamily: "'DM Mono', monospace", fontSize: "0.62rem",
                        color: "#DC2626", background: "rgba(239,68,68,0.08)",
                        border: "1px solid rgba(239,68,68,0.2)",
                        padding: "2px 7px", borderRadius: 3, letterSpacing: "0.05em",
                      }}>
                        AI Failed
                      </span>
                    )}
                    <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.7rem", color: "var(--muted)", whiteSpace: "nowrap" }}>
                      {new Date(paper.upload_date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                    </span>
                    <button
                      onClick={(e) => handleDelete(paper.paper_id, e)}
                      disabled={deleting === paper.paper_id}
                      style={{
                        width: 26, height: 26, borderRadius: 5, border: "1px solid transparent",
                        background: "transparent", cursor: "pointer", display: "flex",
                        alignItems: "center", justifyContent: "center",
                        color: "var(--muted)", transition: "all 0.15s", flexShrink: 0,
                      }}
                      title="Delete paper"
                    >
                      {deleting === paper.paper_id ? (
                        <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none">
                          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
                          <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                        </svg>
                      ) : (
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" /><path d="M10 11v6M14 11v6" />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>

                {/* Authors */}
                {paper.authors?.length > 0 && (
                  <p style={{ fontSize: "0.77rem", color: "var(--text-secondary)", fontStyle: "italic", marginBottom: 8 }}>
                    {paper.authors.slice(0, 4).join(", ")}
                    {paper.authors.length > 4 ? ` +${paper.authors.length - 4} more` : ""}
                  </p>
                )}

                {/* Abstract snippet */}
                {paper.abstract && (
                  <p style={{
                    fontSize: "0.81rem", color: "var(--text-secondary)", lineHeight: 1.65,
                    marginBottom: 10,
                    display: "-webkit-box", WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical", overflow: "hidden",
                  }}>
                    {paper.abstract}
                  </p>
                )}

                {/* Topics + AI badge */}
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5, alignItems: "center" }}>
                  {paper.topics?.slice(0, 3).map((t: string) => (
                    <span key={t} className="badge" style={{ fontSize: "0.65rem" }}>{t}</span>
                  ))}
                  {paper.has_summary && (
                    <span className="badge badge-amber" style={{ fontSize: "0.63rem", display: "inline-flex", alignItems: "center", gap: 3 }}>
                      <svg width="8" height="8" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                      </svg>
                      AI Summary
                    </span>
                  )}
                </div>
              </Link>
            ))}
          </div>

          {filtered.length === 0 && search && (
            <p style={{ textAlign: "center", padding: "48px 0", color: "var(--text-secondary)", fontSize: "0.88rem" }}>
              No papers match &ldquo;{search}&rdquo;
            </p>
          )}
        </>
      )}

      <style jsx>{`
        .paper-row:hover { background: var(--surface-alt); margin: 0 -16px; padding-left: 16px; padding-right: 16px; border-radius: 6px; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes spin { to { transform: rotate(360deg); } }
        .animate-spin { animation: spin 0.8s linear infinite; }
      `}</style>
    </div>
  );
}