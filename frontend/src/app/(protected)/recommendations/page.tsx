"use client";
import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { papersApi } from "@/lib/api";

function ExternalIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
    </svg>
  );
}

function RecommendationsContent() {
  const searchParams = useSearchParams();
  const preselectedId = searchParams.get("paper") || "";

  const [papers, setPapers] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState(preselectedId);
  const [recs, setRecs] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    papersApi.list().then((r) => setPapers(r.data)).catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedId) fetchRecs(selectedId);
  }, []);

  const fetchRecs = async (paperId: string) => {
    setSelectedId(paperId);
    setLoading(true); setError(""); setRecs(null);
    try {
      const res = await papersApi.recommendations(paperId);
      setRecs(res.data);
    } catch {
      setError("Failed to load recommendations. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const selectedPaper = papers.find((p) => p.paper_id === selectedId);

  return (
    <div style={{ padding: "48px 52px", maxWidth: 860 }}>
      {/* Header */}
      <div style={{ marginBottom: 40 }}>
        <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.65rem", letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--amber)", marginBottom: 8 }}>
          Discover
        </p>
        <h1 style={{ fontSize: "2rem", fontFamily: "'DM Serif Display', serif", color: "var(--navy)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
          Recommendations
        </h1>
        <p style={{ marginTop: 10, color: "var(--text-secondary)", fontSize: "0.88rem" }}>
          Surface related work from your library and arXiv based on shared topics, methods, and entities.
        </p>
      </div>

      {/* Paper selector */}
      <div style={{ padding: "20px 22px", border: "1px solid var(--border-strong)", borderRadius: 8, background: "var(--surface)", marginBottom: 28 }}>
        <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.62rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)", marginBottom: 12 }}>
          Source Paper
        </p>
        {papers.length === 0 ? (
          <p style={{ fontSize: "0.84rem", color: "var(--muted)" }}>
            No papers in library.{" "}
            <Link href="/upload" style={{ color: "var(--amber)", textDecoration: "none", fontWeight: 500 }}>
              Upload one →
            </Link>
          </p>
        ) : (
          <div style={{ display: "flex", gap: 10 }}>
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              style={{
                flex: 1, fontSize: "0.85rem", border: "1px solid var(--border-strong)", borderRadius: 6,
                padding: "9px 12px", background: "var(--surface)", color: "var(--text)",
                fontFamily: "'DM Sans', sans-serif", outline: "none"
              }}
            >
              <option value="">Choose a paper…</option>
              {papers.map((p) => (
                <option key={p.paper_id} value={p.paper_id}>
                  {p.title.length > 72 ? p.title.slice(0, 72) + "…" : p.title}
                </option>
              ))}
            </select>
            <button
              onClick={() => selectedId && fetchRecs(selectedId)}
              disabled={!selectedId || loading}
              style={{
                padding: "9px 20px", background: selectedId && !loading ? "var(--navy)" : "var(--border)",
                color: selectedId && !loading ? "var(--cream)" : "var(--muted)",
                border: "none", borderRadius: 6, fontSize: "0.84rem", fontWeight: 500,
                cursor: selectedId && !loading ? "pointer" : "not-allowed",
                fontFamily: "'DM Sans', sans-serif", transition: "all 0.15s", whiteSpace: "nowrap"
              }}
            >
              {loading ? "Searching…" : "Find Related"}
            </button>
          </div>
        )}
      </div>

      {/* Selected paper info */}
      {selectedPaper && (
        <div style={{
          padding: "14px 18px", background: "rgba(200,146,42,0.07)",
          border: "1px solid rgba(200,146,42,0.25)", borderRadius: 6, marginBottom: 28
        }}>
          <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.62rem", color: "var(--amber)", letterSpacing: "0.08em", marginBottom: 6 }}>
            SEARCHING RELATIVE TO
          </p>
          <p style={{ fontWeight: 500, fontSize: "0.86rem", color: "var(--navy)" }}>{selectedPaper.title}</p>
          {selectedPaper.topics?.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 8 }}>
              {selectedPaper.topics.slice(0, 5).map((t: string) => (
                <span key={t} className="badge badge-amber">{t}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {error && (
        <div style={{
          padding: "12px 16px", background: "rgba(220,38,38,0.05)",
          border: "1px solid rgba(220,38,38,0.2)", borderRadius: 6,
          fontSize: "0.82rem", color: "#b91c1c", marginBottom: 24
        }}>
          {error}
        </div>
      )}

      {loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="skeleton" style={{ height: 72, borderRadius: 6 }} />
          ))}
        </div>
      )}

      {recs && !loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
          {/* Library section */}
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.1rem", color: "var(--navy)" }}>
                From Your Library
              </p>
              <span className="badge">{recs.local?.length || 0} found</span>
            </div>
            <div className="rule" style={{ marginBottom: 14 }} />

            {recs.local?.length === 0 ? (
              <p style={{ fontSize: "0.84rem", color: "var(--muted)", padding: "20px 0" }}>
                No related papers in your library yet.
              </p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column" }}>
                {recs.local.map((r: any, i: number) => (
                  <div key={i} style={{
                    padding: "16px 0", borderBottom: "1px solid var(--border)",
                    display: "flex", gap: 16, alignItems: "flex-start",
                    animation: `fadeUp 0.3s ease ${i * 0.05}s both`
                  }}>
                    <span className="badge badge-navy" style={{ marginTop: 2, flexShrink: 0 }}>Local</span>
                    <div style={{ flex: 1 }}>
                      <p style={{ fontWeight: 500, fontSize: "0.86rem", color: "var(--navy)", lineHeight: 1.35, marginBottom: 4 }}>
                        {r.title}
                      </p>
                      {r.authors?.length > 0 && (
                        <p style={{ fontSize: "0.76rem", color: "var(--text-secondary)", marginBottom: 4 }}>
                          {r.authors.slice(0, 3).join(", ")}
                        </p>
                      )}
                      <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.72rem", color: "var(--muted)" }}>
                        {r.relevance}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* arXiv section */}
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.1rem", color: "var(--navy)" }}>
                From arXiv
              </p>
              <span className="badge badge-amber">{recs.arxiv?.length || 0} found</span>
            </div>
            <div className="rule" style={{ marginBottom: 14 }} />

            {recs.arxiv?.length === 0 ? (
              <p style={{ fontSize: "0.84rem", color: "var(--muted)", padding: "20px 0" }}>
                No arXiv recommendations found.
              </p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column" }}>
                {recs.arxiv.map((r: any, i: number) => (
                  <div key={i} style={{
                    padding: "18px 0", borderBottom: "1px solid var(--border)",
                    display: "flex", gap: 16, alignItems: "flex-start",
                    animation: `fadeUp 0.3s ease ${i * 0.05}s both`
                  }}>
                    <span className="badge badge-amber" style={{ marginTop: 2, flexShrink: 0 }}>arXiv</span>
                    <div style={{ flex: 1 }}>
                      {r.url ? (
                        <a
                          href={r.url} target="_blank" rel="noopener noreferrer"
                          style={{
                            fontWeight: 500, fontSize: "0.86rem", color: "var(--navy)",
                            textDecoration: "none", lineHeight: 1.35, display: "inline-flex",
                            alignItems: "flex-start", gap: 5
                          }}
                        >
                          {r.title}
                          <span style={{ color: "var(--amber)", marginTop: 2, flexShrink: 0 }}><ExternalIcon /></span>
                        </a>
                      ) : (
                        <p style={{ fontWeight: 500, fontSize: "0.86rem", color: "var(--navy)", lineHeight: 1.35 }}>{r.title}</p>
                      )}
                      {r.authors?.length > 0 && (
                        <p style={{ fontSize: "0.76rem", color: "var(--text-secondary)", marginTop: 4, marginBottom: 6 }}>
                          {r.authors.slice(0, 3).join(", ")}
                        </p>
                      )}
                      {r.abstract && (
                        <p style={{
                          fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.6,
                          display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden"
                        }}>
                          {r.abstract}
                        </p>
                      )}
                      {r.relevance && (
                        <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.7rem", color: "var(--muted)", marginTop: 6, display: "block" }}>
                          {r.relevance}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {!recs && !loading && !error && (
        <div style={{ textAlign: "center", padding: "80px 0" }}>
          <div style={{
            width: 48, height: 48, borderRadius: "50%", border: "1px solid var(--border-strong)",
            display: "flex", alignItems: "center", justifyContent: "center",
            margin: "0 auto 18px", color: "var(--muted)"
          }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </div>
          <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.1rem", color: "var(--navy)", marginBottom: 6 }}>
            Select a paper above
          </p>
          <p style={{ fontSize: "0.82rem", color: "var(--muted)" }}>
            We'll search your library and arXiv for related work
          </p>
        </div>
      )}

      <style jsx>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

export default function RecommendationsPage() {
  return (
    <Suspense fallback={
      <div style={{ padding: "48px 52px" }}>
        <div className="skeleton" style={{ width: 180, height: 36, marginBottom: 16 }} />
        <div className="skeleton" style={{ width: 320, height: 18 }} />
      </div>
    }>
      <RecommendationsContent />
    </Suspense>
  );
}