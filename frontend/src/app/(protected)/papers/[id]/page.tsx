"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { papersApi } from "@/lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────

interface ResearchSummary {
  problem_definition?: string;
  background_and_prior_work?: string;
  methodology?: string;
  technical_novelty?: string;
  experimental_setup?: string;
  results_and_analysis?: string;
  limitations?: string;
  conclusion?: string;
}

interface ResearchGap {
  gap_identification?: string;
  remaining_limitations?: string;
  methodological_weaknesses?: string;
  future_directions?: string[];
}

interface Section {
  title: string;
  content: string;
  level?: number;
  subsections?: Section[];
}

interface Entities {
  methods?: string[];
  datasets?: string[];
  models?: string[];
  tasks?: string[];
  topics?: string[];
}

interface Paper {
  paper_id: string;
  title: string;
  authors: string[];
  abstract?: string;
  summary?: string;
  research_summary?: ResearchSummary;
  research_gap?: ResearchGap;
  upload_date: string;
  sections: Section[];
  entities: Entities;
  topics: string[];
  tables_and_figures?: any[];
  status: string;
  llm_status?: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function hasAiData(paper: Paper): boolean {
  const rs = paper.research_summary || {};
  const ents = paper.entities || {};
  return !!(
    rs.methodology ||
    rs.problem_definition ||
    rs.results_and_analysis ||
    (ents.methods && ents.methods.length > 0) ||
    (paper.topics && paper.topics.length > 0)
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function SectionCard({ label, value }: { label: string; value?: string }) {
  if (!value) return null;
  return (
    <div style={{
      marginBottom: 16,
      padding: "16px 18px",
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: 8,
    }}>
      <p style={{
        fontFamily: "'DM Mono', monospace",
        fontSize: "0.62rem",
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        color: "var(--amber)",
        marginBottom: 8,
      }}>
        {label}
      </p>
      <p style={{ fontSize: "0.875rem", color: "var(--text)", lineHeight: 1.7 }}>
        {value}
      </p>
    </div>
  );
}

function EntityPills({ label, items, amber }: { label: string; items?: string[]; amber?: boolean }) {
  if (!items || items.length === 0) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <p style={{
        fontFamily: "'DM Mono', monospace",
        fontSize: "0.58rem",
        letterSpacing: "0.1em",
        textTransform: "uppercase",
        color: amber ? "var(--amber)" : "var(--text-secondary)",
        marginBottom: 6,
      }}>
        {label}
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {items.map((item) => (
          <span
            key={item}
            className={amber ? "badge badge-amber" : "badge"}
            style={{ fontSize: "0.68rem" }}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function ProcessingBanner({ llmStatus }: { llmStatus?: string }) {
  if (llmStatus === "quota_exceeded") {
    return (
      <div style={{
        padding: "14px 18px",
        borderRadius: 8,
        background: "rgba(239,68,68,0.06)",
        border: "1px solid rgba(239,68,68,0.2)",
        marginBottom: 24,
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}>
        <span>⚠️</span>
        <p style={{ fontSize: "0.83rem", color: "#DC2626" }}>
          AI quota exhausted for today — analysis will complete automatically tomorrow or after re-enrichment.
        </p>
      </div>
    );
  }
  return (
    <div style={{
      padding: "14px 18px",
      borderRadius: 8,
      background: "var(--amber-dim)",
      border: "1px solid rgba(200,146,42,0.25)",
      marginBottom: 24,
      display: "flex",
      alignItems: "center",
      gap: 10,
    }}>
      <svg className="spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" strokeWidth={2.5} strokeLinecap="round">
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </svg>
      <p style={{ fontSize: "0.83rem", color: "var(--amber)" }}>
        AI analysis in progress — entities and summaries will appear shortly…
      </p>
    </div>
  );
}

// ─── Right sidebar: Entities ──────────────────────────────────────────────────

function EntitiesSidebar({ paper, isProcessing }: { paper: Paper; isProcessing: boolean }) {
  const ents = paper.entities || {};
  const hasEntities = hasAiData(paper);

  return (
    <aside style={{
      width: 220,
      flexShrink: 0,
      position: "sticky",
      top: 32,
      alignSelf: "flex-start",
    }}>
      <p style={{
        fontFamily: "'DM Mono', monospace",
        fontSize: "0.6rem",
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: "var(--text-secondary)",
        marginBottom: 16,
        paddingBottom: 8,
        borderBottom: "1px solid var(--border)",
      }}>
        Entities
      </p>

      {isProcessing ? (
        // Skeletons while processing
        <div>
          {[0, 1, 2].map((i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              <div className="skeleton" style={{ width: "50%", height: 9, marginBottom: 8, borderRadius: 3 }} />
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                {[55, 70, 45].slice(0, i === 0 ? 3 : i === 1 ? 2 : 1).map((w, j) => (
                  <div key={j} className="skeleton" style={{ width: w, height: 22, borderRadius: 12 }} />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : !hasEntities ? (
        <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
          No entities extracted yet.
        </p>
      ) : (
        <>
          {paper.topics?.length > 0 && (
            <EntityPills label="Topics" items={paper.topics} amber />
          )}
          <EntityPills label="Methods"  items={ents.methods} />
          <EntityPills label="Datasets" items={ents.datasets} />
          <EntityPills label="Models"   items={ents.models} />
          <EntityPills label="Tasks"    items={ents.tasks} />
        </>
      )}
    </aside>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function PaperDetailPage() {
  const params = useParams();
  const router = useRouter();
  const paperId = params?.id as string;

  const [paper, setPaper]     = useState<Paper | null>(null);
  const [loading, setLoading]  = useState(true);
  const [error, setError]      = useState("");

  // polling ref — cleared when component unmounts or data is complete
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchPaper = useCallback(async () => {
    if (!paperId) return null;

    try {
      const res = await papersApi.get(paperId);
      setPaper(res.data);
      return res.data as Paper;
    } catch (err: any) {
      if (err?.response?.status === 404) {
        setError("Paper not found.");
      } else {
        setError("Could not load paper. Please check your connection.");
      }
      return null;
    }
  }, [paperId]);

  const scheduleNextPoll = useCallback((attempt: number, currentPaper: Paper) => {
    if (pollRef.current) clearTimeout(pollRef.current);

    const needsPolling =
      currentPaper.status === "processing" ||
      (currentPaper.status === "ready" && !hasAiData(currentPaper));

    if (!needsPolling) return;

    const delay = attempt < 3 ? 3000 : attempt < 6 ? 5000 : 10000;

    pollRef.current = setTimeout(async () => {
      try {
        const statusRes = await papersApi.getStatus(paperId);
        const { status, has_ai_data, llm_status } = statusRes.data;

        if (status === "processing" || (status === "ready" && !has_ai_data)) {
          setPaper((prev) =>
            prev ? { ...prev, status, llm_status: llm_status ?? prev.llm_status } : prev
          );
          scheduleNextPoll(attempt + 1, { ...currentPaper, status });
          return;
        }

        const updated = await fetchPaper();
        if (updated) {
          setPaper(updated);
          if (!hasAiData(updated)) {
            scheduleNextPoll(attempt + 1, updated);
          }
        }
      } catch {
        scheduleNextPoll(attempt + 1, currentPaper);
      }
    }, delay);
  }, [paperId, fetchPaper]);

  useEffect(() => {
    if (!paperId) return;
    let cancelled = false;

    (async () => {
      setLoading(true);
      setError("");
      const p = await fetchPaper();
      if (cancelled) return;
      setLoading(false);
      if (p) {
        scheduleNextPoll(0, p);
      }
    })();

    return () => {
      cancelled = true;
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, [fetchPaper, scheduleNextPoll]);

  // ── Render states ──────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div style={{ padding: "48px 52px", maxWidth: 1100 }}>
        {[70, 40, 90, 55].map((w, i) => (
          <div key={i} className="skeleton" style={{ width: `${w}%`, height: i === 0 ? 28 : 14, marginBottom: 14, borderRadius: 4 }} />
        ))}
      </div>
    );
  }

  if (error || !paper) {
    return (
      <div style={{ padding: "48px 52px", maxWidth: 860 }}>
        <div style={{
          padding: "24px", borderRadius: 8,
          background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)",
          display: "flex", alignItems: "center", gap: 16,
        }}>
          <span style={{ fontSize: "1.2rem" }}>⚠️</span>
          <div>
            <p style={{ fontSize: "0.9rem", color: "#DC2626", fontWeight: 600 }}>
              {error || "Paper not found."}
            </p>
            <Link href="/papers" style={{
              marginTop: 8, display: "inline-block", fontSize: "0.8rem",
              color: "var(--navy)", textDecoration: "underline",
            }}>
              ← Back to Library
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const isProcessing = paper.status === "processing" || (paper.status === "ready" && !hasAiData(paper));
  const rs  = paper.research_summary || {};
  const rg  = paper.research_gap     || {};

  return (
    <div style={{ padding: "48px 52px", maxWidth: 1100 }}>

      {/* ── Breadcrumb ── */}
      <div style={{ marginBottom: 24 }}>
        <Link href="/papers" style={{
          fontFamily: "'DM Mono', monospace",
          fontSize: "0.7rem",
          color: "var(--text-secondary)",
          textDecoration: "none",
          display: "inline-flex",
          alignItems: "center",
          gap: 5,
        }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Library
        </Link>
      </div>

      {/* ── Header ── */}
      <div style={{ marginBottom: 28 }}>
        <p style={{
          fontFamily: "'DM Mono', monospace", fontSize: "0.65rem",
          letterSpacing: "0.14em", textTransform: "uppercase",
          color: "var(--amber)", marginBottom: 8,
        }}>
          Paper Detail
        </p>
        <h1 style={{
          fontSize: "1.75rem", fontFamily: "'DM Serif Display', serif",
          fontWeight: 400, color: "var(--navy)", letterSpacing: "-0.03em",
          lineHeight: 1.25, marginBottom: 12,
        }}>
          {paper.title}
        </h1>

        {paper.authors?.length > 0 && (
          <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", fontStyle: "italic", marginBottom: 10 }}>
            {paper.authors.slice(0, 6).join(", ")}
            {paper.authors.length > 6 ? ` +${paper.authors.length - 6} more` : ""}
          </p>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.68rem", color: "var(--muted)" }}>
            {new Date(paper.upload_date).toLocaleDateString("en-US", {
              month: "long", day: "numeric", year: "numeric",
            })}
          </span>
          {paper.status === "failed" && (
            <span style={{
              fontFamily: "'DM Mono', monospace", fontSize: "0.62rem",
              color: "#DC2626", background: "rgba(239,68,68,0.08)",
              border: "1px solid rgba(239,68,68,0.2)",
              padding: "2px 8px", borderRadius: 3,
            }}>
              AI Failed
            </span>
          )}
          {isProcessing && paper.status !== "failed" && (
            <span style={{
              fontFamily: "'DM Mono', monospace", fontSize: "0.62rem",
              color: "var(--amber)", background: "var(--amber-dim)",
              border: "1px solid rgba(200,146,42,0.25)",
              padding: "2px 8px", borderRadius: 3,
            }}>
              Processing…
            </span>
          )}
        </div>
      </div>

      {/* ── Action row ── */}
      <div style={{ display: "flex", gap: 10, marginBottom: 28, flexWrap: "wrap" }}>
        <Link
          href={`/dashboard?paper_id=${paper.paper_id}`}
          style={{
            padding: "9px 18px", background: "var(--navy)", color: "var(--cream)",
            fontSize: "0.8rem", fontWeight: 500, borderRadius: 6,
            textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 6,
          }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          Chat about this paper
        </Link>
        <Link
          href={`/compare?paper_ids=${paper.paper_id}`}
          style={{
            padding: "9px 18px", background: "var(--surface)", color: "var(--navy)",
            fontSize: "0.8rem", fontWeight: 500, borderRadius: 6,
            textDecoration: "none", border: "1px solid var(--border-strong)",
            display: "inline-flex", alignItems: "center", gap: 6,
          }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" />
          </svg>
          Compare
        </Link>
        <Link
          href={`/recommendations?paper_id=${paper.paper_id}`}
          style={{
            padding: "9px 18px", background: "var(--surface)", color: "var(--navy)",
            fontSize: "0.8rem", fontWeight: 500, borderRadius: 6,
            textDecoration: "none", border: "1px solid var(--border-strong)",
            display: "inline-flex", alignItems: "center", gap: 6,
          }}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          Discover related
        </Link>
      </div>

      {/* ── Two-column layout: main + sidebar ── */}
      <div style={{ display: "flex", gap: 36, alignItems: "flex-start" }}>

        {/* ── Left: main content ── */}
        <div style={{ flex: 1, minWidth: 0 }}>

          {/* Abstract */}
          {paper.abstract && (
            <div style={{
              padding: "16px 20px",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              marginBottom: 24,
            }}>
              <p style={{
                fontFamily: "'DM Mono', monospace", fontSize: "0.6rem",
                letterSpacing: "0.1em", textTransform: "uppercase",
                color: "var(--text-secondary)", marginBottom: 8,
              }}>
                Abstract
              </p>
              <p style={{ fontSize: "0.855rem", color: "var(--text)", lineHeight: 1.7 }}>
                {paper.abstract}
              </p>
            </div>
          )}

          {/* Processing banner */}
          {isProcessing && paper.status !== "failed" && (
            <ProcessingBanner llmStatus={paper.llm_status} />
          )}

          {/* AI Summary heading */}
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 18,
            paddingBottom: 10,
            borderBottom: "1px solid var(--border)",
          }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style={{ color: "var(--amber)", flexShrink: 0 }}>
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
            </svg>
            <p style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: "0.62rem",
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--navy)",
              fontWeight: 600,
            }}>
              AI Summary
            </p>
          </div>

          {/* AI Summary content */}
          {!isProcessing && !hasAiData(paper) && paper.status !== "failed" ? (
            <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
              No AI summary available. Try re-enriching this paper.
            </p>
          ) : isProcessing ? (
            <div>
              {[1, 2, 3].map((i) => (
                <div key={i} style={{ marginBottom: 20 }}>
                  <div className="skeleton" style={{ width: "30%", height: 10, marginBottom: 10, borderRadius: 4 }} />
                  <div className="skeleton" style={{ width: "100%", height: 13, marginBottom: 6, borderRadius: 4 }} />
                  <div className="skeleton" style={{ width: "90%", height: 13, marginBottom: 6, borderRadius: 4 }} />
                  <div className="skeleton" style={{ width: "75%", height: 13, borderRadius: 4 }} />
                </div>
              ))}
            </div>
          ) : (
            <>
              <SectionCard label="Problem Definition"      value={rs.problem_definition} />
              <SectionCard label="Background & Prior Work" value={rs.background_and_prior_work} />
              <SectionCard label="Methodology"             value={rs.methodology} />
              <SectionCard label="Technical Novelty"       value={rs.technical_novelty} />
              <SectionCard label="Experimental Setup"      value={rs.experimental_setup} />
              <SectionCard label="Results & Analysis"      value={rs.results_and_analysis} />
              <SectionCard label="Limitations"             value={rs.limitations} />
              <SectionCard label="Conclusion"              value={rs.conclusion} />

              {/* Research gap */}
              {(rg.gap_identification || rg.future_directions?.length) && (
                <div style={{
                  marginTop: 24,
                  padding: "20px",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                }}>
                  <p style={{
                    fontFamily: "'DM Mono', monospace", fontSize: "0.62rem",
                    letterSpacing: "0.12em", textTransform: "uppercase",
                    color: "var(--amber)", marginBottom: 12,
                  }}>
                    Research Gap Analysis
                  </p>
                  {rg.gap_identification && (
                    <p style={{ fontSize: "0.875rem", color: "var(--text)", lineHeight: 1.7, marginBottom: 12 }}>
                      {rg.gap_identification}
                    </p>
                  )}
                  {rg.remaining_limitations && (
                    <p style={{ fontSize: "0.875rem", color: "var(--text)", lineHeight: 1.7, marginBottom: 12 }}>
                      <strong>Remaining limitations:</strong> {rg.remaining_limitations}
                    </p>
                  )}
                  {rg.future_directions && rg.future_directions.length > 0 && (
                    <div>
                      <p style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: 8 }}>
                        Future directions:
                      </p>
                      <ul style={{ paddingLeft: 18, margin: 0 }}>
                        {rg.future_directions.map((d, i) => (
                          <li key={i} style={{ fontSize: "0.855rem", color: "var(--text)", lineHeight: 1.65, marginBottom: 4 }}>
                            {d}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* ── Right: entities sidebar ── */}
        <EntitiesSidebar paper={paper} isProcessing={isProcessing} />

      </div>{/* end two-column */}

      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
      `}</style>
    </div>
  );
}