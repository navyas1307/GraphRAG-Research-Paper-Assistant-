"use client";
import { useState, useEffect, useCallback } from "react";
import { papersApi } from "@/lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────

interface PaperMeta {
  paper_id: string;
  title: string;
  authors: string[];
  topics: string[];
  has_summary: boolean;
  status?: string;
}

interface CompressedPaper {
  paper_id: string;
  title: string;
  authors: string[];
  topics: string[];
  problem: string;
  methodology: string;
  results: string;
  methods: string[];
  datasets: string[];
  models: string[];
  metrics: string[];
}

interface GraphSignals {
  similarity_edges: {
    paper_a: string;
    paper_b: string;
    score: number;
    shared_topics: string[];
    shared_methods: string[];
  }[];
  shared_methods: string[];
  shared_datasets: string[];
  shared_topics: string[];
  shared_models: string[];
  unique_methods: Record<string, string[]>;
}

interface CompareResult {
  papers: CompressedPaper[];
  comparison_table: {
    problem: string[];
    methodology: string[];
    datasets: string[];
    results: string[];
  };
  key_differences: string[];
  problem_solution_relationship: string;
  strengths_vs_weaknesses: Record<string, { strengths: string[]; weaknesses: string[] }>;
  best_use_cases: Record<string, string>;
  research_evolution: string;
  graph_signals: GraphSignals;
  similarity_scores: Record<string, number> | null;
  source: string;
}

// ─── Small components ─────────────────────────────────────────────────────────

function Tag({ children, color = "default" }: { children: React.ReactNode; color?: "amber" | "navy" | "green" | "default" }) {
  const styles: Record<string, React.CSSProperties> = {
    amber: { color: "var(--amber)", background: "var(--amber-dim)", border: "1px solid rgba(200,146,42,0.25)" },
    navy:  { color: "var(--cream)", background: "var(--navy)", border: "1px solid var(--navy)" },
    green: { color: "#065F46", background: "rgba(6,95,70,0.08)", border: "1px solid rgba(6,95,70,0.2)" },
    default: { color: "var(--text-secondary)", background: "var(--surface-alt)", border: "1px solid var(--border-strong)" },
  };
  return (
    <span style={{
      display: "inline-flex", alignItems: "center",
      fontFamily: "'DM Mono', monospace", fontSize: "0.66rem",
      fontWeight: 500, letterSpacing: "0.04em",
      padding: "2px 8px", borderRadius: 3,
      ...styles[color],
    }}>
      {children}
    </span>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p style={{
      fontFamily: "'DM Mono', monospace", fontSize: "0.6rem",
      letterSpacing: "0.12em", textTransform: "uppercase",
      color: "var(--muted)", marginBottom: 12,
    }}>
      {children}
    </p>
  );
}

function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 8, padding: "20px 22px",
      ...style,
    }}>
      {children}
    </div>
  );
}

// ─── Graph signals panel ──────────────────────────────────────────────────────

function GraphSignalsPanel({
  signals,
  papers,
  simScores,
}: {
  signals: GraphSignals;
  papers: CompressedPaper[];
  simScores: Record<string, number> | null;
}) {
  const pidToTitle = Object.fromEntries(papers.map((p, i) => [p.paper_id, `P${i + 1}`]));

  const hasSomething =
    signals.similarity_edges.length > 0 ||
    signals.shared_methods.length > 0 ||
    signals.shared_datasets.length > 0 ||
    signals.shared_topics.length > 0 ||
    Object.values(signals.unique_methods).some((m) => m.length > 0);

  if (!hasSomething && !simScores) return null;

  return (
    <Card style={{ borderColor: "rgba(200,146,42,0.2)", background: "rgba(200,146,42,0.02)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{
          width: 28, height: 28, borderRadius: "50%",
          background: "var(--amber-dim)", border: "1px solid rgba(200,146,42,0.3)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
          </svg>
        </div>
        <div>
          <p style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--navy)" }}>Neo4j Graph Signals</p>
          <p style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: 1 }}>
            Entity overlap and similarity computed from knowledge graph
          </p>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

        {/* Similarity edges */}
        {signals.similarity_edges.map((edge, i) => {
          const pa = pidToTitle[edge.paper_a] || "P?";
          const pb = pidToTitle[edge.paper_b] || "P?";
          const pct = Math.round(edge.score * 100);
          const barColor = pct >= 70 ? "#065F46" : pct >= 40 ? "var(--amber)" : "#b42828";
          return (
            <div key={i}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                  <strong style={{ color: "var(--navy)" }}>{pa}</strong> ↔ <strong style={{ color: "var(--navy)" }}>{pb}</strong> semantic similarity
                </span>
                <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.76rem", color: barColor, fontWeight: 600 }}>
                  {pct}%
                </span>
              </div>
              {/* Progress bar */}
              <div style={{ height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden", marginBottom: 8 }}>
                <div style={{ height: "100%", width: `${pct}%`, background: barColor, borderRadius: 2, transition: "width 0.6s ease" }} />
              </div>
              {/* Shared tags from this edge */}
              {(edge.shared_topics.length > 0 || edge.shared_methods.length > 0) && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {edge.shared_topics.slice(0, 3).map((t) => (
                    <Tag key={t} color="amber">{t}</Tag>
                  ))}
                  {edge.shared_methods.slice(0, 3).map((m) => (
                    <Tag key={m}>{m}</Tag>
                  ))}
                </div>
              )}
            </div>
          );
        })}

        {/* Shared entity rows */}
        {[
          { label: "Shared Techniques", items: signals.shared_methods,  color: "default" as const },
          { label: "Shared Datasets",   items: signals.shared_datasets, color: "green"   as const },
          { label: "Shared Topics",     items: signals.shared_topics,   color: "amber"   as const },
          { label: "Shared Models",     items: signals.shared_models,   color: "navy"    as const },
        ].map(({ label, items, color }) =>
          items.length > 0 ? (
            <div key={label}>
              <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.6rem", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
                {label}
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {items.map((item) => <Tag key={item} color={color}>{item}</Tag>)}
              </div>
            </div>
          ) : null
        )}

        {/* Unique methods per paper */}
        {Object.entries(signals.unique_methods).map(([pid, methods]) => {
          if (!methods.length) return null;
          const label = pidToTitle[pid] || "P?";
          return (
            <div key={pid}>
              <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.6rem", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
                Unique to {label}
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {methods.map((m) => <Tag key={m}>{m}</Tag>)}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ─── Comparison table ─────────────────────────────────────────────────────────

function ComparisonTable({
  table,
  papers,
}: {
  table: CompareResult["comparison_table"];
  papers: CompressedPaper[];
}) {
  const rows: { key: keyof typeof table; label: string }[] = [
    { key: "problem",     label: "Problem" },
    { key: "methodology", label: "Methodology" },
    { key: "datasets",    label: "Datasets" },
    { key: "results",     label: "Key Results" },
  ];

  const PAPER_COLORS = ["#1E3A5F", "#065F46", "#501464"];

  return (
    <Card>
      <SectionLabel>Comparison Table</SectionLabel>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
          <thead>
            <tr>
              <th style={{
                width: 110, textAlign: "left", padding: "8px 12px 8px 0",
                fontFamily: "'DM Mono', monospace", fontSize: "0.6rem",
                letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)",
                borderBottom: "1px solid var(--border)",
              }}>
                Dimension
              </th>
              {papers.map((p, i) => (
                <th key={p.paper_id} style={{
                  textAlign: "left", padding: "8px 12px",
                  fontWeight: 600, fontSize: "0.78rem",
                  color: PAPER_COLORS[i] || "var(--navy)",
                  borderBottom: "1px solid var(--border)",
                  borderLeft: "1px solid var(--border)",
                }}>
                  <span style={{
                    display: "inline-block", fontFamily: "'DM Mono', monospace",
                    fontSize: "0.62rem", background: PAPER_COLORS[i] || "var(--navy)",
                    color: "#fff", padding: "1px 6px", borderRadius: 2, marginBottom: 3,
                  }}>
                    P{i + 1}
                  </span>
                  <br />
                  {p.title.length > 50 ? p.title.slice(0, 50) + "…" : p.title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ key, label }) => (
              <tr key={key}>
                <td style={{
                  padding: "10px 12px 10px 0", verticalAlign: "top",
                  fontFamily: "'DM Mono', monospace", fontSize: "0.62rem",
                  color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em",
                  borderBottom: "1px solid var(--border)",
                }}>
                  {label}
                </td>
                {papers.map((p, i) => (
                  <td key={p.paper_id} style={{
                    padding: "10px 12px", verticalAlign: "top",
                    color: "var(--text)", lineHeight: 1.6,
                    borderBottom: "1px solid var(--border)",
                    borderLeft: "1px solid var(--border)",
                  }}>
                    {table[key]?.[i] || "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// ─── Strengths & weaknesses ───────────────────────────────────────────────────

function SwPanel({
  sw,
  papers,
}: {
  sw: Record<string, { strengths: string[]; weaknesses: string[] }>;
  papers: CompressedPaper[];
}) {
  const PAPER_COLORS = ["#1E3A5F", "#065F46", "#501464"];
  return (
    <Card>
      <SectionLabel>Strengths & Weaknesses</SectionLabel>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${papers.length}, 1fr)`, gap: 16 }}>
        {papers.map((p, i) => {
          const key = `paper${i + 1}`;
          const data = sw[key] || { strengths: [], weaknesses: [] };
          const color = PAPER_COLORS[i] || "var(--navy)";
          return (
            <div key={p.paper_id}>
              <div style={{
                display: "flex", alignItems: "center", gap: 8, marginBottom: 12,
                paddingBottom: 10, borderBottom: `2px solid ${color}20`,
              }}>
                <span style={{
                  fontFamily: "'DM Mono', monospace", fontSize: "0.62rem",
                  background: color, color: "#fff", padding: "2px 8px", borderRadius: 2,
                }}>P{i + 1}</span>
                <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--navy)" }}>
                  {p.title.length > 36 ? p.title.slice(0, 36) + "…" : p.title}
                </span>
              </div>

              {data.strengths.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.58rem", color: "#065F46", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
                    ✓ Strengths
                  </p>
                  {data.strengths.map((s, j) => (
                    <div key={j} style={{ display: "flex", gap: 6, marginBottom: 5 }}>
                      <span style={{ color: "#065F46", fontSize: "0.7rem", marginTop: 1, flexShrink: 0 }}>▸</span>
                      <span style={{ fontSize: "0.78rem", color: "var(--text)", lineHeight: 1.55 }}>{s}</span>
                    </div>
                  ))}
                </div>
              )}

              {data.weaknesses.length > 0 && (
                <div>
                  <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.58rem", color: "#b42828", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
                    ✗ Weaknesses
                  </p>
                  {data.weaknesses.map((w, j) => (
                    <div key={j} style={{ display: "flex", gap: 6, marginBottom: 5 }}>
                      <span style={{ color: "#b42828", fontSize: "0.7rem", marginTop: 1, flexShrink: 0 }}>▸</span>
                      <span style={{ fontSize: "0.78rem", color: "var(--text)", lineHeight: 1.55 }}>{w}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ─── Source badge ─────────────────────────────────────────────────────────────

function SourceBadge({ source }: { source: string }) {
  const map: Record<string, { label: string; color: string; bg: string }> = {
    graphrag_llm:        { label: "GraphRAG + LLM",       color: "#065F46", bg: "rgba(6,95,70,0.08)" },
    graphrag_structural: { label: "GraphRAG Structural",   color: "var(--amber)", bg: "var(--amber-dim)" },
    graphrag_cached:     { label: "GraphRAG (cached)",     color: "var(--navy-400)", bg: "var(--surface-alt)" },
  };
  const s = map[source] || { label: source, color: "var(--muted)", bg: "var(--surface-alt)" };
  return (
    <span style={{
      fontFamily: "'DM Mono', monospace", fontSize: "0.62rem", fontWeight: 500,
      letterSpacing: "0.04em", padding: "3px 9px", borderRadius: 3,
      color: s.color, background: s.bg,
      border: `1px solid ${s.color}30`,
    }}>
      {s.label}
    </span>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function ComparePage() {
  const [library, setLibrary] = useState<PaperMeta[]>([]);
  const [libLoading, setLibLoading] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);

  const [result, setResult] = useState<CompareResult | null>(null);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState("");

  // Load library once
  useEffect(() => {
    papersApi.list(0, 50)
      .then((r) => setLibrary(r.data))
      .catch(() => setError("Could not load library."))
      .finally(() => setLibLoading(false));
  }, []);

  const toggleSelect = (id: string) => {
    setSelected((prev) =>
      prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length < 3
        ? [...prev, id]
        : prev
    );
    // Reset result when selection changes
    setResult(null);
    setError("");
  };

  const runCompare = useCallback(async () => {
    if (selected.length < 2) return;
    setComparing(true);
    setError("");
    setResult(null);
    try {
      const res = await papersApi.compareDeep(selected);
      setResult(res.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || "Comparison failed. Please try again.");
    } finally {
      setComparing(false);
    }
  }, [selected]);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
      {/* Page header */}
      <div style={{
        background: "var(--surface)", borderBottom: "1px solid var(--border)",
        padding: "20px 40px",
      }}>
        <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.65rem", letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--amber)", marginBottom: 6 }}>
          GraphRAG Compare
        </p>
        <h1 style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.7rem", fontWeight: 400, color: "var(--navy)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
          Compare Papers
        </h1>
        <p style={{ marginTop: 6, fontSize: "0.84rem", color: "var(--text-secondary)" }}>
          Select 2–3 papers. The comparison uses Neo4j entity signals and Qdrant semantic chunks — not just LLM guessing.
        </p>
      </div>

      <div style={{ padding: "32px 40px", maxWidth: 1100, margin: "0 auto" }}>

        {/* Paper selector */}
        <Card style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <div>
              <SectionLabel>Step 1 — Select Papers</SectionLabel>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: -8 }}>
                Choose 2 or 3 papers to compare
                {selected.length > 0 && (
                  <span style={{ color: "var(--amber)", fontWeight: 600 }}> · {selected.length} selected</span>
                )}
              </p>
            </div>
            {selected.length >= 2 && (
              <button
                onClick={runCompare}
                disabled={comparing}
                style={{
                  padding: "10px 22px",
                  background: comparing ? "#9CA3AF" : "var(--navy)",
                  color: "#fff", border: "none", borderRadius: 6,
                  fontSize: "0.84rem", fontWeight: 600, cursor: comparing ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", gap: 8, transition: "background 0.15s",
                }}
              >
                {comparing ? (
                  <>
                    <div style={{
                      width: 12, height: 12, border: "2px solid rgba(255,255,255,0.3)",
                      borderTopColor: "#fff", borderRadius: "50%",
                      animation: "spin 0.7s linear infinite",
                    }} />
                    Comparing…
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
                    </svg>
                    Run GraphRAG Compare
                  </>
                )}
              </button>
            )}
          </div>

          {libLoading ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {[1, 2, 3].map((i) => (
                <div key={i} className="skeleton" style={{ height: 52, borderRadius: 6 }} />
              ))}
            </div>
          ) : library.length === 0 ? (
            <div style={{ textAlign: "center", padding: "32px 0", color: "var(--muted)", fontSize: "0.85rem" }}>
              No papers in library. <a href="/upload" style={{ color: "var(--amber)", textDecoration: "none" }}>Upload papers</a> to compare.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {library.map((p) => {
                const isSelected = selected.includes(p.paper_id);
                const selIdx = selected.indexOf(p.paper_id);
                const isDisabled = !isSelected && selected.length >= 3;
                const COLORS = ["#1E3A5F", "#065F46", "#501464"];
                const selColor = selIdx >= 0 ? COLORS[selIdx] : undefined;

                return (
                  <button
                    key={p.paper_id}
                    onClick={() => !isDisabled && toggleSelect(p.paper_id)}
                    disabled={isDisabled}
                    style={{
                      width: "100%", padding: "12px 16px", textAlign: "left",
                      border: `1.5px solid ${isSelected ? (selColor || "var(--navy)") : "var(--border)"}`,
                      borderRadius: 6,
                      background: isSelected ? `${selColor || "var(--navy)"}0a` : "var(--surface)",
                      cursor: isDisabled ? "not-allowed" : "pointer",
                      opacity: isDisabled ? 0.45 : 1,
                      transition: "all 0.12s",
                      display: "flex", alignItems: "center", gap: 14,
                    }}
                  >
                    {/* Checkbox */}
                    <div style={{
                      width: 20, height: 20, borderRadius: 4, flexShrink: 0,
                      border: `2px solid ${isSelected ? (selColor || "var(--navy)") : "var(--border-strong)"}`,
                      background: isSelected ? (selColor || "var(--navy)") : "transparent",
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                      {isSelected && (
                        selIdx >= 0 ? (
                          <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.6rem", color: "#fff", fontWeight: 700 }}>
                            P{selIdx + 1}
                          </span>
                        ) : (
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        )
                      )}
                    </div>

                    {/* Paper info */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{ fontSize: "0.85rem", fontWeight: 600, color: isSelected ? (selColor || "var(--navy)") : "var(--navy)", lineHeight: 1.3, marginBottom: 3 }}>
                        {p.title}
                      </p>
                      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                        {p.authors?.slice(0, 2).map((a: string) => (
                          <span key={a} style={{ fontSize: "0.71rem", color: "var(--text-secondary)", fontStyle: "italic" }}>{a}</span>
                        ))}
                        {p.topics?.slice(0, 2).map((t: string) => (
                          <Tag key={t}>{t}</Tag>
                        ))}
                        {p.has_summary && <Tag color="amber">AI Ready</Tag>}
                        {p.status === "processing" && <Tag color="amber">Processing…</Tag>}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </Card>

        {/* Error */}
        {error && (
          <div style={{
            marginBottom: 24, padding: "12px 16px", borderRadius: 6,
            background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)",
            fontSize: "0.83rem", color: "#DC2626",
          }}>
            {error}
          </div>
        )}

        {/* Comparing spinner */}
        {comparing && (
          <Card style={{ textAlign: "center", padding: "40px 24px", marginBottom: 24 }}>
            <div style={{
              width: 36, height: 36, border: "3px solid var(--border)", borderTopColor: "var(--navy)",
              borderRadius: "50%", animation: "spin 0.7s linear infinite", margin: "0 auto 16px",
            }} />
            <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.1rem", color: "var(--navy)", marginBottom: 6 }}>
              Running GraphRAG comparison…
            </p>
            <p style={{ fontSize: "0.82rem", color: "var(--muted)" }}>
              Querying Neo4j → Qdrant → LLM
            </p>
          </Card>
        )}

        {/* Results */}
        {result && !comparing && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20, animation: "fadeUp 0.4s ease" }}>

            {/* Result header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div>
                <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.2rem", color: "var(--navy)", letterSpacing: "-0.02em" }}>
                  Comparison Results
                </p>
                <p style={{ fontSize: "0.76rem", color: "var(--muted)", marginTop: 3 }}>
                  {result.papers.length} papers analysed
                </p>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <SourceBadge source={result.source} />
                <button
                  onClick={runCompare}
                  style={{
                    padding: "6px 14px", fontSize: "0.76rem", color: "var(--text-secondary)",
                    border: "1px solid var(--border-strong)", borderRadius: 5,
                    background: "var(--surface)", cursor: "pointer",
                  }}
                >
                  Refresh
                </button>
              </div>
            </div>

            {/* Graph signals — first, as they are ground truth */}
            {result.graph_signals && (
              <GraphSignalsPanel
                signals={result.graph_signals}
                papers={result.papers}
                simScores={result.similarity_scores}
              />
            )}

            {/* Comparison table */}
            {result.comparison_table && (
              <ComparisonTable table={result.comparison_table} papers={result.papers} />
            )}

            {/* Key differences */}
            {result.key_differences?.length > 0 && (
              <Card>
                <SectionLabel>Key Technical Differences</SectionLabel>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {result.key_differences.map((diff, i) => (
                    <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                      <div style={{
                        width: 20, height: 20, borderRadius: "50%", flexShrink: 0, marginTop: 1,
                        background: "var(--navy)", display: "flex", alignItems: "center", justifyContent: "center",
                      }}>
                        <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.55rem", color: "var(--amber)" }}>
                          {i + 1}
                        </span>
                      </div>
                      <p style={{ fontSize: "0.84rem", color: "var(--text)", lineHeight: 1.6 }}>{diff}</p>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Problem→Solution relationship */}
            {result.problem_solution_relationship && (
              <Card>
                <SectionLabel>Problem → Solution Relationship</SectionLabel>
                <p style={{ fontSize: "0.85rem", color: "var(--text)", lineHeight: 1.75 }}>
                  {result.problem_solution_relationship}
                </p>
              </Card>
            )}

            {/* Strengths & weaknesses */}
            {result.strengths_vs_weaknesses && Object.keys(result.strengths_vs_weaknesses).length > 0 && (
              <SwPanel sw={result.strengths_vs_weaknesses} papers={result.papers} />
            )}

            {/* Best use cases */}
            {result.best_use_cases && Object.keys(result.best_use_cases).length > 0 && (
              <Card>
                <SectionLabel>Best Use Cases</SectionLabel>
                <div style={{ display: "grid", gridTemplateColumns: `repeat(${result.papers.length}, 1fr)`, gap: 16 }}>
                  {result.papers.map((p, i) => {
                    const key = `paper${i + 1}`;
                    const uc = result.best_use_cases[key] || "";
                    const COLORS = ["#1E3A5F", "#065F46", "#501464"];
                    const color = COLORS[i] || "var(--navy)";
                    return (
                      <div key={p.paper_id} style={{
                        padding: "14px 16px", borderRadius: 6,
                        background: `${color}06`, border: `1px solid ${color}20`,
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                          <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.62rem", background: color, color: "#fff", padding: "1px 7px", borderRadius: 2 }}>
                            P{i + 1}
                          </span>
                          <span style={{ fontSize: "0.78rem", fontWeight: 600, color, lineHeight: 1.3 }}>
                            {p.title.length > 38 ? p.title.slice(0, 38) + "…" : p.title}
                          </span>
                        </div>
                        <p style={{ fontSize: "0.81rem", color: "var(--text)", lineHeight: 1.65 }}>{uc}</p>
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}

            {/* Research evolution */}
            {result.research_evolution && (
              <Card style={{ borderColor: "rgba(13,27,42,0.15)", background: "var(--navy)" }}>
                <SectionLabel>
                  <span style={{ color: "rgba(250,248,244,0.4)" }}>Research Evolution</span>
                </SectionLabel>
                <p style={{ fontSize: "0.85rem", color: "var(--cream)", lineHeight: 1.75 }}>
                  {result.research_evolution}
                </p>
              </Card>
            )}

          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}