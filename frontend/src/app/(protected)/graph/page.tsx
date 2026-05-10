"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { graphApi, papersApi } from "@/lib/api";
import api from "@/lib/api";
import * as d3 from "d3";

// ── Constants ─────────────────────────────────────────────────────────────────

const NODE_COLORS: Record<string, string> = {
  Paper:   "#0d1b2a",
  Method:  "#c8922a",
  Dataset: "#0f523c",
  Model:   "#501464",
  Topic:   "#b42828",
  Author:  "#4a5a72",
};

const EDGE_STYLES: Record<string, { dash: string; color: string; width: number; label: string }> = {
  SIMILAR_TO:    { dash: "none",  color: "#c8922a", width: 2.2, label: "similarity" },
  USES_METHOD:   { dash: "6,3",  color: "#0f523c", width: 1.2, label: "method" },
  USES_DATASET:  { dash: "2,4",  color: "#501464", width: 1.2, label: "dataset" },
  PROPOSES_MODEL:{ dash: "8,2",  color: "#b42828", width: 1.4, label: "model" },
  HAS_TOPIC:     { dash: "none", color: "rgba(13,27,42,0.12)", width: 1, label: "topic" },
  AUTHORED_BY:   { dash: "none", color: "rgba(13,27,42,0.07)", width: 0.8, label: "author" },
};

const LABELS = ["All", "Paper", "Method", "Dataset", "Model", "Topic"];

// ── Cluster detection (connected components on Paper→Paper edges) ──────────────

function detectClusters(nodes: any[], edges: any[]): Map<string, number> {
  const paperIds = new Set(nodes.filter(n => n.label === "Paper").map(n => n.id));
  const adj: Map<string, Set<string>> = new Map();
  paperIds.forEach(id => adj.set(id, new Set()));

  edges.forEach(e => {
    const s = typeof e.source === "object" ? e.source.id : e.source;
    const t = typeof e.target === "object" ? e.target.id : e.target;
    if (paperIds.has(s) && paperIds.has(t)) {
      adj.get(s)!.add(t);
      adj.get(t)!.add(s);
    }
  });

  const cluster = new Map<string, number>();
  let cid = 0;
  paperIds.forEach(id => {
    if (!cluster.has(id)) {
      const stack = [id];
      while (stack.length) {
        const cur = stack.pop()!;
        if (cluster.has(cur)) continue;
        cluster.set(cur, cid);
        adj.get(cur)?.forEach(nb => { if (!cluster.has(nb)) stack.push(nb); });
      }
      cid++;
    }
  });

  // Assign non-paper nodes to the cluster of the paper they're connected to most
  nodes.filter(n => n.label !== "Paper").forEach(n => {
    const connected = edges
      .filter(e => {
        const s = typeof e.source === "object" ? e.source.id : e.source;
        const t = typeof e.target === "object" ? e.target.id : e.target;
        return s === n.id || t === n.id;
      })
      .map(e => {
        const s = typeof e.source === "object" ? e.source.id : e.source;
        const t = typeof e.target === "object" ? e.target.id : e.target;
        return s === n.id ? t : s;
      })
      .find(id => cluster.has(id));
    if (connected !== undefined) cluster.set(n.id, cluster.get(connected)!);
    else cluster.set(n.id, cid++);
  });

  return cluster;
}

// ── Cluster label from top keywords ──────────────────────────────────────────
//
// FIX: previously counted edge.shared_topics/shared_methods from ALL edges
// regardless of which cluster they belonged to. Edges cross cluster boundaries,
// so every cluster scored the same global keywords (e.g. "gender equality,
// women empowerment" appearing on all three clusters when only one contained
// that paper).
//
// Fix strategy (priority order):
//   1. Use cluster_label already computed by the backend and stored on Paper
//      nodes in Neo4j — coherence-filtered, per-cluster, correct.
//   2. Fallback: count entity node names INSIDE this cluster only.
//   3. Final fallback: title-derived keywords from papers in this cluster.

function getClusterKeywords(clusterNodes: any[], edges: any[]): string {
  // PRIORITY 1: Use cluster_label from any Paper node in this cluster.
  // The backend writes cluster_label onto every Paper node in Neo4j via
  // write_clusters_to_neo4j(). It is already coherence-filtered (keywords
  // shared by >=50% of cluster papers only).
  const paperNode = clusterNodes.find(n => n.label === "Paper" && n.cluster_label);
  if (paperNode?.cluster_label) {
    return paperNode.cluster_label;
  }

  // PRIORITY 2: Count entity node names INSIDE this cluster only.
  // Do NOT use edge shared_topics/shared_methods -- those are edge-level
  // and bleed across cluster boundaries.
  const nodeIds = new Set(clusterNodes.map(n => n.id));
  const keywords: Record<string, number> = {};
  clusterNodes
    .filter(n => n.label !== "Paper" && n.label !== "Author" && n.name)
    .forEach(n => { keywords[n.name] = (keywords[n.name] || 0) + 1; });

  // Count intra-cluster edges only (both endpoints inside this cluster).
  edges.forEach(e => {
    const s = typeof e.source === "object" ? e.source.id : e.source;
    const t = typeof e.target === "object" ? e.target.id : e.target;
    if (!nodeIds.has(s) || !nodeIds.has(t)) return; // skip cross-cluster edges
    if (e.shared_topics)  e.shared_topics.forEach((kw: string)  => { keywords[kw] = (keywords[kw] || 0) + 1; });
    if (e.shared_methods) e.shared_methods.forEach((kw: string) => { keywords[kw] = (keywords[kw] || 0) + 1; });
  });

  const top = Object.entries(keywords).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k]) => k);
  if (top.length) return top.join(", ");

  // PRIORITY 3: Derive from the paper titles in this cluster.
  const stopWords = new Set(["a","an","the","of","in","on","for","and","with","using","from","via","this","that","are","its"]);
  const titleWords: Record<string, number> = {};
  clusterNodes.filter(n => n.label === "Paper" && n.name).forEach(n => {
    (n.name as string).split(/\s+/).forEach(w => {
      const wl = w.toLowerCase().replace(/[^a-z]/g, "");
      if (wl.length > 4 && !stopWords.has(wl)) titleWords[wl] = (titleWords[wl] || 0) + 1;
    });
  });
  const titleTop = Object.entries(titleWords).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k]) => k);
  return titleTop.length ? titleTop.join(", ") : "General Research";
}

// ── Cluster colors (distinct, readable) ───────────────────────────────────────

const CLUSTER_PALETTE = [
  "#1d6fbf", "#b35900", "#1a7a4a", "#6b2d8b",
  "#b5291b", "#2d7a7a", "#7a5c1a", "#3d3d8f",
];
function clusterColor(id: number) { return CLUSTER_PALETTE[id % CLUSTER_PALETTE.length]; }

// ── Main Component ─────────────────────────────────────────────────────────────

export default function GraphPage() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [graphData, setGraphData] = useState<{ nodes: any[]; edges: any[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<any>(null);
  const [selectedEdge, setSelectedEdge] = useState<any>(null);
  const [filter, setFilter] = useState<string>("All");
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [searchQuery, setSearchQuery] = useState("");
  const [searchFocus, setSearchFocus] = useState<string | null>(null);
  const [focusNode, setFocusNode] = useState<string | null>(null); // focus mode: id of clicked paper
  const [viewMode, setViewMode] = useState<"graph" | "cluster">("graph");
  const [clusterMap, setClusterMap] = useState<Map<string, number>>(new Map());
  const [clusters, setClusters] = useState<Map<number, any[]>>(new Map());
  // LLM cluster summaries: clusterId → summary string. Fetched on-demand, cached.
  const [clusterSummaries, setClusterSummaries] = useState<Record<number, string>>({});
  const [summaryLoading, setSummaryLoading] = useState<Record<number, boolean>>({});
  const [paperDetails, setPaperDetails] = useState<Record<string, any>>({});
  const simulationRef = useRef<any>(null);
  const d3Ref = useRef<any>(null);

  // ── Load graph ───────────────────────────────────────────────────────────────

  useEffect(() => {
    graphApi.full()
      .then((r) => {
        setGraphData(r.data);
        const c: Record<string, number> = {};
        r.data.nodes.forEach((n: any) => { c[n.label] = (c[n.label] || 0) + 1; });
        setCounts(c);

        const cm = detectClusters(r.data.nodes, r.data.edges);
        setClusterMap(cm);

        const clusterGroups = new Map<number, any[]>();
        cm.forEach((cid, nodeId) => {
          const node = r.data.nodes.find((n: any) => n.id === nodeId);
          if (node) {
            if (!clusterGroups.has(cid)) clusterGroups.set(cid, []);
            clusterGroups.get(cid)!.push(node);
          }
        });
        setClusters(clusterGroups);
      })
      .catch(() => setError("Could not load graph. Is Neo4j running?"))
      .finally(() => setLoading(false));
  }, []);

  // ── Fetch paper detail for insight panel (on-demand, cached) ─────────────────

  const fetchPaperDetail = useCallback(async (paperId: string) => {
    if (paperDetails[paperId]) return;
    try {
      const r = await papersApi.get(paperId);
      setPaperDetails(prev => ({ ...prev, [paperId]: r.data }));
    } catch { /* silently skip */ }
  }, [paperDetails]);

  useEffect(() => {
    if (selected?.label === "Paper" && selected.id) {
      fetchPaperDetail(selected.id);
    }
  }, [selected]);

  // ── LLM cluster summary: on-demand, cached per clusterId ─────────────────────
  // Strategy: send ONLY paper titles + top-3 keywords to LLM (tiny prompt → no rate limit pressure).
  // Uses /api/query (existing endpoint) with a synthetic question, caches result in state.

  const fetchClusterSummary = useCallback(async (clusterId: number) => {
    if (clusterSummaries[clusterId] !== undefined) return;
    if (summaryLoading[clusterId]) return;

    const clusterNodes = clusters.get(clusterId) || [];
    const paperNodes = clusterNodes.filter(n => n.label === "Paper");
    if (paperNodes.length === 0) return;

    setSummaryLoading(prev => ({ ...prev, [clusterId]: true }));

    try {
      // Build a compact prompt from just titles (no abstract text → saves tokens)
      const titles = paperNodes.slice(0, 8).map(n => `- ${n.name}`).join("\n");
      const keywords = getClusterKeywords(clusterNodes, graphData?.edges || []);

      // Use existing /api/query endpoint — no new backend needed
      const res = await api.post("/api/query", {
        question: `In 1-2 sentences, what research theme connects these papers? Papers:\n${titles}\nTop keywords: ${keywords}. Be concise.`,
        paper_id: null,
        chat_history: [],
      });

      const summary = res.data?.answer || "Research cluster";
      setClusterSummaries(prev => ({ ...prev, [clusterId]: summary }));
    } catch {
      setClusterSummaries(prev => ({ ...prev, [clusterId]: "Could not generate summary." }));
    } finally {
      setSummaryLoading(prev => ({ ...prev, [clusterId]: false }));
    }
  }, [clusters, clusterSummaries, summaryLoading, graphData]);

  // ── Render graph (D3) ────────────────────────────────────────────────────────

  useEffect(() => {
    if (!graphData || !svgRef.current || viewMode !== "graph") return;
    renderGraph();
  }, [graphData, filter, focusNode, searchFocus, viewMode, clusterMap]);

  const renderGraph = () => {
    // d3 is statically imported at the top of the file
    const svg = d3.select(svgRef.current!);
    svg.selectAll("*").remove();

    const width = svgRef.current!.clientWidth || 900;
    const height = svgRef.current!.clientHeight || 640;

    // Filter nodes
    let nodes = filter === "All"
      ? graphData!.nodes.map(n => ({ ...n }))
      : graphData!.nodes.filter(n => n.label === "Paper" || n.label === filter).map(n => ({ ...n }));
    const nodeIds = new Set(nodes.map((n: any) => n.id));
    let edges = graphData!.edges.filter((e: any) => {
      const s = typeof e.source === "object" ? e.source.id : e.source;
      const t = typeof e.target === "object" ? e.target.id : e.target;
      return nodeIds.has(s) && nodeIds.has(t);
    }).map(e => ({ ...e }));

    // Compute node degree (connection count)
    const degree: Record<string, number> = {};
    edges.forEach(e => {
      const s = typeof e.source === "object" ? e.source.id : e.source;
      const t = typeof e.target === "object" ? e.target.id : e.target;
      degree[s] = (degree[s] || 0) + 1;
      degree[t] = (degree[t] || 0) + 1;
    });

    // Base sizes by label, scaled by degree
    const baseSize: Record<string, number> = { Paper: 14, Method: 9, Dataset: 9, Model: 9, Topic: 7, Author: 6 };
    const nodeRadius = (d: any) => {
      const base = baseSize[d.label] || 8;
      const deg = degree[d.id] || 0;
      return d.label === "Paper" ? Math.max(base, Math.min(base + deg * 2, 32)) : base;
    };

    // Alpha (opacity) for focus mode
    const getNodeOpacity = (d: any) => {
      if (!focusNode) return 1;
      if (d.id === focusNode) return 1;
      const connected = edges.some(e => {
        const s = typeof e.source === "object" ? e.source.id : e.source;
        const t = typeof e.target === "object" ? e.target.id : e.target;
        return (s === focusNode && t === d.id) || (t === focusNode && s === d.id);
      });
      return connected ? 0.9 : 0.12;
    };

    const getEdgeOpacity = (e: any) => {
      if (!focusNode) return e.type === "SIMILAR_TO" ? 0.75 : 0.6;
      const s = typeof e.source === "object" ? e.source.id : e.source;
      const t = typeof e.target === "object" ? e.target.id : e.target;
      return (s === focusNode || t === focusNode) ? 0.9 : 0.05;
    };

    // Search highlight
    const searchHighlight = (d: any) => {
      if (!searchQuery) return false;
      return (d.name || "").toLowerCase().includes(searchQuery.toLowerCase());
    };

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id((d: any) => d.id).distance((e: any) => {
        // Stronger separation for similarity edges
        return e.type === "SIMILAR_TO" ? 140 : 90;
      }).strength((e: any) => e.type === "SIMILAR_TO" ? 0.5 : 0.3))
      .force("charge", d3.forceManyBody().strength((d: any) => d.label === "Paper" ? -350 : -120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide((d: any) => nodeRadius(d) + 14))
      // Cluster separation: push papers of different clusters apart
      .force("clusterX", d3.forceX((d: any) => {
        const cid = clusterMap.get(d.id);
        if (cid === undefined) return width / 2;
        const total = Array.from(clusterMap.values()).filter((v, i, a) => a.indexOf(v) === i).length;
        const angle = (cid / total) * 2 * Math.PI;
        return width / 2 + Math.cos(angle) * (width * 0.28);
      }).strength(0.08))
      .force("clusterY", d3.forceY((d: any) => {
        const cid = clusterMap.get(d.id);
        if (cid === undefined) return height / 2;
        const total = Array.from(clusterMap.values()).filter((v, i, a) => a.indexOf(v) === i).length;
        const angle = (cid / total) * 2 * Math.PI;
        return height / 2 + Math.sin(angle) * (height * 0.28);
      }).strength(0.08));

    simulationRef.current = simulation;

    const g = svg.append("g");

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 6])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    // ── Edges ──
    const link = g.append("g").selectAll("line").data(edges).join("line")
      .attr("stroke", (d: any) => (EDGE_STYLES[d.type] || EDGE_STYLES["HAS_TOPIC"]).color)
      .attr("stroke-width", (d: any) => (EDGE_STYLES[d.type] || EDGE_STYLES["HAS_TOPIC"]).width)
      .attr("stroke-dasharray", (d: any) => (EDGE_STYLES[d.type] || EDGE_STYLES["HAS_TOPIC"]).dash)
      .attr("opacity", getEdgeOpacity)
      .attr("cursor", (d: any) => d.type === "SIMILAR_TO" ? "pointer" : "default")
      .on("click", (_: any, d: any) => {
        if (d.type === "SIMILAR_TO") { setSelectedEdge(d); setSelected(null); }
      });

    // ── Edge score labels (only SIMILAR_TO) ──
    const edgeLabel = g.append("g").selectAll("text")
      .data(edges.filter(e => e.type === "SIMILAR_TO")).join("text")
      .attr("font-size", 7)
      .attr("fill", "var(--amber, #c8922a)")
      .attr("text-anchor", "middle")
      .attr("font-family", "'DM Mono', monospace")
      .attr("pointer-events", "none")
      .text((d: any) => `${Math.round((d.score || 0) * 100)}%`);

    // ── Cluster label overlays (floating text per cluster centroid) ──
    const clusterLabelData: any[] = [];
    clusters.forEach((cnodes, cid) => {
      const paperNodes = cnodes.filter((n: any) => n.label === "Paper");
      if (paperNodes.length < 2) return;
      const kw = getClusterKeywords(cnodes, graphData!.edges);
      clusterLabelData.push({ cid, label: kw, nodes: paperNodes });
    });

    const clusterLabelG = g.append("g").attr("class", "cluster-labels").attr("pointer-events", "none");

    // ── Nodes ──
    const node = g.append("g")
      .selectAll<SVGGElement, any>("g").data(nodes).join("g")
      .attr("cursor", "pointer")
      .attr("opacity", getNodeOpacity)
      .call(
        d3.drag<SVGGElement, any>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      )
      .on("click", (_: any, d: any) => {
        setSelected(d);
        setSelectedEdge(null);
        // Toggle focus mode on paper double-concept: click once = select, click same again = focus
        if (d.label === "Paper") {
          setFocusNode(prev => prev === d.id ? null : d.id);
        }
      });

    // Node circle
    node.append("circle")
      .attr("r", nodeRadius)
      .attr("fill", (d: any) => {
        if (searchHighlight(d)) return "#f59e0b";
        const cid = clusterMap.get(d.id);
        if (d.label === "Paper" && cid !== undefined) return clusterColor(cid);
        return NODE_COLORS[d.label] || "#94a3b8";
      })
      .attr("stroke", (d: any) => {
        if (searchHighlight(d)) return "#fbbf24";
        if (d.id === focusNode) return "#ffffff";
        return "rgba(250,248,244,0.85)";
      })
      .attr("stroke-width", (d: any) => {
        if (searchHighlight(d) || d.id === focusNode) return 3;
        return 1.8;
      });

    // Importance ring (top 20% degree papers)
    const maxDeg = Math.max(...Object.values(degree), 1);
    node.filter((d: any) => d.label === "Paper" && (degree[d.id] || 0) >= maxDeg * 0.7)
      .append("circle")
      .attr("r", (d: any) => nodeRadius(d) + 5)
      .attr("fill", "none")
      .attr("stroke", (d: any) => {
        const cid = clusterMap.get(d.id);
        return cid !== undefined ? clusterColor(cid) : "var(--amber, #c8922a)";
      })
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "3,2")
      .attr("opacity", 0.5);

    // Node label
    node.append("text")
      .attr("dy", (d: any) => nodeRadius(d) + 11)
      .attr("text-anchor", "middle")
      .attr("font-size", (d: any) => d.label === "Paper" ? 9 : 8)
      .attr("font-family", "'DM Sans', sans-serif")
      .attr("fill", "var(--text-secondary, #4a5568)")
      .attr("pointer-events", "none")
      .text((d: any) => (d.name || d.id || "").slice(0, 20));

    // Degree badge on highly-connected papers
    node.filter((d: any) => d.label === "Paper" && (degree[d.id] || 0) >= 3)
      .append("circle")
      .attr("cx", (d: any) => nodeRadius(d) * 0.7)
      .attr("cy", (d: any) => -nodeRadius(d) * 0.7)
      .attr("r", 7)
      .attr("fill", "#fff")
      .attr("stroke", (d: any) => {
        const cid = clusterMap.get(d.id);
        return cid !== undefined ? clusterColor(cid) : "#c8922a";
      })
      .attr("stroke-width", 1.5);

    node.filter((d: any) => d.label === "Paper" && (degree[d.id] || 0) >= 3)
      .append("text")
      .attr("x", (d: any) => nodeRadius(d) * 0.7)
      .attr("y", (d: any) => -nodeRadius(d) * 0.7 + 3.5)
      .attr("text-anchor", "middle")
      .attr("font-size", 7)
      .attr("font-weight", "700")
      .attr("font-family", "'DM Mono', monospace")
      .attr("fill", "#0d1b2a")
      .attr("pointer-events", "none")
      .text((d: any) => degree[d.id] || "");

    // ── Tick ──
    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x).attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x).attr("y2", (d: any) => d.target.y);

      edgeLabel
        .attr("x", (d: any) => (d.source.x + d.target.x) / 2)
        .attr("y", (d: any) => (d.source.y + d.target.y) / 2 - 3);

      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
      node.attr("opacity", getNodeOpacity);
      link.attr("opacity", getEdgeOpacity);

      // Update cluster label positions (centroid of paper nodes in cluster)
      clusterLabelG.selectAll("*").remove();
      clusterLabelData.forEach(({ cid, label, nodes: clNodes }) => {
        const live = nodes.filter(n => clNodes.some((cn: any) => cn.id === n.id));
        if (!live.length) return;
        const cx = live.reduce((s, n: any) => s + (n.x || 0), 0) / live.length;
        const cy = live.reduce((s, n: any) => s + (n.y || 0), 0) / live.length - (Math.max(...live.map((n: any) => nodeRadius(n))) + 28);

        const labelG = clusterLabelG.append("g").attr("transform", `translate(${cx},${cy})`);
        const bgRect = labelG.append("rect")
          .attr("rx", 4)
          .attr("fill", clusterColor(cid))
          .attr("opacity", 0.18);
        const txt = labelG.append("text")
          .attr("text-anchor", "middle")
          .attr("font-size", 9)
          .attr("font-family", "'DM Mono', monospace")
          .attr("fill", clusterColor(cid))
          .attr("font-weight", "600")
          .attr("letter-spacing", "0.04em")
          .text(`● ${label}`);
        // Size rect to text after render — approximate
        bgRect.attr("x", -55).attr("y", -11).attr("width", 110).attr("height", 16);
      });
    });

    // Auto-focus on search result
    if (searchFocus) {
      const targetNode = nodes.find((n: any) => n.id === searchFocus);
      if (targetNode) {
        simulation.on("end", () => {
          const scale = 2;
          const x = width / 2 - (targetNode.x || 0) * scale;
          const y = height / 2 - (targetNode.y || 0) * scale;
          svg.transition().duration(700)
            .call(zoom.transform as any, d3.zoomIdentity.translate(x, y).scale(scale));
        });
      }
    }
  };

  // ── Search handler ────────────────────────────────────────────────────────────

  const handleSearch = useCallback((q: string) => {
    setSearchQuery(q);
    if (!q || !graphData) { setSearchFocus(null); return; }
    const match = graphData.nodes.find(n =>
      (n.name || "").toLowerCase().includes(q.toLowerCase())
    );
    setSearchFocus(match?.id || null);
    if (match) { setSelected(match); setFocusNode(null); }
  }, [graphData]);

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg, #f5f2ec)" }}>

      {/* ── Header ── */}
      <div style={{
        background: "var(--surface, #fff)", borderBottom: "1px solid var(--border, #e5e1d9)",
        padding: "12px 24px", display: "flex", alignItems: "center", gap: 20, flexShrink: 0, flexWrap: "wrap"
      }}>
        <div>
          <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1rem", color: "var(--navy, #0d1b2a)", letterSpacing: "-0.02em" }}>
            Knowledge Graph
          </p>
          {graphData && (
            <p style={{ fontSize: "0.7rem", color: "var(--muted, #94a3b8)", fontFamily: "'DM Mono', monospace", marginTop: 2 }}>
              {graphData.nodes.length} nodes · {graphData.edges.length} edges · {clusters.size} clusters
            </p>
          )}
        </div>

        {/* Search */}
        <div style={{ position: "relative", flexShrink: 0 }}>
          <svg style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", opacity: 0.4 }}
            width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
          </svg>
          <input
            type="text"
            placeholder="Search node…"
            value={searchQuery}
            onChange={e => handleSearch(e.target.value)}
            style={{
              paddingLeft: 28, paddingRight: 10, paddingTop: 6, paddingBottom: 6,
              borderRadius: 6, border: "1px solid var(--border-strong, #d1cec8)",
              fontSize: "0.78rem", fontFamily: "'DM Sans', sans-serif",
              background: "var(--bg, #f5f2ec)", color: "var(--navy, #0d1b2a)",
              width: 170, outline: "none",
            }}
          />
        </div>

        {/* View mode toggle */}
        <div style={{ display: "flex", borderRadius: 6, border: "1px solid var(--border-strong, #d1cec8)", overflow: "hidden", flexShrink: 0 }}>
          {(["graph", "cluster"] as const).map(mode => (
            <button key={mode} onClick={() => setViewMode(mode)} style={{
              padding: "5px 14px", fontSize: "0.76rem", fontWeight: 500,
              background: viewMode === mode ? "var(--navy, #0d1b2a)" : "transparent",
              color: viewMode === mode ? "#fff" : "var(--text-secondary, #4a5568)",
              border: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif", transition: "all 0.15s",
            }}>
              {mode === "graph" ? "Graph" : "Clusters"}
            </button>
          ))}
        </div>

        {/* Focus mode indicator */}
        {focusNode && (
          <button onClick={() => { setFocusNode(null); setSelected(null); }} style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "5px 12px", borderRadius: 6, fontSize: "0.75rem",
            background: "rgba(200,146,42,0.12)", color: "var(--amber, #c8922a)",
            border: "1px solid rgba(200,146,42,0.3)", cursor: "pointer",
            fontFamily: "'DM Sans', sans-serif", fontWeight: 500, flexShrink: 0
          }}>
            <span>Focus mode ON</span>
            <span style={{ opacity: 0.6 }}>✕ clear</span>
          </button>
        )}

        {/* Node type filters */}
        <div style={{ marginLeft: "auto", display: "flex", gap: 5, flexWrap: "wrap" }}>
          {LABELS.map(l => (
            <button key={l} onClick={() => setFilter(l)} style={{
              padding: "4px 10px", borderRadius: 4, fontSize: "0.73rem", fontWeight: 500,
              border: `1px solid ${filter === l ? NODE_COLORS[l] || "var(--navy, #0d1b2a)" : "var(--border-strong, #d1cec8)"}`,
              background: filter === l ? (NODE_COLORS[l] || "var(--navy, #0d1b2a)") : "transparent",
              color: filter === l ? "#fff" : "var(--text-secondary, #4a5568)",
              cursor: "pointer", transition: "all 0.15s", fontFamily: "'DM Sans', sans-serif",
              display: "flex", alignItems: "center", gap: 4,
            }}>
              {l !== "All" && (
                <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: filter === l ? "rgba(255,255,255,0.55)" : NODE_COLORS[l] }} />
              )}
              {l}
              {l !== "All" && counts[l] ? <span style={{ opacity: 0.6, fontSize: "0.68rem" }}>({counts[l]})</span> : null}
            </button>
          ))}
        </div>
      </div>

      {/* ── Main area ── */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>

        {/* Loading */}
        {loading && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg, #f5f2ec)", flexDirection: "column", gap: 14, zIndex: 10 }}>
            <div style={{ width: 30, height: 30, border: "2px solid var(--border, #e5e1d9)", borderTopColor: "var(--navy, #0d1b2a)", borderRadius: "50%", animation: "spin 0.7s linear infinite" }} />
            <p style={{ fontSize: "0.82rem", color: "var(--text-secondary, #4a5568)" }}>Loading graph…</p>
          </div>
        )}

        {error && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", zIndex: 10 }}>
            <div style={{ textAlign: "center", maxWidth: 320 }}>
              <p style={{ fontWeight: 500, fontSize: "0.9rem", color: "var(--navy, #0d1b2a)", marginBottom: 6 }}>{error}</p>
              <p style={{ fontSize: "0.8rem", color: "var(--muted, #94a3b8)" }}>Ensure Neo4j is running and papers have been uploaded.</p>
            </div>
          </div>
        )}

        {!loading && !error && graphData?.nodes.length === 0 && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <p style={{ fontFamily: "'DM Serif Display', serif", fontSize: "1.1rem", color: "var(--navy, #0d1b2a)" }}>
              No graph data yet — upload papers first.
            </p>
          </div>
        )}

        {/* ── Cluster View ── */}
        {viewMode === "cluster" && !loading && !error && (
          <div style={{ padding: 24, overflowY: "auto", height: "100%", display: "flex", flexWrap: "wrap", gap: 16, alignContent: "flex-start" }}>
            {Array.from(clusters.entries()).map(([cid, cnodes]) => {
              const paperNodes = cnodes.filter((n: any) => n.label === "Paper");
              if (paperNodes.length === 0) return null;
              const kw = getClusterKeywords(cnodes, graphData?.edges || []);
              const summary = clusterSummaries[cid];
              const isLoading = summaryLoading[cid];
              const color = clusterColor(cid);
              return (
                <div key={cid} style={{
                  background: "var(--surface, #fff)", border: `1px solid ${color}33`,
                  borderRadius: 10, padding: 18, width: 280,
                  boxShadow: `0 2px 16px ${color}18`, flexShrink: 0,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, flexShrink: 0 }} />
                    <span style={{ fontSize: "0.7rem", fontFamily: "'DM Mono', monospace", color, fontWeight: 600, letterSpacing: "0.06em" }}>
                      CLUSTER {cid + 1}
                    </span>
                    <span style={{ marginLeft: "auto", fontSize: "0.68rem", color: "var(--muted, #94a3b8)", fontFamily: "'DM Mono', monospace" }}>
                      {paperNodes.length} paper{paperNodes.length !== 1 ? "s" : ""}
                    </span>
                  </div>

                  <p style={{ fontSize: "0.8rem", color: "var(--navy, #0d1b2a)", fontWeight: 500, marginBottom: 6, lineHeight: 1.4 }}>
                    {kw}
                  </p>

                  <div style={{ marginBottom: 12 }}>
                    {paperNodes.slice(0, 4).map(n => (
                      <div key={n.id} style={{
                        fontSize: "0.73rem", color: "var(--text-secondary, #4a5568)",
                        padding: "3px 0", borderBottom: "1px solid var(--border, #e5e1d9)",
                        lineHeight: 1.35, cursor: "pointer",
                      }}
                        onClick={() => { setViewMode("graph"); setSelected(n); setFocusNode(n.id); }}
                      >
                        {(n.name || n.id).slice(0, 40)}
                      </div>
                    ))}
                    {paperNodes.length > 4 && (
                      <div style={{ fontSize: "0.7rem", color: "var(--muted, #94a3b8)", marginTop: 4 }}>
                        +{paperNodes.length - 4} more
                      </div>
                    )}
                  </div>

                  {/* LLM summary — on-demand */}
                  {summary ? (
                    <div style={{ background: `${color}0d`, borderRadius: 6, padding: "8px 10px", borderLeft: `3px solid ${color}` }}>
                      <p style={{ fontSize: "0.72rem", color: "var(--text-secondary, #4a5568)", lineHeight: 1.55, fontStyle: "italic" }}>
                        {summary}
                      </p>
                    </div>
                  ) : (
                    <button
                      onClick={() => fetchClusterSummary(cid)}
                      disabled={isLoading}
                      style={{
                        width: "100%", padding: "6px 0", borderRadius: 6, fontSize: "0.73rem",
                        border: `1px solid ${color}44`, background: "transparent",
                        color, cursor: isLoading ? "default" : "pointer",
                        fontFamily: "'DM Sans', sans-serif", display: "flex", alignItems: "center", justifyContent: "center", gap: 6
                      }}
                    >
                      {isLoading ? (
                        <>
                          <div style={{ width: 10, height: 10, border: `1.5px solid ${color}`, borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.7s linear infinite" }} />
                          Generating…
                        </>
                      ) : "✦ Generate AI Summary"}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* ── Graph SVG ── */}
        {viewMode === "graph" && (
          <svg ref={svgRef} style={{ width: "100%", height: "100%" }} />
        )}

        {/* ── Legend ── */}
        {viewMode === "graph" && (
          <div style={{
            position: "absolute", bottom: 20, left: 20,
            background: "var(--surface, #fff)", border: "1px solid var(--border, #e5e1d9)",
            borderRadius: 8, padding: "12px 14px", boxShadow: "0 2px 12px rgba(0,0,0,0.06)", zIndex: 5
          }}>
            <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.58rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted, #94a3b8)", marginBottom: 8 }}>
              Node Types
            </p>
            {Object.entries(NODE_COLORS).map(([label, color]) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
                <div style={{ width: 9, height: 9, borderRadius: "50%", background: color, flexShrink: 0 }} />
                <span style={{ fontSize: "0.72rem", color: "var(--text-secondary, #4a5568)" }}>
                  {label}
                  {counts[label] ? <span style={{ fontFamily: "'DM Mono', monospace", color: "var(--muted, #94a3b8)", marginLeft: 4 }}>({counts[label]})</span> : ""}
                </span>
              </div>
            ))}
            <div style={{ borderTop: "1px solid var(--border, #e5e1d9)", marginTop: 8, paddingTop: 8 }}>
              <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.58rem", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted, #94a3b8)", marginBottom: 6 }}>
                Edge Types
              </p>
              {[
                { dash: "none", color: "#c8922a", label: "Similarity" },
                { dash: "6,3",  color: "#0f523c", label: "Method" },
                { dash: "2,4",  color: "#501464", label: "Dataset" },
              ].map(({ dash, color, label }) => (
                <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
                  <svg width="22" height="8">
                    <line x1="0" y1="4" x2="22" y2="4" stroke={color} strokeWidth="1.8" strokeDasharray={dash} />
                  </svg>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-secondary, #4a5568)" }}>{label}</span>
                </div>
              ))}
            </div>
            <div style={{ borderTop: "1px solid var(--border, #e5e1d9)", marginTop: 6, paddingTop: 6 }}>
              <p style={{ fontSize: "0.65rem", color: "var(--muted, #94a3b8)", lineHeight: 1.4 }}>
                Click paper → Focus mode<br/>
                Bigger node = more connections
              </p>
            </div>
          </div>
        )}

        {/* ── Node Insight Panel ── */}
        {selected && viewMode === "graph" && (
          <div style={{
            position: "absolute", top: 16, right: 16,
            background: "var(--surface, #fff)", border: "1px solid var(--border-strong, #d1cec8)",
            borderRadius: 10, padding: "16px 18px", width: 300,
            boxShadow: "0 4px 24px rgba(0,0,0,0.10)", animation: "fadeUp 0.2s ease", zIndex: 10
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
              <span style={{
                padding: "3px 9px", borderRadius: 3, fontSize: "0.66rem",
                fontFamily: "'DM Mono', monospace", letterSpacing: "0.06em", fontWeight: 600,
                color: "#fff",
                background: selected.label === "Paper"
                  ? clusterColor(clusterMap.get(selected.id) ?? 0)
                  : NODE_COLORS[selected.label] || "var(--muted, #94a3b8)"
              }}>
                {selected.label}
              </span>
              <button onClick={() => { setSelected(null); setFocusNode(null); }}
                style={{ border: "none", background: "none", cursor: "pointer", color: "var(--muted, #94a3b8)", fontSize: "1.1rem", padding: 2, lineHeight: 1 }}>×</button>
            </div>

            <p style={{ fontWeight: 600, fontSize: "0.88rem", color: "var(--navy, #0d1b2a)", lineHeight: 1.4, marginBottom: 10 }}>
              {selected.name || selected.id}
            </p>

            {selected.label === "Paper" && (() => {
              const cid = clusterMap.get(selected.id);
              const clNodes = cid !== undefined ? clusters.get(cid) || [] : [];
              const paperCount = clNodes.filter(n => n.label === "Paper").length;
              const kw = cid !== undefined ? getClusterKeywords(clNodes, graphData?.edges || []) : "";
              const deg = graphData?.edges.filter(e => {
                const s = typeof e.source === "object" ? e.source.id : e.source;
                const t = typeof e.target === "object" ? e.target.id : e.target;
                return s === selected.id || t === selected.id;
              }).length || 0;
              const detail = paperDetails[selected.id];

              return (
                <>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 12 }}>
                    {cid !== undefined && (
                      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                        <div style={{ width: 8, height: 8, borderRadius: "50%", background: clusterColor(cid), flexShrink: 0 }} />
                        <div>
                          <span style={{ fontSize: "0.68rem", fontFamily: "'DM Mono', monospace", color: "var(--muted, #94a3b8)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Cluster</span>
                          <p style={{ fontSize: "0.75rem", color: "var(--text-secondary, #4a5568)", marginTop: 1 }}>{kw} ({paperCount} papers)</p>
                        </div>
                      </div>
                    )}
                    <div style={{ display: "flex", gap: 16 }}>
                      <div>
                        <span style={{ fontSize: "0.68rem", fontFamily: "'DM Mono', monospace", color: "var(--muted, #94a3b8)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Connections</span>
                        <p style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--navy, #0d1b2a)" }}>{deg}</p>
                      </div>
                      {detail?.topics?.length > 0 && (
                        <div>
                          <span style={{ fontSize: "0.68rem", fontFamily: "'DM Mono', monospace", color: "var(--muted, #94a3b8)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Topics</span>
                          <p style={{ fontSize: "0.75rem", color: "var(--text-secondary, #4a5568)", marginTop: 1 }}>{detail.topics.slice(0, 2).join(", ")}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {detail?.abstract && (
                    <div style={{ background: "var(--bg, #f5f2ec)", borderRadius: 6, padding: "8px 10px", marginBottom: 10 }}>
                      <p style={{ fontSize: "0.7rem", color: "var(--text-secondary, #4a5568)", lineHeight: 1.55 }}>
                        {detail.abstract.slice(0, 220)}{detail.abstract.length > 220 ? "…" : ""}
                      </p>
                    </div>
                  )}

                  {detail?.authors?.length > 0 && (
                    <p style={{ fontSize: "0.7rem", color: "var(--muted, #94a3b8)", marginBottom: 8 }}>
                      {detail.authors.slice(0, 3).join(", ")}{detail.authors.length > 3 ? " et al." : ""}
                    </p>
                  )}

                  <button
                    onClick={() => setFocusNode(prev => prev === selected.id ? null : selected.id)}
                    style={{
                      width: "100%", padding: "7px 0", borderRadius: 6, fontSize: "0.75rem",
                      border: `1px solid ${focusNode === selected.id ? "var(--amber, #c8922a)" : "var(--border-strong, #d1cec8)"}`,
                      background: focusNode === selected.id ? "rgba(200,146,42,0.1)" : "transparent",
                      color: focusNode === selected.id ? "var(--amber, #c8922a)" : "var(--text-secondary, #4a5568)",
                      cursor: "pointer", fontFamily: "'DM Sans', sans-serif", fontWeight: 500, transition: "all 0.15s"
                    }}
                  >
                    {focusNode === selected.id ? "✕ Exit Focus Mode" : "⊙ Focus on this Paper"}
                  </button>
                </>
              );
            })()}

            {selected.label !== "Paper" && (
              <p style={{ fontSize: "0.75rem", color: "var(--muted, #94a3b8)", lineHeight: 1.5 }}>
                {selected.label} node shared across {
                  graphData?.edges.filter(e => {
                    const s = typeof e.source === "object" ? e.source.id : e.source;
                    const t = typeof e.target === "object" ? e.target.id : e.target;
                    return s === selected.id || t === selected.id;
                  }).length || 0
                } connection{(graphData?.edges.filter(e => {
                  const s = typeof e.source === "object" ? e.source.id : e.source;
                  const t = typeof e.target === "object" ? e.target.id : e.target;
                  return s === selected.id || t === selected.id;
                }).length || 0) !== 1 ? "s" : ""}.
              </p>
            )}
          </div>
        )}

        {/* ── Edge insight panel ── */}
        {selectedEdge && !selected && viewMode === "graph" && (
          <div style={{
            position: "absolute", top: 16, right: 16,
            background: "var(--surface, #fff)", border: "1px solid rgba(200,146,42,0.35)",
            borderRadius: 10, padding: "16px 18px", width: 300,
            boxShadow: "0 4px 24px rgba(0,0,0,0.10)", animation: "fadeUp 0.2s ease", zIndex: 10
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <span style={{
                padding: "3px 9px", borderRadius: 3, fontSize: "0.66rem",
                fontFamily: "'DM Mono', monospace", fontWeight: 600,
                color: "var(--amber, #c8922a)", background: "rgba(200,146,42,0.1)",
                border: "1px solid rgba(200,146,42,0.2)"
              }}>
                {Math.round((selectedEdge.score || 0) * 100)}% similar
              </span>
              <button onClick={() => setSelectedEdge(null)}
                style={{ border: "none", background: "none", cursor: "pointer", color: "var(--muted, #94a3b8)", fontSize: "1.1rem", padding: 2 }}>×</button>
            </div>

            <p style={{ fontSize: "0.75rem", color: "var(--navy, #0d1b2a)", fontWeight: 600, marginBottom: 12, lineHeight: 1.35 }}>
              Why are these papers connected?
            </p>

            {selectedEdge.shared_topics?.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.58rem", color: "var(--muted, #94a3b8)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 5 }}>
                  Shared Topics
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {selectedEdge.shared_topics.map((t: string) => (
                    <span key={t} style={{ fontSize: "0.7rem", padding: "2px 7px", borderRadius: 3, background: "rgba(200,146,42,0.08)", color: "var(--amber, #c8922a)", border: "1px solid rgba(200,146,42,0.2)" }}>{t}</span>
                  ))}
                </div>
              </div>
            )}

            {selectedEdge.shared_methods?.length > 0 && (
              <div style={{ marginBottom: 10 }}>
                <p style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.58rem", color: "var(--muted, #94a3b8)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 5 }}>
                  Shared Methods
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {selectedEdge.shared_methods.map((m: string) => (
                    <span key={m} style={{ fontSize: "0.7rem", padding: "2px 7px", borderRadius: 3, background: "var(--bg, #f5f2ec)", color: "var(--text-secondary, #4a5568)", border: "1px solid var(--border-strong, #d1cec8)" }}>{m}</span>
                  ))}
                </div>
              </div>
            )}

            {!selectedEdge.shared_topics?.length && !selectedEdge.shared_methods?.length && (
              <p style={{ fontSize: "0.76rem", color: "var(--muted, #94a3b8)", lineHeight: 1.5 }}>
                Connected via semantic similarity of title and abstract.
              </p>
            )}
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}