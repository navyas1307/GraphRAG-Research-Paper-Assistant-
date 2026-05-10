"use client";
import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { papersApi } from "@/lib/api";

async function pollUntilReady(
  paperId: string,
  onProgress: (msg: string) => void,
  maxAttempts = 20
): Promise<"ready" | "failed"> {
  const intervals = [3000, 5000, 5000, 7000, 7000];
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, intervals[Math.min(i, intervals.length - 1)]));
    try {
      const res = await papersApi.getStatus(paperId);
      const { status, llm_status } = res.data;
      if (llm_status === "quota_exceeded") {
        onProgress("AI quota reached — paper saved with keyword analysis.");
        return "ready";
      }
      if (status === "ready") {
        onProgress("Done! Opening paper…");
        return "ready";
      }
      if (status === "failed") {
        onProgress("AI enrichment failed — basic data saved.");
        return "failed";
      }
      onProgress(
        i < 3 ? "Extracting sections & metadata…"
        : i < 7 ? "Building knowledge graph…"
        : "Finalising AI analysis…"
      );
    } catch {
      // transient — keep polling
    }
  }
  return "ready";
}

export default function UploadPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging]     = useState(false);
  const [isUploading, setIsUploading]   = useState(false);
  const [progress, setProgress]         = useState("");
  const [uploadPct, setUploadPct]       = useState(0);
  const [error, setError]               = useState("");

  const handleFile = (file: File) => {
    if (!file.name.endsWith(".pdf")) { setError("Only PDF files are supported."); return; }
    if (file.size > 50 * 1024 * 1024) { setError("File too large. Maximum 50 MB."); return; }
    setError("");
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile || isUploading) return;
    setIsUploading(true);
    setError("");
    setUploadPct(0);
    setProgress("Uploading…");

    try {
      const res = await papersApi.upload(selectedFile, "auto", (pct) => {
        setUploadPct(pct);
        setProgress(pct < 100 ? `Uploading… ${pct}%` : "Processing PDF…");
      });

      const { paper_id } = res.data;
      setProgress("Extracting sections & metadata…");

      const finalStatus = await pollUntilReady(paper_id, setProgress);
      if (finalStatus === "failed") setProgress("Opening paper (AI enrichment had issues)…");
      router.push(`/papers/${paper_id}`);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || "Upload failed. Please try again.");
      setIsUploading(false);
      setProgress("");
      setUploadPct(0);
    }
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "var(--bg)",
      display: "flex",
      alignItems: "flex-start",
      justifyContent: "center",
      padding: "56px 16px",
    }}>
      <div style={{ width: "100%", maxWidth: 520, fontFamily: "'DM Sans', sans-serif" }}>

        {/* Header */}
        <h1 style={{
          fontFamily: "'DM Serif Display', serif",
          fontSize: "1.9rem",
          fontWeight: 400,
          color: "var(--navy)",
          marginBottom: 6,
        }}>
          Upload Paper
        </h1>
        <p style={{ fontSize: "0.88rem", color: "var(--muted)", marginBottom: 40, lineHeight: 1.6 }}>
          Drop a PDF and the system will extract sections, build a knowledge graph, and run AI analysis automatically.
        </p>

        {/* Drop Zone */}
        <div
          onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={e => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
          onClick={() => !isUploading && fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${isDragging ? "var(--navy)" : selectedFile ? "#10B981" : "var(--border)"}`,
            borderRadius: 12,
            padding: "52px 24px",
            textAlign: "center",
            cursor: isUploading ? "default" : "pointer",
            marginBottom: 20,
            background: isDragging ? "rgba(30,58,95,0.04)" : selectedFile ? "rgba(16,185,129,0.04)" : "var(--surface)",
            transition: "all 0.15s ease",
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            style={{ display: "none" }}
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
          />

          {selectedFile ? (
            <>
              <div style={{ fontSize: "2rem", marginBottom: 10 }}>📄</div>
              <p style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--navy)", marginBottom: 4 }}>
                {selectedFile.name}
              </p>
              <p style={{ fontSize: "0.76rem", color: "var(--muted)" }}>
                {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
                {!isUploading && " · Click to change"}
              </p>
            </>
          ) : (
            <>
              <div style={{ fontSize: "2rem", marginBottom: 10, opacity: 0.3 }}>⬆</div>
              <p style={{ fontSize: "0.92rem", color: "var(--text-secondary)", marginBottom: 4 }}>
                Drop your PDF here or{" "}
                <span style={{ color: "var(--navy)", fontWeight: 600 }}>browse</span>
              </p>
              <p style={{ fontSize: "0.74rem", color: "var(--muted)" }}>PDF only · Max 50 MB</p>
            </>
          )}
        </div>

        {/* Upload progress bar */}
        {isUploading && uploadPct > 0 && uploadPct < 100 && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ height: 3, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
              <div style={{
                height: "100%",
                background: "var(--navy)",
                borderRadius: 2,
                width: `${uploadPct}%`,
                transition: "width 0.3s ease",
              }} />
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            marginBottom: 16,
            padding: "10px 14px",
            borderRadius: 6,
            background: "rgba(239,68,68,0.06)",
            border: "1px solid rgba(239,68,68,0.2)",
            fontSize: "0.82rem",
            color: "#DC2626",
          }}>
            {error}
          </div>
        )}

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          style={{
            width: "100%",
            padding: "14px 24px",
            background: selectedFile && !isUploading ? "var(--navy)" : "#9CA3AF",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            fontSize: "0.92rem",
            fontWeight: 600,
            cursor: selectedFile && !isUploading ? "pointer" : "not-allowed",
            transition: "background 0.15s ease",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 10,
          }}
        >
          {isUploading ? (
            <>
              <div style={{
                width: 14, height: 14, borderRadius: "50%",
                border: "2px solid rgba(255,255,255,0.3)",
                borderTopColor: "#fff",
                animation: "spin 0.7s linear infinite",
              }} />
              {progress}
            </>
          ) : selectedFile ? "Upload & Process" : "Select a PDF to upload"}
        </button>

      </div>

      <style dangerouslySetInnerHTML={{ __html: `@keyframes spin { to { transform: rotate(360deg); } }` }} />
    </div>
  );
}