import axios, { AxiosInstance, AxiosError } from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api: AxiosInstance = axios.create({
  baseURL: API_URL,
  headers: { "Content-Type": "application/json" },
  // FIX: add a default timeout so requests don't hang forever
  timeout: 30000,
});

// Attach JWT token on every request
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// FIX: Handle 401 globally + retry transient failures with exponential backoff
const MAX_RETRIES = 2;

api.interceptors.response.use(
  (res) => res,
  async (err: AxiosError) => {
    const config = err.config as any;

    // 401 → clear token and redirect
    if (err.response?.status === 401 && typeof window !== "undefined") {
      localStorage.removeItem("token");
      window.location.href = "/auth";
      return Promise.reject(err);
    }

    // Don't retry non-idempotent mutations (POST with body) except /upload
    const isRetriable =
      err.config?.method === "get" ||
      err.config?.url?.includes("/upload") === false;

    // Retry on network errors or 5xx
    const isTransient =
      !err.response || (err.response.status >= 500 && err.response.status < 600);

    if (isRetriable && isTransient) {
      config._retryCount = config._retryCount || 0;
      if (config._retryCount < MAX_RETRIES) {
        config._retryCount += 1;
        const delay = Math.pow(2, config._retryCount) * 500; // 1s, 2s
        await new Promise((r) => setTimeout(r, delay));
        return api(config);
      }
    }

    return Promise.reject(err);
  }
);

// ─── Auth ─────────────────────────────────────────────────────────────────────

export const authApi = {
  signup: (email: string, username: string, password: string) =>
    api.post("/api/auth/signup", { email, username, password }),

  login: (email: string, password: string) =>
    api.post("/api/auth/login", { email, username: email, password }),

  me: () => api.get("/api/auth/me"),
};

// ─── Papers ───────────────────────────────────────────────────────────────────

export const papersApi = {
  list: (skip = 0, limit = 20) =>
    api.get("/api/papers", { params: { skip, limit } }),

  get: (paperId: string) => api.get(`/api/papers/${paperId}`),

  // FIX: lightweight status-only poll — use this while status === "processing"
  getStatus: (paperId: string) => api.get(`/api/papers/${paperId}/status`),

  upload: (
    file: File,
    formatHint: string = "auto",
    onProgress?: (pct: number) => void
  ) => {
    const form = new FormData();
    form.append("file", file);
    form.append("format_hint", formatHint);
    return api.post("/api/papers/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
      // FIX: longer timeout for upload + AI processing
      timeout: 120000,
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      },
    });
  },

  delete: (paperId: string) => api.delete(`/api/papers/${paperId}`),

  sectionSummary: (paperId: string, sectionName: string) =>
    api.get(`/api/papers/${paperId}/section-summary`, {
      params: { section_name: sectionName },
    }),

  graph: (paperId: string) => api.get(`/api/papers/${paperId}/graph`),

  recommendations: (paperId: string) =>
    api.get(`/api/recommendations/${paperId}`),

  figureSummary: (paperId: string, label: string) =>
    api.get(`/api/papers/${paperId}/figure-summary`, { params: { label } }),

  reEnrich: (paperId: string) =>
    api.post(`/api/papers/${paperId}/re-enrich`),

  reEnrichAll: () =>
    api.post("/api/papers/re-enrich-all"),

  compare: (paperIds: string[]) =>
    api.post("/api/compare", { paper_ids: paperIds }),

  compareDeep: (paperIds: string[]) =>
    api.post("/api/papers/compare", { paper_ids: paperIds }),
};

// ─── Search & Chat ────────────────────────────────────────────────────────────

export const searchApi = {
  search: (query: string, paperId?: string, topK = 5) =>
    api.get("/api/search", { params: { query, paper_id: paperId, top_k: topK } }),

  query: (question: string, paperId?: string, chatHistory: any[] = []) =>
    api.post("/api/query", {
      question,
      paper_id: paperId,
      chat_history: chatHistory,
    }),
};

// ─── Graph ────────────────────────────────────────────────────────────────────

export const graphApi = {
  full: () => api.get("/api/graph"),
};

export default api;