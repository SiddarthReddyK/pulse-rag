// Base URL of the Pulse RAG API.
// Leave VITE_API_BASE_URL empty in development: Vite proxies /api and /health
// to the backend (see vite.config.ts). Set it when the API lives elsewhere,
// e.g. VITE_API_BASE_URL=https://api.example.com
export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "";
