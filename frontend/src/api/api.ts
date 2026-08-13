import axios from "axios";

const rawUrl = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const baseURL = rawUrl.replace(/\/+$/, "");

const api = axios.create({
    baseURL: baseURL,
    timeout: 60000, // 60s timeout to gracefully handle Render cold-starts
    headers: {
        "Content-Type": "application/json",
    },
});

export const streamUrl = `${api.defaults.baseURL}/camera/live`;

export default api;
