import axios from "axios";

const rawUrl = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const baseURL = rawUrl.replace(/\/+$/, "");

const api = axios.create({
    baseURL: baseURL,
    timeout: 300000, // 5 min timeout for video processing on cloud CPU
    headers: {
        "Content-Type": "application/json",
    },
});

export const streamUrl = `${api.defaults.baseURL}/camera/live`;

export default api;
