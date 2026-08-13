import { useState } from "react";
import api from "../../api/api";

import type { VideoInsights } from "../../types/event";

interface Props {
    onInsightsReady: (insights: VideoInsights, sessionId: string) => void;
}

function VideoUpload({ onInsightsReady }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [statusText, setStatusText] = useState("");

    const handleProcess = async () => {
        if (!file) return;
        setUploading(true);
        setStatusText("Uploading...");

        const formData = new FormData();
        formData.append("file", file);

        try {
            // 1. Upload
            setStatusText("Uploading video...");
            await api.post("/video/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            
            // 2. Process
            setStatusText("Analyzing video (this may take a minute)...");
            const res = await api.post<VideoInsights>("/video/process", {
                filename: file.name
            });
            
            console.log("Processing complete", res.data);
            onInsightsReady(res.data, res.data.session_id);
            setStatusText("");
        } catch (err) {
            console.error("Failed to upload video:", err);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
            <div>
                <h2 className="text-lg font-bold text-slate-200">
                    📁 Upload Video
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                    Upload an MP4 file to run YOLO object detection and MobileCLIP analysis.
                </p>
            </div>

            <div className="flex items-center gap-3">
                <input
                    type="file"
                    accept="video/*"
                    id="video-input"
                    onChange={(e) => {
                        if (e.target.files) {
                            setFile(e.target.files[0]);
                        }
                    }}
                    className="hidden"
                />
                
                <label
                    htmlFor="video-input"
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-2 px-3 rounded-lg border border-slate-700 cursor-pointer transition-colors"
                >
                    Choose File
                </label>
                <span className="text-xs text-slate-400 truncate max-w-[150px]">
                    {file ? file.name : "No file chosen"}
                </span>
            </div>

            {statusText && (
                <div className="text-xs text-blue-400 text-center animate-pulse py-1">
                    {statusText}
                </div>
            )}

            <button
                disabled={!file || uploading}
                onClick={handleProcess}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold py-2.5 px-4 rounded-lg transition-colors shadow-md"
            >
                {uploading ? "Processing..." : "Analyze Video ▶"}
            </button>
        </div>
    );
}

export default VideoUpload;