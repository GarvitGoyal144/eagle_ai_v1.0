import { useState, useRef, useEffect } from "react";
import api from "../../api/api";

import type { VideoInsights } from "../../types/event";

interface Props {
    onInsightsReady: (insights: VideoInsights, sessionId: string) => void;
}

function VideoUpload({ onInsightsReady }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);
    const [statusText, setStatusText] = useState("");
    const [errorMessage, setErrorMessage] = useState("");
    const videoRef = useRef<HTMLVideoElement | null>(null);

    // Create and cleanup video preview URL
    useEffect(() => {
        if (file) {
            const url = URL.createObjectURL(file);
            setVideoPreviewUrl(url);
            return () => {
                URL.revokeObjectURL(url);
            };
        } else {
            setVideoPreviewUrl(null);
        }
    }, [file]);

    const handleProcess = async () => {
        if (!file) return;
        setUploading(true);
        setErrorMessage("");
        setStatusText("Uploading video footage to surveillance engine...");

        const formData = new FormData();
        formData.append("file", file);

        try {
            // 1. Upload
            await api.post("/video/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            
            // 2. Process
            setStatusText("Analyzing video frames (tracking objects & scene semantics)...");
            const res = await api.post<VideoInsights>("/video/process", {
                filename: file.name
            });
            
            console.log("Processing complete", res.data);
            onInsightsReady(res.data, res.data.session_id);
            setStatusText("");
        } catch (err: any) {
            console.error("Failed to process video:", err);
            const msg = err?.response?.data?.detail || err?.message || "Failed to process video";
            setErrorMessage(msg);
            setStatusText("");
        } finally {
            setUploading(false);
        }
    };

    const formatFileSize = (bytes: number) => {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    };

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-base font-bold text-slate-200 flex items-center gap-2">
                        <span>📹</span> Surveillance Video Feed
                    </h2>
                    <p className="text-xs text-slate-400 mt-0.5">
                        Select and inspect footage with YOLO object tracking & scene intelligence.
                    </p>
                </div>

                <div className="flex items-center gap-2">
                    <input
                        type="file"
                        accept="video/*"
                        id="video-input"
                        onChange={(e) => {
                            if (e.target.files && e.target.files[0]) {
                                setFile(e.target.files[0]);
                                setErrorMessage("");
                            }
                        }}
                        className="hidden"
                    />
                    
                    <label
                        htmlFor="video-input"
                        className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-2 px-3.5 rounded-lg border border-slate-700 cursor-pointer transition-all hover:border-slate-600 shadow-sm flex items-center gap-1.5"
                    >
                        <span>📁</span>
                        <span>{file ? "Change Video" : "Select Video File"}</span>
                    </label>
                </div>
            </div>

            {/* Video Player Preview Box */}
            {videoPreviewUrl ? (
                <div className="space-y-2">
                    <div className="relative aspect-video bg-black rounded-lg overflow-hidden border border-slate-800 shadow-inner group">
                        <video
                            ref={videoRef}
                            src={videoPreviewUrl}
                            controls
                            playsInline
                            className="w-full h-full object-contain"
                        />
                        <div className="absolute top-2 left-2 bg-slate-950/80 backdrop-blur-sm text-slate-200 text-[10px] font-mono px-2 py-0.5 rounded border border-slate-700/50">
                            🔴 Live Preview • {file?.name} ({file ? formatFileSize(file.size) : ""})
                        </div>
                    </div>
                </div>
            ) : (
                <div
                    onClick={() => document.getElementById("video-input")?.click()}
                    className="border-2 border-dashed border-slate-800 hover:border-blue-500/50 rounded-xl p-8 text-center cursor-pointer transition-colors bg-slate-950/50 hover:bg-slate-950 flex flex-col items-center justify-center space-y-2"
                >
                    <span className="text-3xl opacity-80">🎥</span>
                    <p className="text-xs font-medium text-slate-300">
                        Click to select an MP4 surveillance video for analysis
                    </p>
                    <p className="text-[10px] text-slate-500">
                        Supports MP4, AVI, MOV up to 1080p
                    </p>
                </div>
            )}

            {/* Status Feedback */}
            {statusText && (
                <div className="text-xs text-blue-400 text-center animate-pulse py-1 font-medium bg-blue-950/30 border border-blue-900/40 rounded-lg">
                    🔄 {statusText}
                </div>
            )}

            {errorMessage && (
                <div className="text-xs text-rose-400 bg-rose-950/40 border border-rose-900/60 rounded-lg p-2.5 text-center font-medium">
                    ⚠️ {errorMessage}
                </div>
            )}

            {/* Action Button */}
            <button
                disabled={!file || uploading}
                onClick={handleProcess}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold py-2.5 px-4 rounded-lg transition-all shadow-md flex items-center justify-center gap-2"
            >
                {uploading ? (
                    <>
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                        <span>Processing Surveillance Intelligence...</span>
                    </>
                ) : (
                    <span>Analyze Surveillance Video ▶</span>
                )}
            </button>
        </div>
    );
}

export default VideoUpload;