import { useState, useRef, useEffect } from "react";
import api from "../../api/api";
import type { VideoInsights } from "../../types/event";

interface Props {
    onInsightsReady: (insights: VideoInsights, sessionId: string) => void;
}

interface ProgressState {
    status: "idle" | "uploading" | "processing" | "completed" | "error";
    progress: number;
    current_frame: number;
    total_frames: number;
    detections: number;
    unique_tracks: number;
    elapsed_sec: number;
    step: string;
}

function VideoUpload({ onInsightsReady }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");
    const [progressState, setProgressState] = useState<ProgressState>({
        status: "idle",
        progress: 0,
        current_frame: 0,
        total_frames: 0,
        detections: 0,
        unique_tracks: 0,
        elapsed_sec: 0,
        step: "",
    });

    const videoRef = useRef<HTMLVideoElement | null>(null);
    const pollIntervalRef = useRef<number | null>(null);

    // Create and cleanup video preview URL
    useEffect(() => {
        if (file) {
            const url = URL.createObjectURL(file);
            setVideoPreviewUrl(url);
            setErrorMessage("");
            setProgressState({
                status: "idle",
                progress: 0,
                current_frame: 0,
                total_frames: 0,
                detections: 0,
                unique_tracks: 0,
                elapsed_sec: 0,
                step: "Ready for analysis",
            });
            return () => {
                URL.revokeObjectURL(url);
            };
        } else {
            setVideoPreviewUrl(null);
        }
    }, [file]);

    // Clean up polling interval on unmount
    useEffect(() => {
        return () => {
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
            }
        };
    }, []);

    const startProgressPolling = (filename: string) => {
        if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
        }

        pollIntervalRef.current = setInterval(async () => {
            try {
                const res = await api.get(`/video/progress/${filename}`);
                if (res.data) {
                    setProgressState((prev) => ({
                        ...prev,
                        ...res.data,
                    }));

                    if (res.data.status === "completed" || res.data.status === "error") {
                        if (pollIntervalRef.current) {
                            clearInterval(pollIntervalRef.current);
                            pollIntervalRef.current = null;
                        }
                    }
                }
            } catch {
                // Ignore transient polling network glitches
            }
        }, 350);
    };

    const handleProcess = async () => {
        if (!file) return;
        setUploading(true);
        setErrorMessage("");

        setProgressState({
            status: "uploading",
            progress: 5,
            current_frame: 0,
            total_frames: 0,
            detections: 0,
            unique_tracks: 0,
            elapsed_sec: 0,
            step: "Uploading video to surveillance engine...",
        });

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Step 1: Upload
            await api.post("/video/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            // Step 2: Start polling live progress
            setProgressState((prev) => ({
                ...prev,
                status: "processing",
                progress: 10,
                step: "Initializing YOLO neural network...",
            }));
            startProgressPolling(file.name);

            // Step 3: Trigger process
            const res = await api.post<VideoInsights>("/video/process", {
                filename: file.name,
            });

            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
            }

            setProgressState({
                status: "completed",
                progress: 100,
                current_frame: res.data.total_frames_sampled || 0,
                total_frames: res.data.total_frames_sampled || 0,
                detections: res.data.total_detections || 0,
                unique_tracks: res.data.unique_tracks || 0,
                elapsed_sec: res.data.processing_time_seconds || 0,
                step: "Surveillance intelligence analysis complete!",
            });

            onInsightsReady(res.data, res.data.session_id);
        } catch (err: any) {
            console.error("Failed to process video:", err);
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
            }
            const msg = err?.response?.data?.detail || err?.message || "Failed to process video";
            setErrorMessage(msg);
            setProgressState((prev) => ({
                ...prev,
                status: "error",
                step: "Processing encountered an error",
            }));
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
                        Inspect footage with real-time YOLO object tracking & scene intelligence.
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

            {/* Live Progress Bar Section */}
            {uploading && (
                <div className="bg-slate-950 border border-blue-900/40 rounded-xl p-4 space-y-3 animate-in fade-in duration-300">
                    <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping"></span>
                            <span className="font-semibold text-slate-200">
                                {progressState.step || "Processing..."}
                            </span>
                        </div>
                        <span className="font-mono font-bold text-blue-400">
                            {progressState.progress}%
                        </span>
                    </div>

                    {/* Glowing Progress Bar */}
                    <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden border border-slate-800">
                        <div
                            className="bg-gradient-to-r from-blue-600 via-indigo-500 to-cyan-400 h-full rounded-full transition-all duration-300 ease-out shadow-[0_0_12px_rgba(59,130,246,0.6)]"
                            style={{ width: `${Math.max(5, progressState.progress)}%` }}
                        ></div>
                    </div>

                    {/* Live Telemetry Badges */}
                    <div className="grid grid-cols-3 gap-2 text-[10px] font-mono text-slate-400 pt-1 border-t border-slate-900">
                        <div className="bg-slate-900/80 px-2 py-1 rounded border border-slate-800/60 flex items-center justify-between">
                            <span>🎞️ Frame:</span>
                            <span className="font-bold text-slate-200">
                                {progressState.current_frame} / {progressState.total_frames || "--"}
                            </span>
                        </div>
                        <div className="bg-slate-900/80 px-2 py-1 rounded border border-slate-800/60 flex items-center justify-between">
                            <span>🎯 Objects:</span>
                            <span className="font-bold text-emerald-400">
                                {progressState.unique_tracks || progressState.detections}
                            </span>
                        </div>
                        <div className="bg-slate-900/80 px-2 py-1 rounded border border-slate-800/60 flex items-center justify-between">
                            <span>⏱️ Time:</span>
                            <span className="font-bold text-blue-300">
                                {progressState.elapsed_sec.toFixed(1)}s
                            </span>
                        </div>
                    </div>
                </div>
            )}

            {/* Error Message */}
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
                        <span>Analyzing Footage ({progressState.progress}%)...</span>
                    </>
                ) : (
                    <span>Analyze Surveillance Video ▶</span>
                )}
            </button>
        </div>
    );
}

export default VideoUpload;