import { useEffect } from "react";
import api from "../../api/api";

interface Props {
    running: boolean;
    setRunning: (running: boolean) => void;
    aiEnabled: boolean;
    setAiEnabled: (enabled: boolean) => void;
}

function CameraControls({ running, setRunning, aiEnabled, setAiEnabled }: Props) {
    const fetchStatus = async () => {
        try {
            const res = await api.get("/camera/status");
            setRunning(res.data.is_running);
            setAiEnabled(res.data.ai_enabled);
        } catch (err) {
            console.error("Camera status error:", err);
        }
    };

    useEffect(() => {
        fetchStatus();
    }, []);

    const startCamera = async () => {
        try {
            await api.post("/camera/start");
            await fetchStatus();
        } catch (err) {
            console.error("Failed to start camera:", err);
        }
    };

    const stopCamera = async () => {
        try {
            await api.post("/camera/stop");
            await fetchStatus();
        } catch (err) {
            console.error("Failed to stop camera:", err);
        }
    };

    const toggleAI = async () => {
        try {
            const res = await api.post("/camera/ai/toggle");
            setAiEnabled(res.data.ai_enabled);
        } catch (err) {
            console.error("Failed to toggle AI:", err);
        }
    };

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
                <h2 className="text-base font-bold text-slate-200">Live Camera</h2>
                <span
                    className={`text-xs px-2.5 py-0.5 rounded-full font-medium border ${
                        running
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                            : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                    }`}
                >
                    Status: {running ? "🟢 Running" : "🔴 Stopped"}
                </span>
            </div>

            <div className="flex gap-2">
                <button
                    onClick={startCamera}
                    disabled={running}
                    className="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white text-xs font-semibold py-2 px-3 rounded-lg transition-colors"
                >
                    Start Camera
                </button>

                <button
                    onClick={stopCamera}
                    disabled={!running}
                    className="flex-1 bg-rose-600 hover:bg-rose-500 disabled:opacity-40 text-white text-xs font-semibold py-2 px-3 rounded-lg transition-colors"
                >
                    Stop Camera
                </button>
            </div>

            {/* AI Features Toggle */}
            <div className={`flex items-center justify-between rounded-lg border px-3 py-2.5 transition-colors ${
                aiEnabled
                    ? "bg-violet-500/5 border-violet-500/20"
                    : "bg-slate-800/50 border-slate-700/50"
            }`}>
                <div className="flex items-center gap-2">
                    <span className="text-sm">{aiEnabled ? "🧠" : "📹"}</span>
                    <div>
                        <p className="text-xs font-semibold text-slate-200">
                            {aiEnabled ? "AI Vision Active" : "Raw Feed Mode"}
                        </p>
                        <p className="text-[10px] text-slate-500">
                            {aiEnabled
                                ? "Detection, tracking & annotations"
                                : "No AI processing — direct stream"
                            }
                        </p>
                    </div>
                </div>

                <button
                    onClick={toggleAI}
                    disabled={!running}
                    className={`
                        relative w-11 h-6 rounded-full transition-colors duration-200
                        focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-offset-slate-900
                        disabled:opacity-30 disabled:cursor-not-allowed
                        ${aiEnabled
                            ? "bg-violet-600 focus:ring-violet-500"
                            : "bg-slate-600 focus:ring-slate-500"
                        }
                    `}
                    aria-label="Toggle AI features"
                >
                    <span
                        className={`
                            absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white
                            shadow-sm transition-transform duration-200
                            ${aiEnabled ? "translate-x-5" : "translate-x-0"}
                        `}
                    />
                </button>
            </div>
        </div>
    );
}

export default CameraControls;