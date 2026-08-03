import { streamUrl } from "../../api/api";

interface Props {
    running: boolean;
    aiEnabled: boolean;
}

function LiveCamera({ running, aiEnabled }: Props) {
    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col h-full">
            <div className="flex items-center justify-between mb-3">
                <h2 className="text-base font-bold text-slate-200">Camera Feed</h2>
                {running && (
                    <span
                        className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border ${
                            aiEnabled
                                ? "bg-violet-500/10 text-violet-400 border-violet-500/20"
                                : "bg-slate-700/30 text-slate-400 border-slate-600/30"
                        }`}
                    >
                        {aiEnabled ? "🧠 AI Vision" : "📹 Raw Feed"}
                    </span>
                )}
            </div>

            <div className="flex-1 min-h-[360px] bg-slate-950 rounded-lg border border-slate-800/80 overflow-hidden flex items-center justify-center">
                {!running ? (
                    <div className="text-center p-6">
                        <p className="text-slate-500 text-sm font-medium">Camera is currently stopped</p>
                        <p className="text-slate-600 text-xs mt-1">Press "Start Camera" to initiate live feed</p>
                    </div>
                ) : (
                    <img
                        src={streamUrl}
                        alt="Live Camera Stream"
                        className="w-full h-full object-contain"
                    />
                )}
            </div>
        </div>
    );
}

export default LiveCamera;
