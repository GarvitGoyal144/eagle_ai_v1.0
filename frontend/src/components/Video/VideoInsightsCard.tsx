import type { VideoInsights } from "../../types/event";

interface Props {
    insights: VideoInsights;
}

function VideoInsightsCard({ insights }: Props) {
    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h2 className="text-base font-bold text-slate-200 flex items-center gap-2">
                    <span className="text-green-500">✅</span> Video Analyzed
                </h2>
                <span className="text-xs text-slate-400 font-mono">
                    {insights.processing_time_seconds.toFixed(1)}s proc
                </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-950 rounded-lg p-3 border border-slate-800/80">
                    <p className="text-slate-500 text-xs mb-1">Detections</p>
                    <p className="text-slate-200 font-bold text-xl">{insights.total_detections}</p>
                </div>
                
                <div className="bg-slate-950 rounded-lg p-3 border border-slate-800/80">
                    <p className="text-slate-500 text-xs mb-1">Unique Tracks</p>
                    <p className="text-slate-200 font-bold text-xl">{insights.unique_tracks}</p>
                </div>

                <div className="bg-slate-950 rounded-lg p-3 border border-slate-800/80">
                    <p className="text-slate-500 text-xs mb-1">Duration</p>
                    <p className="text-slate-200 font-bold text-xl">{insights.duration_seconds}s</p>
                </div>

                <div className="bg-slate-950 rounded-lg p-3 border border-slate-800/80">
                    <p className="text-slate-500 text-xs mb-1">Top Classes</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                        {Object.entries(insights.class_counts || {})
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3)
                            .map(([cls, count]) => (
                                <span key={cls} className="text-[10px] bg-blue-900/40 text-blue-300 border border-blue-700/50 px-1.5 py-0.5 rounded">
                                    {cls} <span className="opacity-60 ml-0.5">{count}</span>
                                </span>
                            ))}
                        {Object.keys(insights.class_counts || {}).length === 0 && (
                            <span className="text-[10px] text-slate-500 italic">None found</span>
                        )}
                    </div>
                </div>
            </div>

            <div className="pt-2 text-center">
                <p className="text-xs font-semibold text-blue-400">
                    💬 Ask the AI chat anything about this video →
                </p>
            </div>
        </div>
    );
}

export default VideoInsightsCard;
