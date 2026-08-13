import type { VideoInsights } from "../../types/event";

interface Props {
    insights: VideoInsights;
}

function VideoInsightsCard({ insights }: Props) {
    const counts = insights.class_counts || {};

    // Category aggregation
    const vehicleClasses = ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"];
    const personClasses = ["person", "pedestrian"];
    
    let totalVehicles = 0;
    const vehicleBreakdown: [string, number][] = [];
    
    let totalPersons = 0;
    const personBreakdown: [string, number][] = [];

    let totalOther = 0;
    const otherBreakdown: [string, number][] = [];

    Object.entries(counts).forEach(([cls, count]) => {
        const lower = cls.toLowerCase();
        if (vehicleClasses.some(v => lower.includes(v))) {
            totalVehicles += count;
            vehicleBreakdown.push([cls, count]);
        } else if (personClasses.some(p => lower.includes(p))) {
            totalPersons += count;
            personBreakdown.push([cls, count]);
        } else {
            totalOther += count;
            otherBreakdown.push([cls, count]);
        }
    });

    vehicleBreakdown.sort((a, b) => b[1] - a[1]);
    personBreakdown.sort((a, b) => b[1] - a[1]);
    otherBreakdown.sort((a, b) => b[1] - a[1]);

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
                    <h2 className="text-base font-bold text-slate-100">
                        Surveillance Intelligence Overview
                    </h2>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
                    <span className="bg-slate-950 px-2.5 py-1 rounded border border-slate-800">
                        ⏱️ {insights.duration_seconds}s video
                    </span>
                    <span className="bg-slate-950 px-2.5 py-1 rounded border border-slate-800 text-blue-400">
                        ⚡ {insights.processing_time_seconds.toFixed(1)}s processed
                    </span>
                </div>
            </div>

            {/* Category Mapping Cards (Verkada / Surveillance Workstation Style) */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                
                {/* 1. Vehicles Category */}
                <div className="bg-slate-950 rounded-xl p-3.5 border border-blue-500/20 flex flex-col justify-between space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-blue-400 flex items-center gap-1.5">
                            🚗 Vehicles
                        </span>
                        <span className="text-xl font-extrabold text-blue-200 font-mono">
                            {totalVehicles}
                        </span>
                    </div>
                    <div className="flex flex-wrap gap-1 pt-1 border-t border-slate-800/60 min-h-[24px]">
                        {vehicleBreakdown.length > 0 ? (
                            vehicleBreakdown.map(([cls, count]) => (
                                <span key={cls} className="text-[10px] bg-blue-950 text-blue-300 border border-blue-800/60 px-1.5 py-0.5 rounded flex items-center gap-1">
                                    {cls} <span className="font-bold text-white bg-blue-900/80 px-1 rounded">{count}</span>
                                </span>
                            ))
                        ) : (
                            <span className="text-[10px] text-slate-600 italic">None detected</span>
                        )}
                    </div>
                </div>

                {/* 2. Persons Category */}
                <div className="bg-slate-950 rounded-xl p-3.5 border border-emerald-500/20 flex flex-col justify-between space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                            👤 Pedestrians / People
                        </span>
                        <span className="text-xl font-extrabold text-emerald-200 font-mono">
                            {totalPersons}
                        </span>
                    </div>
                    <div className="flex flex-wrap gap-1 pt-1 border-t border-slate-800/60 min-h-[24px]">
                        {personBreakdown.length > 0 ? (
                            personBreakdown.map(([cls, count]) => (
                                <span key={cls} className="text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800/60 px-1.5 py-0.5 rounded flex items-center gap-1">
                                    {cls} <span className="font-bold text-white bg-emerald-900/80 px-1 rounded">{count}</span>
                                </span>
                            ))
                        ) : (
                            <span className="text-[10px] text-slate-600 italic">None detected</span>
                        )}
                    </div>
                </div>

                {/* 3. Operational Telemetry & Tracks */}
                <div className="bg-slate-950 rounded-xl p-3.5 border border-amber-500/20 flex flex-col justify-between space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-amber-400 flex items-center gap-1.5">
                            🎯 Unique Tracks
                        </span>
                        <span className="text-xl font-extrabold text-amber-200 font-mono">
                            {insights.unique_tracks}
                        </span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1 border-t border-slate-800/60">
                        <span>Total Detections:</span>
                        <span className="font-bold text-slate-200">{insights.total_detections}</span>
                    </div>
                </div>

            </div>

            {/* Prompt Banner */}
            <div className="bg-gradient-to-r from-blue-950/40 via-slate-950 to-blue-950/40 border border-blue-800/30 rounded-lg p-2.5 text-center">
                <p className="text-xs text-slate-300 flex items-center justify-center gap-2">
                    <span>💡</span>
                    <span>Ask the AI assistant: <strong className="text-blue-300 font-semibold">"Describe the traffic scene"</strong> or <strong className="text-blue-300 font-semibold">"Were there any collisions or accidents?"</strong></span>
                </p>
            </div>
        </div>
    );
}

export default VideoInsightsCard;
