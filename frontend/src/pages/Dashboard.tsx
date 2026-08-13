import { useState } from "react";
import VideoUpload from "../components/Video/VideoUpload";
import VideoInsightsCard from "../components/Video/VideoInsightsCard";
import FrameGallery from "../components/Video/FrameGallery";
import EventTimeline from "../components/Timeline/EventTimeline";
import ChatPanel from "../components/Chat/ChatPanel";
import type { VideoInsights } from "../types/event";

function Dashboard() {
    const [insights, setInsights] = useState<VideoInsights | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);

    const handleInsightsReady = (newInsights: VideoInsights, newSessionId: string) => {
        setInsights(newInsights);
        setSessionId(newSessionId);
    };

    return (
        <div className="h-screen max-h-screen bg-slate-950 text-slate-100 p-4 font-sans flex flex-col">
            {/* Top Bar */}
            <header className="max-w-[1400px] w-full mx-auto mb-4 border-b border-slate-800 pb-3 flex items-center justify-between shrink-0">
                <div>
                    <h1 className="text-2xl font-extrabold tracking-tight text-blue-500 flex items-center gap-2">
                        🦅 Eagle AI <span className="text-sm font-medium bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded border border-blue-500/20">v1.1</span>
                    </h1>
                    <p className="text-xs text-slate-400 mt-1">
                        Video Intelligence & Analysis
                    </p>
                </div>
                
                <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span className="text-xs text-slate-300 font-medium">Video Mode Active</span>
                </div>
            </header>

            {/* Main Grid: Left Side Feed/Timeline (65%), Right Side Chat (35%) */}
            <main className="max-w-[1400px] w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-0 overflow-hidden">
                
                {/* LEFT COLUMN: Video Upload, Insights, Timeline */}
                <div className="lg:col-span-8 flex flex-col gap-4 overflow-y-auto pr-1 pb-4 scrollbar-thin scrollbar-thumb-slate-700">
                    <VideoUpload onInsightsReady={handleInsightsReady} />
                    
                    {insights && (
                        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <VideoInsightsCard insights={insights} />
                        </div>
                    )}

                    {insights && (
                        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 delay-150">
                            <FrameGallery sessionId={sessionId!} />
                        </div>
                    )}

                    {/* Timeline Positioned directly under insights */}
                    <EventTimeline />
                </div>

                {/* RIGHT COLUMN: AI Chat */}
                <div className="lg:col-span-4 flex flex-col h-full overflow-hidden">
                    <ChatPanel sessionId={sessionId || undefined} />
                </div>

            </main>
        </div>
    );
}

export default Dashboard;