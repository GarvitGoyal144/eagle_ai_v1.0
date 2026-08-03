import { useState } from "react";
import CameraControls from "../components/Camera/CameraControls";
import LiveCamera from "../components/Camera/LiveCamera";
import VideoUpload from "../components/Video/VideoUpload";
import VideoStatus from "../components/Video/VideoStatus";
import EventTimeline from "../components/Timeline/EventTimeline";
import ChatPanel from "../components/Chat/ChatPanel";
import api from "../api/api";

function Dashboard() {
    const [running, setRunning] = useState(false);
    const [aiEnabled, setAiEnabled] = useState(true);
    const [activeMode, setActiveMode] = useState<"live" | "video">("live");

    const handleStartVideoProcessing = async () => {
        // Stop the live camera if active
        if (running) {
            try {
                await api.post("/camera/stop");
                setRunning(false);
            } catch (err) {
                console.error(err);
            }
        }
        setActiveMode("video");
    };

    const handleCameraStart = () => {
        setActiveMode("live");
    };

    return (
        <div className="min-h-screen bg-slate-950 text-slate-100 p-4 font-sans flex flex-col justify-between">
            {/* Top Bar */}
            <header className="max-w-7xl w-full mx-auto mb-4 border-b border-slate-800 pb-3 flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-extrabold tracking-tight text-blue-500">
                        🦅 Eagle AI Surveillance
                    </h1>
                    <p className="text-xs text-slate-400">
                        Control Panel &amp; Real-time Vision Analytics
                    </p>
                </div>
            </header>

            {/* Main Grid: Left Side Feed/Timeline (65%), Right Side Controls/Chat (35%) */}
            <main className="max-w-7xl w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1">
                
                {/* LEFT COLUMN: Main Visual Monitor + Event Timeline */}
                <div className="lg:col-span-8 flex flex-col gap-4">
                    {/* Dynamic Feed Display */}
                    <div className="flex-1">
                        {activeMode === "live" ? (
                            <LiveCamera running={running} aiEnabled={aiEnabled} />
                        ) : (
                            <VideoStatus />
                        )}
                    </div>

                    {/* Timeline Positioned directly under video output */}
                    <EventTimeline />
                </div>

                {/* RIGHT COLUMN: Camera Controls, Video Upload, AI Chat */}
                <div className="lg:col-span-4 flex flex-col gap-4">
                    <CameraControls 
                        running={running} 
                        setRunning={(val) => {
                            setRunning(val);
                            if (val) handleCameraStart();
                        }}
                        aiEnabled={aiEnabled}
                        setAiEnabled={setAiEnabled}
                    />
                    
                    <VideoUpload onStartProcessing={handleStartVideoProcessing} />
                    
                    <div className="flex-1">
                        <ChatPanel />
                    </div>
                </div>

            </main>
        </div>
    );
}

export default Dashboard;