import { useEffect, useState } from "react";
import api from "../../api/api";
import type { Event } from "../../types/event";

function eventLabel(event: Event): string {
    if (event.event_type === "VEHICLE_COLLISION") {
        return `⚠️ VEHICLE COLLISION ALERT (Track #${event.track_id})`;
    }
    if (event.event_type === "TRACK_LOST") {
        const cls = event.class_name ? ` (${event.class_name})` : "";
        return `Track #${event.track_id}${cls} left scene`;
    }
    if (event.event_type === "PERSON_ENTERED") {
        return `Person entered (Track #${event.track_id})`;
    }
    if (event.class_name) {
        return `${event.class_name} detected (Track #${event.track_id})`;
    }
    return event.event_type ?? "Event";
}

function eventColor(eventType?: string): string {
    switch (eventType) {
        case "VEHICLE_COLLISION":
            return "text-rose-400 font-semibold flex items-center gap-1";
        case "PERSON_ENTERED":
            return "text-emerald-400";
        case "OBJECT_DETECTED":
            return "text-blue-400";
        case "TRACK_LOST":
            return "text-slate-400";
        default:
            return "text-slate-300";
    }
}

function EventTimeline() {
    const [events, setEvents] = useState<Event[]>([]);
    const [currentTime, setCurrentTime] = useState("");

    useEffect(() => {
        const updateClock = () => setCurrentTime(new Date().toLocaleTimeString());
        updateClock();
        const clockInterval = setInterval(updateClock, 1000);
        return () => clearInterval(clockInterval);
    }, []);

    useEffect(() => {
        const loadEvents = () => {
            api.get<Event[]>("/events")
                .then((response) => setEvents(response.data))
                .catch(console.error);
        };

        loadEvents();
        const interval = setInterval(loadEvents, 5000); // poll every 5s
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="text-base">🕒</span>
                    <h2 className="text-base font-bold text-slate-200">Surveillance Event Log</h2>
                </div>
                <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2 py-1 rounded border border-slate-800">
                    {currentTime || "Loading..."}
                </span>
            </div>

            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                {events.length === 0 ? (
                    <div className="flex items-center justify-between p-2.5 bg-slate-950 rounded-lg border border-slate-800/80 text-xs">
                        <span className="text-slate-400 italic">Awaiting detection events...</span>
                        <span className="text-slate-500 font-mono">--:--:--</span>
                    </div>
                ) : (
                    events.map((event) => (
                        <div
                            key={event.event_id}
                            className={`flex items-center justify-between p-2.5 bg-slate-950 rounded-lg border text-xs transition-colors ${
                                event.event_type === "VEHICLE_COLLISION"
                                    ? "border-rose-800/60 bg-rose-950/20"
                                    : "border-slate-800/80 hover:border-slate-700"
                            }`}
                        >
                            <span className={eventColor(event.event_type)}>
                                {eventLabel(event)}
                            </span>
                            <span className="text-slate-500 font-mono">
                                {new Date(event.timestamp).toLocaleTimeString()}
                            </span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

export default EventTimeline;
