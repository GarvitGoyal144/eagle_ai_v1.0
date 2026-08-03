import { useEffect, useState } from "react";
import api from "../../api/api";
import type { Event } from "../../types/event";

function eventLabel(event: Event): string {
    if (event.event_type === "TRACK_LOST") {
        return `Track #${event.track_id} lost`;
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
        case "PERSON_ENTERED":
            return "text-emerald-400";
        case "OBJECT_DETECTED":
            return "text-blue-400";
        case "TRACK_LOST":
            return "text-amber-400";
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
        const interval = setInterval(loadEvents, 1000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
                <h2 className="text-base font-bold text-slate-200">Event Timeline</h2>
                <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2 py-1 rounded border border-slate-800">
                    {currentTime || "Loading..."}
                </span>
            </div>

            <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
                {events.length === 0 ? (
                    <div className="flex items-center justify-between p-2.5 bg-slate-950 rounded-lg border border-slate-800/80 text-xs">
                        <span className="text-slate-400 italic">Awaiting detection events...</span>
                        <span className="text-slate-500 font-mono">--:--:--</span>
                    </div>
                ) : (
                    events.map((event) => (
                        <div
                            key={event.event_id}
                            className="flex items-center justify-between p-2.5 bg-slate-950 rounded-lg border border-slate-800/80 text-xs"
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
