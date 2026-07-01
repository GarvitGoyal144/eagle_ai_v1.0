import { useEffect, useState } from "react";
import api from "../api/api";
import type { Event } from "../types/event";

function EventTimeline() {

    const [events, setEvents] = useState<Event[]>([]);

    useEffect(() => {

        const loadEvents = () => {
            api.get("/events")
                .then((response) => {
                    setEvents(response.data);
                })
                .catch(console.error);
        };

        loadEvents();

        const interval = setInterval(loadEvents, 1000);

        return () => clearInterval(interval);

    }, []);

    return (

        <div>

            <h2 className="text-2xl mb-4">
                Live Timeline
            </h2>

            <div className="space-y-2">

                {
                    events.map((event) => (

                        <div
                            key={event.event_id}
                            className="border rounded-lg p-3"
                        >

                            <div>

                                <strong>{event.class_name}</strong>

                                {" "}

                                #{event.track_id}

                            </div>

                            <div>

                                {new Date(event.timestamp).toLocaleTimeString()}

                            </div>

                        </div>

                    ))
                }

            </div>

        </div>

    );

}

export default EventTimeline;