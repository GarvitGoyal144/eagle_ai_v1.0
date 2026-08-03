export interface Event {
    event_id: string;
    event_type?: string;
    track_id: number;
    class_id?: number;
    class_name?: string;
    confidence?: number;
    bbox?: number[];
    camera: string;
    timestamp: string;
}

export interface ChatMessage {
    role: "user" | "assistant";
    content: string;
    sources?: ChatSource[];
}

export interface ChatSource {
    type: "scene" | "event";
    event_type?: string;
    track_id?: number;
    class_name?: string;
    timestamp?: string;
    score: number;
}

export interface ChatResponse {
    answer: string;
    sources: ChatSource[];
}