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
    visual_refs?: VisualRef[];
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
    visual_refs?: VisualRef[];
    provider?: string;
}

export interface VisualRef {
    event_id: string;
    label: string;
    timestamp_sec: number;
    timestamp_display: string;
    frame_url: string;
    clip_url: string;
}

export interface VideoInsights {
    filename: string;
    duration_seconds: number;
    total_frames_sampled: number;
    total_detections: number;
    unique_tracks: number;
    class_counts: Record<string, number>;
    events_saved: number;
    scene_snapshots_saved: number;
    processing_time_seconds: number;
    session_id: string;
}