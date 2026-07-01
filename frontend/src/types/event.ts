export interface Event {
    event_id: string;
    event_type?: string;
    track_id: number;
    class_id: number;
    class_name: string;
    confidence: number;
    bbox: number[];
    camera: string;
    timestamp: string;
}