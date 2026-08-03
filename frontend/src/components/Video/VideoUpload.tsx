import { useState } from "react";
import api from "../../api/api";

interface Props {
    onStartProcessing: () => void;
}

function VideoUpload({ onStartProcessing }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);

    const handleProcess = async () => {
        if (!file) return;
        setUploading(true);
        onStartProcessing(); // Stops the live camera feed automatically in UI

        const formData = new FormData();
        formData.append("file", file);

        try {
            await api.post("/video/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            console.log("Video uploaded successfully");
        } catch (err) {
            console.error("Failed to upload video:", err);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg space-y-3">
            <h2 className="text-base font-bold text-slate-200">
                Video Upload
            </h2>

            <div className="flex items-center gap-3">
                <input
                    type="file"
                    accept="video/*"
                    id="video-input"
                    onChange={(e) => {
                        if (e.target.files) {
                            setFile(e.target.files[0]);
                        }
                    }}
                    className="hidden"
                />
                
                <label
                    htmlFor="video-input"
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-2 px-3 rounded-lg border border-slate-700 cursor-pointer transition-colors"
                >
                    Choose File
                </label>
                <span className="text-xs text-slate-400 truncate max-w-[150px]">
                    {file ? file.name : "No file chosen"}
                </span>
            </div>

            <button
                disabled={!file || uploading}
                onClick={handleProcess}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold py-2 px-4 rounded-lg transition-colors shadow"
            >
                {uploading ? "Uploading & Processing..." : "Process Video"}
            </button>
        </div>
    );
}

export default VideoUpload;