function VideoStatus() {
    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
            <h2 className="text-lg font-bold text-slate-200 mb-4">
                Processed Video Output
            </h2>
            <div className="aspect-video w-full bg-slate-950 rounded-lg border border-slate-800/80 flex items-center justify-center p-6">
                <p className="text-slate-500 text-sm font-medium">
                    No processed video available yet.
                </p>
            </div>
        </div>
    );
}

export default VideoStatus;