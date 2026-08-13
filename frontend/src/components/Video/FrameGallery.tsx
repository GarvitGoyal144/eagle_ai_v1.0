interface Props {
    sessionId: string;
}

function FrameGallery({ sessionId }: Props) {
    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg">
            <h2 className="text-base font-bold text-slate-200 mb-3 flex items-center gap-2">
                <span className="text-indigo-500">🖼️</span> Frame Gallery
            </h2>
            <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                <div className="min-w-[200px] h-28 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-center text-slate-500 text-xs">
                    Frames appear in chat when requested.
                </div>
                <div className="min-w-[200px] h-28 bg-slate-950 border border-slate-800/50 rounded-lg flex items-center justify-center text-slate-600 text-xs italic">
                    "Show me the person ({sessionId.slice(0, 4)}...)"
                </div>
            </div>
        </div>
    );
}

export default FrameGallery;
