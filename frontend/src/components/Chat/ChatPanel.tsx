import { useState, useRef, useEffect } from "react";
import api from "../../api/api";
import type { ChatMessage, ChatResponse } from "../../types/event";

function ChatPanel() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const sendMessage = async () => {
        const question = input.trim();
        if (!question || loading) return;

        // Add user message to chat
        const userMsg: ChatMessage = { role: "user", content: question };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setLoading(true);

        try {
            // Build conversation history for multi-turn (exclude sources)
            const history = messages.map((m) => ({
                role: m.role,
                content: m.content,
            }));

            const res = await api.post<ChatResponse>("/chat", {
                question,
                history,
            });

            const assistantMsg: ChatMessage = {
                role: "assistant",
                content: res.data.answer,
                sources: res.data.sources,
            };
            setMessages((prev) => [...prev, assistantMsg]);
        } catch (err: any) {
            const detail = err?.response?.data?.detail || err?.message || "Backend server unreachable";
            const errorMsg: ChatMessage = {
                role: "assistant",
                content: `⚠️ Could not connect to AI backend (${detail}). Ensure backend is running and VITE_API_URL is configured.`,
            };
            setMessages((prev) => [...prev, errorMsg]);
            console.error("Chat error:", err);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const clearChat = () => {
        setMessages([]);
    };

    const formatTimestamp = (ts?: string) => {
        if (!ts) return "";
        try {
            return new Date(ts).toLocaleTimeString();
        } catch {
            return ts;
        }
    };

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col h-full justify-between space-y-3">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-base font-bold text-slate-200">
                        AI Chat Assistant
                    </h2>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                        Ask about surveillance events • Powered by CLIP + Llama
                    </p>
                </div>
                {messages.length > 0 && (
                    <button
                        onClick={clearChat}
                        className="text-[10px] text-slate-500 hover:text-slate-300 px-2 py-1 rounded border border-slate-800 hover:border-slate-600 transition-colors"
                    >
                        Clear
                    </button>
                )}
            </div>

            {/* Messages Area */}
            <div className="flex-1 min-h-[140px] max-h-[320px] bg-slate-950 rounded-lg border border-slate-800/80 overflow-y-auto p-3 space-y-3">
                {messages.length === 0 ? (
                    <div className="h-full flex items-center justify-center">
                        <div className="text-center space-y-2">
                            <p className="text-slate-600 text-xs">🧠</p>
                            <p className="text-slate-500 text-xs italic">
                                Ask me anything about what the camera has seen...
                            </p>
                            <div className="flex flex-wrap gap-1.5 justify-center mt-2">
                                {[
                                    "Who entered recently?",
                                    "Any suspicious activity?",
                                    "How many people today?",
                                ].map((suggestion) => (
                                    <button
                                        key={suggestion}
                                        onClick={() => {
                                            setInput(suggestion);
                                        }}
                                        className="text-[10px] text-slate-400 bg-slate-900 border border-slate-800 rounded-full px-2.5 py-1 hover:border-slate-600 hover:text-slate-300 transition-colors"
                                    >
                                        {suggestion}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                ) : (
                    messages.map((msg, i) => (
                        <div key={i}>
                            {/* Message bubble */}
                            <div
                                className={`flex ${
                                    msg.role === "user" ? "justify-end" : "justify-start"
                                }`}
                            >
                                <div
                                    className={`max-w-[85%] rounded-lg px-3 py-2 text-xs leading-relaxed ${
                                        msg.role === "user"
                                            ? "bg-blue-600/20 text-blue-200 border border-blue-500/20"
                                            : "bg-slate-800/60 text-slate-300 border border-slate-700/40"
                                    }`}
                                >
                                    {msg.content}
                                </div>
                            </div>

                            {/* Source citations (assistant messages only) */}
                            {msg.role === "assistant" &&
                                msg.sources &&
                                msg.sources.length > 0 && (
                                    <div className="mt-1.5 ml-1 space-y-1">
                                        <p className="text-[9px] text-slate-600 font-medium uppercase tracking-wider">
                                            Sources
                                        </p>
                                        {msg.sources.slice(0, 3).map((src, j) => (
                                            <div
                                                key={j}
                                                className="text-[10px] text-slate-500 flex items-center gap-1.5"
                                            >
                                                <span
                                                    className={`w-1.5 h-1.5 rounded-full ${
                                                        src.type === "scene"
                                                            ? "bg-cyan-500/60"
                                                            : "bg-violet-500/60"
                                                    }`}
                                                />
                                                <span>
                                                    {src.type === "scene"
                                                        ? `Scene ${formatTimestamp(src.timestamp)}`
                                                        : `${src.event_type} #${src.track_id} ${formatTimestamp(src.timestamp)}`}
                                                </span>
                                                <span className="text-slate-600">
                                                    {(src.score * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                        </div>
                    ))
                )}

                {/* Loading indicator */}
                {loading && (
                    <div className="flex justify-start">
                        <div className="bg-slate-800/60 border border-slate-700/40 rounded-lg px-3 py-2 text-xs text-slate-400">
                            <span className="inline-flex gap-1">
                                <span className="animate-bounce" style={{ animationDelay: "0ms" }}>·</span>
                                <span className="animate-bounce" style={{ animationDelay: "150ms" }}>·</span>
                                <span className="animate-bounce" style={{ animationDelay: "300ms" }}>·</span>
                            </span>
                            <span className="ml-1.5 text-slate-500">Thinking...</span>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="flex gap-2">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about events..."
                    disabled={loading}
                    className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:border-blue-500/40 disabled:opacity-50 transition-colors"
                />
                <button
                    onClick={sendMessage}
                    disabled={loading || !input.trim()}
                    className="bg-blue-600 hover:bg-blue-500 text-white font-semibold px-3 py-1.5 rounded-lg text-xs transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                    {loading ? "..." : "Ask"}
                </button>
            </div>
        </div>
    );
}

export default ChatPanel;