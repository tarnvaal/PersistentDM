import { useMemo, useState, useRef, useEffect } from "react";

function App() {
  const apiBase = useMemo(() => {
    const configured = import.meta.env.VITE_API_BASE_URL;
    if (configured) return configured;
    const host = window.location.hostname || "127.0.0.1";
    return `http://${host}:8000`;
  }, []);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [expanded, setExpanded] = useState({});
  const messagesRef = useRef(null);

  useEffect(() => {
    const el = messagesRef.current;
    if (!el) return;
    // Scroll to bottom on new messages
    el.scrollTop = el.scrollHeight;
  }, [history]);

  async function handleSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;

    setHistory((h) => [...h, { role: "user", content: text }]);
    setInput("");
    setSending(true);

    const typingId = `typing-${Date.now()}`;
    setHistory((h) => [...h, { role: "assistant", type: "typing", content: "", id: typingId }]);

    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) {
        let errMsg = res.statusText || "Request failed";
        try {
          const err = await res.json();
          if (err && err.detail && err.detail.message) errMsg = err.detail.message;
        } catch (_) {}
        setHistory((h) => [...h, { role: "assistant", content: `[Error] ${errMsg}` }]);
      } else {
        const data = await res.json();
        const trimmedReply = (data.reply || "").trimEnd();
        setHistory((h) => [
          ...h,
          { role: "assistant", content: trimmedReply, context: data.context, relevance: data.relevance }
        ]);
      }
    } catch (err) {
      setHistory((h) => [...h, { role: "assistant", content: `[Network error] ${String(err)}` }]);
    } finally {
      setHistory((h) => h.filter((m) => m.id !== typingId));
      setSending(false);
    }
  }

  function toggleExpanded(idx) {
    setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }));
  }

  async function handleClear() {
    try {
      const res = await fetch(`${apiBase}/chat/clear`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clear: true }),
      });
      if (res.ok) {
        setHistory([]);
      }
    } catch (_) {}
  }

  return (
    <div id="chat-app" className="max-w-[800px] mt-4 mx-auto">
      <div className="bg-[#2C3539] border border-[#444] rounded-xl p-3 pb-3 flex flex-col h-[calc(100vh-var(--chat-offset))] shadow-[0_4px_12px_rgba(0,0,0,0.3)]">
        <div id="controls" className="flex gap-2 mb-2 items-center justify-between">
          <div className="flex items-center gap-2 flex-1"></div>
          <div className="flex items-center gap-2">
            <button
              className="inline-flex items-center justify-center w-9 h-9 rounded-[10px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] transition-colors ease-linear hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600] active:translate-y-px"
              id="btn-clear"
              aria-label="Clear chat"
              onClick={handleClear}
              disabled={sending}
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21a48.108 48.108 0 0 0-3.478-.397m-12 .397a48.108 48.108 0 0 1 3.478-.397m7.5 0V4.5A2.25 2.25 0 0 0 14.25 2.25h-4.5A2.25 2.25 0 0 0 7.5 4.5V5.79m10.5 0a48.667 48.667 0 0 1-7.5 0M5.25 6.75 4.5 19.5A2.25 2.25 0 0 0 6.75 21h10.5A2.25 2.25 0 0 0 19.5 19.5l-.75-12.75" />
              </svg>
            </button>
          </div>
        </div>

        <div id="messages" ref={messagesRef} className="flex-1 overflow-y-auto p-2 bg-[#1e2a30] rounded-lg mb-4 flex flex-col gap-2">
          {history.map((item, idx) => {
            if (item.type === "typing") {
              return (
                <div
                  key={item.id || idx}
                  className="px-4 py-2 my-1 rounded-[20px] max-w-[90%] leading-[1.45] font-medium text-[1.05rem] bg-[#4a555c] text-[#f1f1f1] self-start italic typing"
                >
                  DM is typing
                </div>
              );
            }
            const isUser = item.role === "user";
            return (
              <div
                key={idx}
                className={`flex flex-col ${
                  isUser ? "self-end items-end" : "self-start items-start"
                }`}
              >
                <div
                  className={
                    `relative px-4 py-2 my-1 rounded-[20px] whitespace-pre-wrap min-h-[64px] leading-[1.45] font-medium text-[1.05rem] ${
                      !isUser ? "pb-8 " : ""
                    }` +
                    (isUser
                      ? "max-w-[90%] min-w-[300px] bg-[#FF6600] text-[#111111]"
                      : "max-w-[90%] min-w-[300px] bg-[#4a555c] text-[#f1f1f1]")
                  }
                >
                  {(isUser ? "You: " : "DM: ") + item.content}
                  {!isUser && (
                    <button
                      className="absolute left-2 -bottom-3 inline-flex items-center justify-center w-7 h-7 rounded-full border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600] active:translate-y-px"
                      aria-label="Expand details"
                      type="button"
                      onClick={() => toggleExpanded(idx)}
                    >
                      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M19 7l-7 7-7-7" />
                        <path d="M19 13l-7 7-7-7" />
                      </svg>
                    </button>
                  )}
                </div>
                {!isUser && expanded[idx] && (
                  <div className="mt-2 text-sm bg-[#1e2a30] text-[#c9d1d9] rounded-lg p-3 w-[95%] max-w-[95%]">
                    {item.relevance && (item.relevance.memories?.length > 0 || item.relevance.npcs?.length > 0) && (
                      <div className="mb-2">
                        {item.relevance.memories?.length > 0 && (
                          <div className="mb-2">
                            <div className="text-xs uppercase tracking-wide opacity-70 mb-1">Relevant memories</div>
                            <ul className="list-disc list-inside space-y-1">
                              {item.relevance.memories.map((m, i) => (
                                <li key={i} className="text-xs">
                                  <span className="opacity-80">[{m.type}]</span> {m.summary}
                                  {m.entities && m.entities.length > 0 && (
                                    <span className="opacity-60"> (entities: {m.entities.join(", ")})</span>
                                  )}
                                  <span className="opacity-60"> — score {m.score?.toFixed ? m.score.toFixed(2) : m.score}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {item.relevance.npcs?.length > 0 && (
                          <div className="mb-2">
                            <div className="text-xs uppercase tracking-wide opacity-70 mb-1">Relevant NPCs</div>
                            <ul className="list-disc list-inside space-y-1">
                              {item.relevance.npcs.map((n, i) => (
                                <li key={i} className="text-xs">
                                  <span className="font-medium">{n.name}</span>
                                  {n.intent && <span className="opacity-80"> — intent: {n.intent}</span>}
                                  {n.last_seen_location && <span className="opacity-60"> — last seen: {n.last_seen_location}</span>}
                                  <span className="opacity-60"> — score {n.score?.toFixed ? n.score.toFixed(2) : n.score}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        <div className="h-px bg-[#344046] my-2" />
                      </div>
                    )}
                    {item.context ? (
                      <>
                        <div className="text-xs uppercase tracking-wide opacity-70 mb-1">Context sent to model</div>
                        <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed">{item.context}</pre>
                      </>
                    ) : (
                      <span className="italic opacity-60">No world context available</span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <form id="chat-form" onSubmit={handleSubmit} className="flex gap-2">
          <input
            id="message-input"
            type="text"
            placeholder="Type a message and press Enter…"
            autoComplete="off"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 px-4 py-3 border border-[#4a555c] bg-[#1e2a30] text-[#f0f0f0] rounded-[20px] text-[1.05rem] font-sans placeholder:text-[#889] focus:outline-none focus:border-[#FF6600] focus:ring-2 focus:ring-[rgba(255,102,0,0.3)]"
          />
          <button
            id="send-btn"
            type="submit"
            disabled={sending || input.trim().length === 0}
            className="px-4 py-3 bg-[#FF6600] text-[#111111] rounded-[20px] cursor-pointer font-semibold text-[1rem] transition-colors hover:bg-[#FF8533] disabled:bg-[#536267] disabled:text-[#aaa] disabled:opacity-70 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
