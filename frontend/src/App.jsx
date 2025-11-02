import { useMemo, useState, useRef, useEffect } from "react";

// Approximate token estimator and sliding-window ingestor
class TextIngestor {
  constructor(text, options, callbacks) {
    this.text = text || "";
    this.options = {
      contextWindow: 32000, // tokens
      baseWindow: 16000,
      worldWindow: 16000,
      stride: 8000,
      checkpointEvery: 10,
      ...options,
    };
    this.callbacks = callbacks || {};
  }

  static estimateTokens(text) {
    // Very rough: ~1.3 tokens per word
    const words = (text.trim().match(/\S+/g) || []).length;
    return Math.max(0, Math.round(words * 1.3));
  }

  static countLines(text) {
    if (!text) return 0;
    const m = text.match(/\n/g);
    return (m ? m.length : 0) + 1;
  }

  async run() {
    const start = performance.now();
    const totalWords = (this.text.trim().match(/\S+/g) || []).length;
    const totalLines = TextIngestor.countLines(this.text);
    const approxTokens = TextIngestor.estimateTokens(this.text);

    // Split text by words for sliding windows
    const words = this.text.trim().length ? this.text.split(/\s+/) : [];
    const tokensPerWord = approxTokens / Math.max(1, totalWords);
    const windowTokens = this.options.baseWindow; // 16k tokens for source slice
    const windowWords = Math.max(1, Math.round(windowTokens / Math.max(1, tokensPerWord)));
    const strideTokens = this.options.stride; // 8k tokens step
    const strideWords = Math.max(1, Math.round(strideTokens / Math.max(1, tokensPerWord)));

    const totalSteps = Math.max(1, Math.ceil(Math.max(0, words.length - windowWords) / strideWords) + 1);
    let consumedWords = 0;
    const checkpoints = [];

    for (let step = 0; step < totalSteps; step++) {
      const startIdx = step * strideWords;
      const endIdx = Math.min(words.length, startIdx + windowWords);
      const chunkWords = words.slice(startIdx, endIdx);
      const chunkText = chunkWords.join(" ");

      // Advance consumed words by stride (no double counting overlap)
      consumedWords = Math.min(words.length, step * strideWords + strideWords);
      const consumedLines = TextIngestor.countLines(words.slice(0, consumedWords).join(" "));
      const progress = Math.min(1, consumedWords / Math.max(1, words.length));

      if (typeof this.callbacks.onProgress === "function") {
        this.callbacks.onProgress({
          progress,
          step,
          totalSteps,
          consumedWords,
          consumedLines,
        });
      }

      // Occasionally emit a naive checkpoint summary
      if (step % Math.max(1, this.options.checkpointEvery) === 0) {
        const snippet = chunkText.split(/(?<=[.!?])\s+/).slice(0, 2).join(" ");
        const summary = snippet || chunkText.slice(0, 200);
        checkpoints.push({ step, summary });
        if (typeof this.callbacks.onCheckpoint === "function") {
          this.callbacks.onCheckpoint({ step, summary });
        }
      }

      // Yield to UI
      // Dynamically decide updates: larger chunks get more UI ticks
      const uiTicks = Math.min(5, 1 + Math.floor(chunkWords.length / Math.max(1, strideWords / 2)));
      for (let i = 0; i < uiTicks; i++) {
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    const ms = Math.max(0, Math.round(performance.now() - start));
    const result = {
      totalWords,
      totalLines,
      approxTokens,
      timeMs: ms,
      checkpoints,
      steps: totalSteps,
    };
    if (typeof this.callbacks.onDone === "function") this.callbacks.onDone(result);
    return result;
  }
}

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
  const [pasteOpen, setPasteOpen] = useState(false);
  const [savedOpen, setSavedOpen] = useState(false);
  const [savedIngests, setSavedIngests] = useState([]);
  const [pasteText, setPasteText] = useState("");
  const messagesRef = useRef(null);
  const activeIngestStreamsRef = useRef(new Map()); // Map of ingestId -> {es: EventSource, timer: interval}
  const cancelledIngestIdsRef = useRef(new Set());
  const [strideWords, setStrideWords] = useState(1000);
  const [strideTouched, setStrideTouched] = useState(false);
  const isAtBottomRef = useRef(true);


  // Load saved ingests when modal opens
  useEffect(() => {
    if (!savedOpen) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${apiBase}/ingest/list`);
        if (!res.ok) throw new Error("list not ok");
        const data = await res.json();
        if (!cancelled) setSavedIngests(Array.isArray(data.ingests) ? data.ingests : []);
      } catch (_) {
        if (!cancelled) setSavedIngests([]);
      }
    })();
    return () => { cancelled = true; };
  }, [savedOpen, apiBase]);

  // Poll model status and update the header dot
  useEffect(() => {
    let timer;
    let cancelled = false;
    const dot = () => document.getElementById("model-status-dot");

    const updateDot = (state) => {
      const el = dot();
      if (!el) return;
      let color = "#666"; // unknown
      if (state === "unloaded" || state === "failed") color = "#c0392b"; // red
      else if (state === "loading") color = "#f39c12"; // yellow
      else if (state === "ready") color = "#2ecc71"; // green
      el.style.backgroundColor = color;
      const label = `Model status: ${state}`;
      el.setAttribute("aria-label", label);
      el.setAttribute("title", label);
    };

    const fetchStatus = async () => {
      try {
        const res = await fetch(`${apiBase}/status`);
        if (!res.ok) throw new Error("status not ok");
        const data = await res.json();
        if (!cancelled) updateDot(data.state || "unknown");
      } catch (_) {
        if (!cancelled) updateDot("unloaded");
      }
    };

    fetchStatus();
    timer = setInterval(fetchStatus, 2000);
    return () => { cancelled = true; if (timer) clearInterval(timer); };
  }, [apiBase]);

  // When expanding a message's details, ensure the expanded content is visible.
  useEffect(() => {
    const el = messagesRef.current;
    if (!el || history.length === 0) return;

    // Find the most recent expanded index (highest index)
    const expandedIndices = Object.entries(expanded)
      .filter(([, v]) => !!v)
      .map(([k]) => Number(k))
      .filter((n) => !Number.isNaN(n));

    if (expandedIndices.length === 0) return;

    const targetIdx = Math.max(...expandedIndices);
    const detailsEl = document.getElementById(`details-${targetIdx}`);
    if (!detailsEl) return;

    // Compute visibility relative to the scroll container and only scroll if needed
    const containerRect = el.getBoundingClientRect();
    const detailsRect = detailsEl.getBoundingClientRect();
    const margin = 8; // small padding when bringing into view

    const belowBottom = detailsRect.bottom > containerRect.bottom;
    const aboveTop = detailsRect.top < containerRect.top;

    if (belowBottom) {
      const delta = (detailsRect.bottom - containerRect.bottom) + margin;
      el.scrollTo({ top: el.scrollTop + delta, behavior: "smooth" });
      return;
    }
    if (aboveTop) {
      const delta = (containerRect.top - detailsRect.top) + margin;
      el.scrollTo({ top: Math.max(0, el.scrollTop - delta), behavior: "smooth" });
      return;
    }
  }, [expanded, history]);

  const scrollToBottom = (behavior = "auto") => {
    const el = messagesRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
  };

  // Stick to bottom only if user is already at bottom
  useEffect(() => {
    if (!messagesRef.current) return;
    if (isAtBottomRef.current) {
      scrollToBottom("auto");
    }
  }, [history]);

  useEffect(() => {
    // Default chunk size is 1000 unless user has edited it
    if (!pasteOpen) return;
    if (strideTouched) return;
    setStrideWords(1000);
  }, [pasteOpen, strideTouched]);

  async function handleSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;

    setHistory((h) => [...h, { role: "user", content: text }]);
    setInput("");
    setSending(true);
    // When the user sends a message, re-enable autoscroll and stick to bottom
    isAtBottomRef.current = true;

    const typingId = `typing-${Date.now()}`;
    setHistory((h) => [...h, { role: "assistant", type: "typing", content: "", id: typingId }]);

    const postFallback = async () => {
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
            { role: "assistant", content: trimmedReply, context: data.context, relevance: data.relevance, ready: true }
          ]);
        }
      } catch (err) {
        setHistory((h) => [...h, { role: "assistant", content: `[Network error] ${String(err)}` }]);
      } finally {
        setHistory((h) => h.filter((m) => m.id !== typingId));
        setSending(false);
      }
    };

    // Prefer streaming SSE for staged UI; fallback to single POST if EventSource unsupported
    if (typeof EventSource !== "undefined") {
      const es = new EventSource(`${apiBase}/chat/stream?message=${encodeURIComponent(text)}`);
      const assistantId = `assistant-${Date.now()}`;

      const closeES = () => {
        try { es.close(); } catch (_) {}
      };

      es.addEventListener("reply", (ev) => {
        try {
          const data = JSON.parse(ev.data || "{}");
          const trimmedReply = (data.reply || "").trimEnd();
          setHistory((h) => {
            const withoutTyping = h.filter((m) => m.id !== typingId);
            return [
              ...withoutTyping,
              { role: "assistant", id: assistantId, content: trimmedReply, ready: false }
            ];
          });
        } catch (_) {}
      });

      es.addEventListener("meta", (ev) => {
        try {
          const data = JSON.parse(ev.data || "{}");
          setHistory((h) => h.map((m) => (
            m.id === assistantId ? { ...m, context: data.context, relevance: data.relevance, ready: true } : m
          )));
        } catch (_) {}
      });

      es.addEventListener("done", () => {
        setSending(false);
        closeES();
      });

      es.addEventListener("error", () => {
        closeES();
        // Seamless fallback to non-streaming POST
        postFallback();
      });
      return;
    }

    // Fallback: single response
    postFallback();
  }

  function toggleExpanded(idx) {
    setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }));
  }

  async function handleClear() {
    // Mark all active ingestion IDs as cancelled
    const activeIds = Array.from(activeIngestStreamsRef.current.keys());
    activeIds.forEach(id => cancelledIngestIdsRef.current.add(id));

    // Close all active ingestion streams and clear their timers
    activeIngestStreamsRef.current.forEach(({ es, timer }) => {
      try {
        if (timer) clearInterval(timer);
        es.close();
      } catch (_) {}
    });

    activeIngestStreamsRef.current.clear();

    try {
      const res = await fetch(`${apiBase}/chat/clear`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clear: true }),
      });
      if (res.ok) {
        setHistory([]);
        cancelledIngestIdsRef.current.clear();
      }
    } catch (_) {}
  }

  return (
    <main id="chat-app" role="main" aria-label="Chat" className="max-w-[800px] mt-4 mx-auto">
      <div className="bg-[#2C3539] border border-[#444] rounded-xl p-3 pb-3 flex flex-col h-[calc(100vh-var(--chat-offset))] shadow-[0_4px_12px_rgba(0,0,0,0.3)]">
        <div id="controls" className="flex gap-2 mb-2 items-center justify-between">
          <div className="flex items-center gap-2 flex-1">
            <span id="model-status-dot" className="status-dot" aria-label="Model status: unknown" title="Model status: unknown"></span>
            <button
              id="btn-paste"
              type="button"
              className="inline-flex items-center justify-center w-9 h-9 rounded-[10px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] transition-colors ease-linear hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600] active:translate-y-px"
              aria-label="Open paste dialog"
              onClick={() => setPasteOpen(true)}
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 5v14M5 12h14" />
              </svg>
            </button>
            <button
              id="btn-saved-ingests"
              type="button"
              className="inline-flex items-center justify-center w-9 h-9 rounded-[10px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] transition-colors ease-linear hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600] active:translate-y-px"
              aria-label="Show saved ingests"
              onClick={() => setSavedOpen(true)}
              title="Show saved ingests"
            >
              {/* Folder icon */}
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
              </svg>
            </button>
          </div>
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

        <div
          id="messages"
          ref={messagesRef}
          className="flex-1 overflow-y-auto p-2 bg-[#1e2a30] rounded-lg mb-4 flex flex-col gap-2"
          role="log"
          aria-live="polite"
          aria-relevant="additions"
          aria-label="Chat messages"
          onScroll={(e) => {
            const el = e.currentTarget;
            const threshold = 24; // px tolerance for being considered at bottom
            const atBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) <= threshold;
            isAtBottomRef.current = atBottom;
          }}
        >
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
            if (item.type === "ingest") {
              const isUser = false;
              const pct = Math.round((item.progress || 0) * 100);
              const loopPct = Math.round(((item.loop || 0) % 1) * 100);
              const meta = item.meta || {};
              return (
                <div key={item.id || idx} className={`flex flex-col self-start items-start`}>
                  <div
                    className={
                      `relative px-4 py-2 my-1 rounded-[20px] whitespace-normal break-words leading-[1.45] font-medium text-[1.05rem] pb-8 ` +
                      "max-w-[90%] min-w-[390px] bg-[#4a555c] text-[#f1f1f1]"
                    }
                  >
                    {item.content}
                    <div className="absolute left-4 right-4 bottom-2 h-3 rounded-full bg-[#2b3940] overflow-hidden border border-[#3a444a]">
                      <div
                        className="h-full bg-[#FF6600] transition-[width] duration-200 ease-linear"
                        style={{ width: `${Math.min(100, pct)}%` }}
                      />
                      {!item.ready && (
                        <div
                          className="absolute top-0 bottom-0 h-full bg-white/20"
                          style={{ left: `${loopPct}%`, width: "20%" }}
                        />
                      )}
                    </div>
                  </div>
                  <div className="mt-1 text-xs opacity-70 flex items-center gap-2">
                    {item.showDetails !== false && (
                      <button
                        type="button"
                        className="inline-flex items-center justify-center w-5 h-5 rounded-full border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
                        aria-label={item.detailsOpen ? "Collapse details" : "Expand details"}
                        aria-expanded={!!item.detailsOpen}
                        onClick={() => setHistory((h) => h.map((m) => (
                          m.id === item.id ? { ...m, detailsOpen: !m.detailsOpen } : m
                        )))}
                      >
                        <svg className={`w-3.5 h-3.5 transition-transform ${!item.detailsOpen ? "rotate-180" : ""}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                          <path d="M19 7l-7 7-7-7" />
                          <path d="M19 13l-7 7-7-7" />
                        </svg>
                      </button>
                    )}
                    {meta.totalSteps != null && (
                      <span>Chunk {Math.min((meta.step || 0) + 1, meta.totalSteps)} / {meta.totalSteps}</span>
                    )}
                    {typeof item.etaMs === "number" && (
                      <span>{" • ETA "}{Math.max(0, Math.floor(item.etaMs/60000))}m {Math.max(0, Math.round((item.etaMs%60000)/1000))}s</span>
                    )}
                  </div>
                  {item.detailsOpen && (
                  <div className="mt-2 text-sm bg-[#1e2a30] text-[#c9d1d9] rounded-lg p-3 w-[95%] max-w-[95%]">
                      <div className="text-xs uppercase tracking-wide opacity-70 mb-1">Checkpoints</div>
                      {(() => {
                        const info = (item.checkpoints || []).find((c) => c.kind === "info");
                        const d = info?.data;
                        if (!d) return null;
                        return (
                          <div className="mb-2 text-xs opacity-80">
                            <div>Tokens≈ {d.approxTokens ?? "?"} • Window {d.windowWords ?? "?"}w • Stride {d.strideWords ?? "?"}w</div>
                            <div>Steps {d.totalSteps ?? "?"} • Checkpoint every ~{d.checkpointTokenInterval ?? "?"} tokens</div>
                          </div>
                        );
                      })()}
                      <ul className="space-y-1">
                        {(item.checkpoints || [])
                          .map((c, i) => ({ c, i }))
                          .filter(({ c }) => c.kind !== "info")
                          .map(({ c, i }) => {
                            const expanded = (item.cpExpanded && item.cpExpanded[i]) || false;
                            const labelPrefix = c.kind === "saved" ? "Saved" : c.kind === "hygiene" ? "Hygiene" : "Summary";
                            return (
                              <li key={`${c.kind || 'cp'}-${i}`} className="text-xs">
                                <button
                                  type="button"
                                  className="inline-flex items-center gap-2 text-left hover:opacity-90"
                                  onClick={() => setHistory((h) => h.map((m) => (
                                    m.id === item.id ? { ...m, cpExpanded: { ...(m.cpExpanded || {}), [i]: !expanded } } : m
                                  )))}
                                >
                                  <span className="inline-flex items-center justify-center w-4 h-4 rounded-[4px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9]">
                                    {expanded ? "−" : "+"}
                                  </span>
                                  <span className="opacity-80">{labelPrefix}:</span>
                                  <span> {c.summary}</span>
                                </button>
                                {expanded && (
                                  <div className="mt-1 ml-6 opacity-80">
                                    {c.kind === "saved" && (
                                      <div>
                                        <div>Type: {c.data?.type}</div>
                                        {Array.isArray(c.data?.entities) && c.data.entities.length > 0 && (
                                          <div>Entities: {c.data.entities.join(", ")}</div>
                                        )}
                                        {typeof c.data?.confidence === "number" && (
                                          <div>Confidence: {c.data.confidence}</div>
                                        )}
                                        {c.data?.explanation && (
                                          <div className="mt-1 italic opacity-80">{c.data.explanation}</div>
                                        )}
                                      </div>
                                    )}
                                    {c.kind === "hygiene" && (
                                      <div>
                                        <div>Merged: {c.data?.merged ?? 0}</div>
                                        <div>Pruned nodes: {c.data?.pruned_nodes ?? 0}</div>
                                        <div>Pruned edges: {c.data?.pruned_edges ?? 0}</div>
                                      </div>
                                    )}
                                  </div>
                                )}
                              </li>
                            );
                          })}
                      </ul>
                      {item.ready && item.stats && (
                        <div className="mt-2 text-xs opacity-80">
                          Words: {item.stats.totalWords} • Lines: {item.stats.totalLines} • Tokens≈ {item.stats.approxTokens ?? "?"} • Time: {Math.round((item.stats.timeMs || 0)/1000)}s
                        </div>
                      )}
                  </div>
                  )}
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
                    `relative px-4 py-2 my-1 rounded-[20px] whitespace-normal break-words leading-[1.45] font-medium text-[1.05rem] ${
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
                      id={`details-button-${idx}`}
                      className="absolute left-2 -bottom-3 inline-flex items-center justify-center w-7 h-7 rounded-full border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600] active:translate-y-px"
                      aria-label="Expand details"
                      type="button"
                      aria-expanded={!!expanded[idx]}
                      disabled={!item.ready}
                      aria-controls={`details-${idx}`}
                      onClick={() => item.ready && toggleExpanded(idx)}
                    >
                      <svg className={`w-4 h-4 transition-transform ${!expanded[idx] ? "rotate-180" : ""}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M19 7l-7 7-7-7" />
                        <path d="M19 13l-7 7-7-7" />
                      </svg>
                    </button>
                  )}
                </div>
                {!isUser && expanded[idx] && (
                  <div
                    className="mt-2 text-sm bg-[#1e2a30] text-[#c9d1d9] rounded-lg p-3 w-[95%] max-w-[95%]"
                    id={`details-${idx}`}
                    role="region"
                    aria-labelledby={`details-button-${idx}`}
                  >
                    {item.relevance && item.relevance.saved && (
                      <div className="mb-2">
                        <div className="text-xs uppercase tracking-wide opacity-70 mb-1">Saved this turn</div>
                        <ul className="list-disc list-inside space-y-1">
                          <li className="text-xs">
                            <span className="opacity-80">[{item.relevance.saved.type}]</span> {item.relevance.saved.summary}
                            {item.relevance.saved.entities && item.relevance.saved.entities.length > 0 && (
                              <span className="opacity-60"> (entities: {item.relevance.saved.entities.join(", ")})</span>
                            )}
                            {typeof item.relevance.saved.confidence === "number" && (
                              <span className="opacity-60"> — conf {item.relevance.saved.confidence.toFixed ? item.relevance.saved.confidence.toFixed(2) : item.relevance.saved.confidence}</span>
                            )}
                          </li>
                        </ul>
                        <div className="h-px bg-[#344046] my-2" />
                      </div>
                    )}
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
                                  {m.explanation && (
                                    <div className="opacity-70 mt-0.5 italic">{m.explanation}</div>
                                  )}
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
          <label htmlFor="message-input" className="sr-only">Message</label>
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
      {pasteOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="paste-title"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
        >
          <div className="bg-[#2C3539] border border-[#444] rounded-xl w-[min(92vw,760px)] h-[min(80vh,640px)] shadow-[0_8px_24px_rgba(0,0,0,0.45)] flex flex-col">
            <div className="px-4 py-3 border-b border-[#3a454b] flex items-center justify-between">
              <h2 id="paste-title" className="text-[#f0f0f0] text-base font-semibold">Paste text</h2>
              <div className="flex items-center gap-2">
                <div className="flex flex-col items-start mr-2">
                  <div className="flex items-center gap-2">
                    <label className="text-xs opacity-80" htmlFor="stride-words-input">Chunk Size</label>
                    <input
                      id="stride-words-input"
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      value={strideWords}
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === "") {
                          setStrideWords("");
                          setStrideTouched(true);
                          return;
                        }
                        const v = Math.max(1, Math.min(5000, parseInt(raw || "1", 10) || 1));
                        setStrideWords(v);
                        setStrideTouched(true);
                      }}
                      className="w-20 py-1 text-sm rounded-md border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] focus:outline-none focus:border-[#FF6600] focus:ring-2 focus:ring-[rgba(255,102,0,0.3)]"
                    />
                    {(() => {
                      const isEmpty = String(strideWords ?? "").trim() === "";
                      if (isEmpty) return null;
                      const text = pasteText || "";
                      const words = text.trim().length === 0 ? 0 : text.trim().split(/\s+/).length;
                      const tokensPerWord = words === 0 ? 1.3 : Math.max(0.5, Math.min(2.0, (text.length / 4) / Math.max(1, words)));
                      const windowTokens = 200;
                      const windowWords = Math.max(1, Math.floor(windowTokens / Math.max(0.0001, tokensPerWord)));
                      const sWords = Math.max(1, Math.min(5000, (parseInt(String(strideWords), 10) || 1)));
                      const totalSteps = Math.max(1, Math.ceil(Math.max(0, words - windowWords) / sWords) + 1);
                      const secondsPerStep = 17.3;
                      const totalSeconds = Math.max(0, Math.round(totalSteps * secondsPerStep));
                      const mm = Math.floor(totalSeconds / 60);
                      const ss = Math.max(0, totalSeconds % 60);
                      return (
                        <div className="text-xs opacity-70">{`≈ ${totalSteps} chunks • ETA ${mm}m ${ss}s`}</div>
                      );
                    })()}
                  </div>
                  <div className="text-[10px] opacity-60">roughly estimated for a 4090</div>
                </div>
                <button
                  type="button"
                  className="px-2 py-1 text-sm rounded-md border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
                  onClick={async () => {
                    const localText = pasteText;
                    setPasteOpen(false);
                    const id = `ingest-${Date.now()}`;
                    setHistory((h) => [
                      ...h,
                      { role: "assistant", id, type: "ingest", content: "Processing pasted text…", ready: false, progress: 0, loop: 0, stats: null, checkpoints: [], etaMs: null, detailsOpen: false },
                    ]);

                    // Upload text to backend
                    let upload;
                    try {
                      const res = await fetch(`${apiBase}/ingest/upload`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ text: localText }),
                      });
                      upload = await res.json();
                    } catch (e) {
                      setHistory((h) => h.map((m) => (m.id === id ? { ...m, content: `[Error] Upload failed`, ready: true } : m)));
                      return;
                    }

                    const totalWords = upload.totalWords || 0;
                    const ingestStart = Date.now();
                    let t0Second = null; // timestamp when second chunk starts
                    let wordsAtSecondStart = 0;

                    // Loop progress animation
                    let loopVal = 0;
                    const loopTimer = setInterval(() => {
                      loopVal = (loopVal + 0.08) % 1;
                      setHistory((h) => h.map((m) => (m.id === id ? { ...m, loop: loopVal } : m)));
                    }, 150);

                    // Open SSE stream
                    const strideParam = String(Math.max(1, Math.min(5000, Number(strideWords) || 1)));
                    const es = new EventSource(`${apiBase}/ingest/stream?id=${encodeURIComponent(upload.id)}&strideWords=${encodeURIComponent(strideParam)}`);
                    activeIngestStreamsRef.current.set(id, { es, timer: loopTimer });
                    es.addEventListener("info", (ev) => {
                      try {
                        const info = JSON.parse(ev.data || "{}");
                        const summary = `~${info.approxTokens ?? "?"} tokens • window ${info.windowWords ?? "?"}w stride ${info.strideWords ?? "?"}w • steps ${info.totalSteps ?? "?"}`;
                        setHistory((h) => h.map((m) => (
                          m.id === id
                            ? {
                                ...m,
                                checkpoints: [...(m.checkpoints || []), { kind: "info", summary, data: info }],
                                stats: { ...(m.stats || {}), approxTokens: info.approxTokens },
                                meta: { step: 0, totalSteps: info.totalSteps, consumedWords: 0, consumedLines: 0 },
                              }
                            : m
                        )));
                      } catch (_) {}
                    });

                    const closeES = () => {
                      const entry = activeIngestStreamsRef.current.get(id);
                      if (entry) {
                        if (entry.timer) clearInterval(entry.timer);
                        try { entry.es.close(); } catch (_) {}
                        activeIngestStreamsRef.current.delete(id);
                      }
                    };

                    es.addEventListener("progress", (ev) => {
                      try {
                        const data = JSON.parse(ev.data || "{}");
                        const step = Number(data.step || 0);
                        const totalSteps = Number(data.totalSteps || 1);
                        const consumedWords = Number(data.consumedWords || 0);
                        const consumedLines = Number(data.consumedLines || 0);
                        const progress = Number(data.progress || 0);

                        // ETA: start timing after first chunk completes; use avg speed since second chunk start
                        if (step === 1 && t0Second == null) {
                          t0Second = Date.now();
                          wordsAtSecondStart = consumedWords;
                        }
                        let etaMs = null;
                        if (t0Second != null && consumedWords > wordsAtSecondStart) {
                          const elapsed = Date.now() - t0Second;
                          const processed = consumedWords - wordsAtSecondStart;
                          const remaining = Math.max(0, totalWords - consumedWords);
                          const speed = processed / Math.max(1, elapsed); // words per ms
                          etaMs = speed > 0 ? Math.round(remaining / speed) : null;
                        }

                        setHistory((h) => h.map((m) => (
                          m.id === id ? { ...m, progress, etaMs, meta: { step, totalSteps, consumedWords, consumedLines } } : m
                        )));
                      } catch (_) {}
                    });

                    es.addEventListener("saved", (ev) => {
                      try {
                        const s = JSON.parse(ev.data || "{}");
                        setHistory((h) => h.map((m) => (
                          m.id === id ? { ...m, checkpoints: [...(m.checkpoints || []), { kind: "saved", step: (m.meta?.step ?? 0), summary: s.summary, data: s }] } : m
                        )));
                      } catch (_) {}
                    });

                    es.addEventListener("checkpoint", (ev) => {
                      try {
                        const cp = JSON.parse(ev.data || "{}");
                        setHistory((h) => h.map((m) => (
                          m.id === id ? { ...m, checkpoints: [...(m.checkpoints || []), { kind: "checkpoint", ...cp }] } : m
                        )));
                      } catch (_) {}
                    });

                    es.addEventListener("hygiene", (ev) => {
                      try {
                        const hg = JSON.parse(ev.data || "{}");
                        const text = `merged ${hg.merged ?? 0}, pruned nodes ${hg.pruned_nodes ?? 0}, pruned edges ${hg.pruned_edges ?? 0}`;
                        setHistory((h) => h.map((m) => (
                          m.id === id ? { ...m, checkpoints: [...(m.checkpoints || []), { kind: "hygiene", step: (m.meta?.step ?? 0), summary: text, data: hg }] } : m
                        )));
                      } catch (_) {}
                    });

                    es.addEventListener("done", (ev) => {
                      clearInterval(loopTimer);
                      let stats = {};
                      try { stats = JSON.parse(ev.data || "{}"); } catch (_) {}
                      const elapsed = Math.max(0, Date.now() - ingestStart);
                      setHistory((h) => h.map((m) => (
                        m.id === id
                          ? {
                              ...m,
                              ready: true,
                              progress: 1,
                              stats: {
                                ...(m.stats || {}),
                                totalWords,
                                totalLines: upload.totalLines,
                                timeMs: elapsed,
                                steps: stats.steps,
                              },
                              content: `Text input of ${totalWords} words processed in ${Math.round(elapsed/1000)}s.`,
                            }
                          : m
                      )));
                      closeES();
                    });

                    es.addEventListener("error", () => {
                      clearInterval(loopTimer);
                      const wasCancelled = cancelledIngestIdsRef.current.has(id);
                      setHistory((h) => h.map((m) => (m.id === id ? {
                        ...m,
                        content: wasCancelled ? `[Cancelled] Ingestion stopped` : `[Error] Ingest stream failed`,
                        ready: true
                      } : m)));
                      cancelledIngestIdsRef.current.delete(id);
                      closeES();
                    });
                  }}
                >Process</button>
                <button
                  type="button"
                  className="px-2 py-1 text-sm rounded-md border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
                  onClick={() => setPasteText("")}
                >Clear</button>
                <button
                  type="button"
                  className="px-2 py-1 text-sm rounded-md border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
                  onClick={() => setPasteOpen(false)}
                >Close</button>
              </div>
            </div>
            <div className="p-3 flex-1 overflow-y-auto">
              <label htmlFor="paste-area" className="sr-only">Text to paste</label>
              <textarea
                id="paste-area"
                value={pasteText}
                onChange={(e) => setPasteText(e.target.value)}
                className="w-full h-full min-h-[320px] resize-none rounded-lg border border-[#4a555c] bg-[#1e2a30] text-[#f0f0f0] p-3 font-mono text-sm leading-relaxed focus:outline-none focus:border-[#FF6600] focus:ring-2 focus:ring-[rgba(255,102,0,0.3)]"
                placeholder="Paste or type your text here…"
              />
            </div>
            <div className="px-4 py-2 border-t border-[#3a454b] text-xs text-[#c9d1d9] flex items-center justify-between">
              <div>
                {(() => {
                  const text = pasteText || "";
                  const lines = text.length === 0 ? 0 : (text.match(/\n/g)?.length ?? 0) + 1;
                  const words = text.trim().length === 0 ? 0 : text.trim().split(/\s+/).length;
                  return `Words: ${words} • Lines: ${lines}`;
                })()}
              </div>
            </div>
          </div>
        </div>
      )}
      {savedOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="saved-title"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
        >
          <div className="bg-[#2C3539] border border-[#444] rounded-xl w-[min(92vw,560px)] h-[min(70vh,520px)] shadow-[0_8px_24px_rgba(0,0,0,0.45)] flex flex-col">
            <div className="px-4 py-3 border-b border-[#3a454b] flex items-center justify-between">
              <h2 id="saved-title" className="text-[#f0f0f0] text-base font-semibold">Saved ingests</h2>
              <button
                type="button"
                className="px-2 py-1 text-sm rounded-md border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
                onClick={() => setSavedOpen(false)}
              >Close</button>
            </div>
            <div className="p-3 flex-1 overflow-y-auto">
              {savedIngests.length === 0 ? (
                <div className="text-sm opacity-70">No saved ingests found.</div>
              ) : (
                <ul className="divide-y divide-[#3a454b]">
                  {savedIngests.map((it) => (
                    <li key={it.id} className="py-2 flex items-center justify-between">
                      <div>
                        <div className="text-[#f0f0f0] text-sm font-medium">{it.name || it.id}</div>
                        <div className="text-xs opacity-60">
                          {(() => {
                            const humanBytes = (n) => {
                              if (typeof n !== "number" || !isFinite(n) || n < 0) return null;
                              if (n < 1024) return `${n} B`;
                              if (n < 1024 * 1024) return `${(n/1024).toFixed(1)} KB`;
                              return `${(n/1024/1024).toFixed(1)} MB`;
                            };
                            const b = humanBytes(it.bytes);
                            const parts = [
                              (typeof it.locations === "number") ? `${it.locations} locations` : null,
                              (typeof it.memories === "number") ? `${it.memories} memories` : null,
                              b ? `(${b})` : null,
                            ].filter(Boolean);
                            return parts.join(" • ");
                          })()}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {/* Rename */}
                        <button
                          type="button"
                          className="inline-flex items-center justify-center w-7 h-7 rounded-[8px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
                          title="Rename"
                          onClick={async () => {
                            const current = it.name || it.id;
                            const next = window.prompt("Rename ingest", current);
                            if (!next) return;
                            try {
                              const res = await fetch(`${apiBase}/ingest/shard/${encodeURIComponent(it.id)}/name`, {
                                method: "PUT",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ name: next }),
                              });
                              if (res.ok) {
                                const data = await res.json();
                                setSavedIngests((arr) => arr.map((x) => (x.id === it.id ? { ...x, name: data.name } : x)));
                              }
                            } catch (_) {}
                          }}
                        >
                          {/* Pencil icon */}
                          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <path d="M12 20h9" />
                            <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
                          </svg>
                        </button>
                        {/* Load */}
                        <button
                          type="button"
                          className="inline-flex items-center justify-center w-7 h-7 rounded-[8px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#2ecc71] hover:text-[#2ecc71]"
                          title="Load into world"
                          onClick={async () => {
                            try {
                              const res = await fetch(`${apiBase}/ingest/shard/${encodeURIComponent(it.id)}/load`, { method: "POST" });
                              if (res.ok) {
                                const info = await res.json();
                                // Reload list to reflect any changes
                                try {
                                  const data = await (await fetch(`${apiBase}/ingest/list`)).json();
                                  setSavedIngests(Array.isArray(data.ingests) ? data.ingests : []);
                                } catch (_) {}
                                // Post chat bubble summarizing the load
                                const name = info.name || it.name || it.id;
                                const locs = typeof info.locations === "number" ? info.locations : undefined;
                                const mems = typeof info.memories === "number" ? info.memories : undefined;
                                const bytes = typeof info.bytes === "number" ? info.bytes : undefined;
                                const humanBytes = (n) => {
                                  if (typeof n !== "number" || !isFinite(n) || n < 0) return null;
                                  if (n < 1024) return `${n} B`;
                                  if (n < 1024 * 1024) return `${(n/1024).toFixed(1)} KB`;
                                  return `${(n/1024/1024).toFixed(1)} MB`;
                                };
                                const sizeStr = humanBytes(bytes);
                                const parts = [
                                  `Loaded: ${name}`,
                                  locs != null ? `${locs} locations` : null,
                                  mems != null ? `${mems} memories` : null,
                                  sizeStr ? `(${sizeStr})` : null,
                                ].filter(Boolean);
                                const msgId = `loaded-${it.id}-${Date.now()}`;
                                setHistory((h) => [
                                  ...h,
                                  {
                                    role: "assistant",
                                    id: msgId,
                                    type: "ingest",
                                    content: parts.join(" — "),
                                    ready: true,
                                    progress: 1,
                                    loop: 0,
                                    stats: { steps: 1 },
                                    checkpoints: [
                                      { kind: "checkpoint", step: 1, summary: `Shard loaded: ${name}`, data: { locations: locs, memories: mems, bytes } },
                                      { kind: "info", summary: `locations ${locs ?? "?"} • memories ${mems ?? "?"}`, data: { approxTokens: undefined, windowWords: undefined, strideWords: undefined, totalSteps: 1, checkpointTokenInterval: undefined } },
                                    ],
                                    detailsOpen: false,
                                    showDetails: false,
                                  },
                                ]);
                              }
                            } catch (_) {}
                          }}
                        >
                          {/* Link/attach icon */}
                          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <path d="M10 13a5 5 0 0 0 7.07 0l3.54-3.54a5 5 0 0 0-7.07-7.07L11 4" />
                            <path d="M14 11a5 5 0 0 0-7.07 0L3.39 14.54a5 5 0 0 0 7.07 7.07L13 20" />
                          </svg>
                        </button>
                        {/* Delete with 1.5s hold */}
                        <HoldToDeleteButton id={it.id} onConfirm={async () => {
                          try {
                            const res = await fetch(`${apiBase}/ingest/shard/${encodeURIComponent(it.id)}`, { method: "DELETE" });
                            if (res.ok) setSavedIngests((arr) => arr.filter((x) => x.id !== it.id));
                          } catch (_) {}
                        }} />
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default App;

function HoldToDeleteButton({ id, onConfirm }) {
  const HOLD_MS = 1500;
  const [progress, setProgress] = useState(0);
  const rafRef = useRef(null);
  const startRef = useRef(null);
  const activeRef = useRef(false);

  const stop = () => {
    activeRef.current = false;
    startRef.current = null;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    setProgress(0);
  };

  const step = (ts) => {
    if (!activeRef.current) return;
    if (!startRef.current) startRef.current = ts;
    const elapsed = ts - startRef.current;
    const p = Math.min(1, elapsed / HOLD_MS);
    setProgress(p);
    if (p >= 1) {
      stop();
      if (typeof onConfirm === "function") onConfirm();
      return;
    }
    rafRef.current = requestAnimationFrame(step);
  };

  const start = () => {
    if (activeRef.current) return;
    activeRef.current = true;
    setProgress(0);
    rafRef.current = requestAnimationFrame(step);
  };

  return (
    <button
      type="button"
      className="relative inline-flex items-center justify-center w-7 h-7 rounded-[8px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#3a2a2a] hover:border-[#c0392b] hover:text-[#c0392b]"
      title="Hold to delete"
      onMouseDown={start}
      onMouseUp={stop}
      onMouseLeave={stop}
      onTouchStart={(e) => { e.preventDefault(); start(); }}
      onTouchEnd={(e) => { e.preventDefault(); stop(); }}
    >
      {/* Trash icon */}
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M3 6h18" />
        <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      </svg>
      {/* Progress ring overlay for border animation */}
      <svg className="absolute inset-0" viewBox="0 0 36 36" aria-hidden="true">
        <circle cx="18" cy="18" r="16" fill="none" stroke="#3a454b" strokeWidth="2" />
        <circle
          cx="18" cy="18" r="16" fill="none"
          stroke="#FF6600"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={`${Math.max(1, 2 * Math.PI * 16)} ${Math.max(1, 2 * Math.PI * 16)}`}
          strokeDashoffset={`${(1 - progress) * 2 * Math.PI * 16}`}
          style={{ transition: activeRef.current ? "none" : "stroke-dashoffset 150ms linear" }}
        />
      </svg>
    </button>
  );
}
