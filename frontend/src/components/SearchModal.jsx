import { useEffect, useMemo, useState } from "react";

export default function SearchModal({ apiBase, open, onClose }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [k, setK] = useState(10);
  const [sinceLocal, setSinceLocal] = useState(""); // datetime-local string
  const [typesState, setTypesState] = useState({ npc: true, location: true, world_state: true, lore: true });
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(null);
  const [listExpanded, setListExpanded] = useState({}); // id -> bool for left list item expansion

  useEffect(() => {
    if (!open) return;
    setError("");
  }, [open]);

  if (!open) return null;

  const typesCsv = useMemo(() => {
    const entries = Object.entries(typesState).filter(([, v]) => !!v).map(([k]) => k);
    return entries.join(",");
  }, [typesState]);

  const isoFromLocal = (v) => {
    if (!v || String(v).trim() === "") return null;
    try {
      const d = new Date(v);
      if (isNaN(d.getTime())) return null;
      return d.toISOString();
    } catch (_) { return null; }
  };

  const doSearch = async () => {
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    setError("");
    setSelected(null);
    try {
      const params = new URLSearchParams();
      params.set("q", q);
      if (mode) params.set("mode", mode);
      if (k != null && String(k).trim() !== "") params.set("k", String(Math.max(1, Math.min(100, parseInt(String(k) || "10", 10) || 10))));
      if (typesCsv) params.set("types", typesCsv);
      const sinceIso = isoFromLocal(sinceLocal);
      if (sinceIso) params.set("since", sinceIso);

      const res = await fetch(`${apiBase}/search?${params.toString()}`);
      if (!res.ok) throw new Error(res.statusText || "search failed");
      const data = await res.json();
      setResults(Array.isArray(data.results) ? data.results : []);
    } catch (e) {
      setError(String(e?.message || e || "Search failed"));
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="search-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
    >
      <div className="bg-[#2C3539] border border-[#444] rounded-xl w-[min(92vw,860px)] h-[min(80vh,640px)] shadow-[0_8px_24px_rgba(0,0,0,0.45)] flex flex-col">
        <div className="px-4 py-3 border-b border-[#3a454b] flex items-center justify-between">
          <h2 id="search-title" className="text-[#f0f0f0] text-base font-semibold">Search memories</h2>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="px-2 py-1 text-sm rounded-md border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
              onClick={onClose}
            >Close</button>
          </div>
        </div>
        <div className="p-3 flex flex-col gap-3">
          <div className="flex flex-wrap items-end gap-2">
            <div className="flex-1 min-w-[220px]">
              <label htmlFor="search-q" className="text-xs opacity-80">Query</label>
              <input
                id="search-q"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") doSearch(); }}
                placeholder="Type a word…"
                className="w-full px-3 py-2 border border-[#4a555c] bg-[#1e2a30] text-[#f0f0f0] rounded-md text-sm focus:outline-none focus:border-[#FF6600] focus:ring-2 focus:ring-[rgba(255,102,0,0.3)]"
              />
            </div>
            <div>
              <label htmlFor="search-mode" className="text-xs opacity-80">Mode</label>
              <select
                id="search-mode"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-[140px] px-3 py-2 border border-[#4a555c] bg-[#1e2a30] text-[#f0f0f0] rounded-md text-sm focus:outline-none focus:border-[#FF6600] focus:ring-2 focus:ring-[rgba(255,102,0,0.3)]"
              >
                <option value="hybrid">hybrid</option>
                <option value="literal">literal</option>
                <option value="semantic">semantic</option>
              </select>
            </div>
            <div>
              <label htmlFor="search-k" className="text-xs opacity-80">Max results</label>
              <input
                id="search-k"
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                value={k}
                onChange={(e) => setK(e.target.value)}
                className="w-[90px] px-3 py-2 border border-[#4a555c] bg-[#1e2a30] text-[#f0f0f0] rounded-md text-sm focus:outline-none focus:border-[#FF6600] focus:ring-2 focus:ring-[rgba(255,102,0,0.3)]"
              />
            </div>
            <div>
              <label htmlFor="search-since" className="text-xs opacity-80">Since</label>
              <input
                id="search-since"
                type="datetime-local"
                value={sinceLocal}
                onChange={(e) => setSinceLocal(e.target.value)}
                className="px-3 py-2 border border-[#4a555c] bg-[#1e2a30] text-[#f0f0f0] rounded-md text-sm focus:outline-none focus:border-[#FF6600] focus:ring-2 focus:ring-[rgba(255,102,0,0.3)]"
              />
            </div>
            <div className="ml-auto">
              <button
                type="button"
                disabled={searching || query.trim() === ""}
                onClick={doSearch}
                className="px-3 py-2 rounded-md border border-[#4a555c] bg-[#FF6600] text-[#111111] text-sm font-semibold hover:bg-[#FF8533] disabled:bg-[#536267] disabled:text-[#aaa]"
              >{searching ? "Searching…" : "Search"}</button>
            </div>
          </div>
          <div className="flex items-center gap-4 flex-wrap">
            <div className="text-xs opacity-80">Types:</div>
            {[
              { key: "npc", label: "npc" },
              { key: "location", label: "location" },
              { key: "world_state", label: "world_state" },
              { key: "lore", label: "lore" },
            ].map((t) => (
              <label key={t.key} className="inline-flex items-center gap-2 text-sm opacity-90">
                <input
                  type="checkbox"
                  checked={!!typesState[t.key]}
                  onChange={(e) => setTypesState((s) => ({ ...s, [t.key]: e.target.checked }))}
                />
                <span>{t.label}</span>
              </label>
            ))}
          </div>
          {error && (
            <div className="text-sm text-[#ff6b6b]">{error}</div>
          )}
        </div>
        <div className="px-3 pb-3 grid grid-cols-1 md:grid-cols-2 gap-3 flex-1 overflow-hidden">
          <div className="flex flex-col overflow-hidden">
            <div className="text-xs uppercase tracking-wide opacity-70 mb-1">Results</div>
            <div className="flex-1 overflow-y-auto rounded-lg border border-[#3a454b]">
              {results.length === 0 ? (
                <div className="p-3 text-sm opacity-70">No results.</div>
              ) : (
                <ul className="divide-y divide-[#3a454b]">
                  {results.map((r, idx) => {
                    const key = r.item_id || String(idx);
                    const isExpanded = !!listExpanded[key];
                    const clampStyle = isExpanded
                      ? {}
                      : { display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" };
                    const typeStr = typeof r.type === "string" ? r.type : "";
                    const rawText = typeof r.text === "string" ? r.text : "";
                    const prefix = typeStr ? `[${typeStr}]` : "";
                    const displayText = prefix && rawText.startsWith(prefix)
                      ? rawText.slice(prefix.length).trimStart()
                      : rawText;
                    return (
                      <li key={key} className="p-2 hover:bg-[#26343b] cursor-pointer" onClick={() => setSelected(r)}>
                        <div className="flex items-start gap-2">
                          <button
                            type="button"
                            className="shrink-0 inline-flex items-center justify-center w-6 h-6 rounded-[6px] border border-[#4a555c] bg-[#2C3539] text-[#c9d1d9] hover:bg-[#3a454b]"
                            onClick={(e) => {
                              e.stopPropagation();
                              setListExpanded((s) => ({ ...s, [key]: !isExpanded }));
                            }}
                            aria-expanded={isExpanded}
                            aria-label={isExpanded ? "Collapse" : "Expand"}
                            title={isExpanded ? "Collapse" : "Expand"}
                          >
                            {isExpanded ? "−" : "+"}
                          </button>
                          <div className="flex-1 min-w-0">
                            <div className="text-sm text-[#f0f0f0]">
                              {typeStr && <span className="opacity-80">[{typeStr}] </span>}
                              <span style={clampStyle}>{displayText}</span>
                            </div>
                            <div className="flex items-center gap-2 text-[11px] opacity-60 mt-0.5">
                              <span>{r.updated_at}</span>
                              <span className="opacity-40">•</span>
                              <span>Score {r.score != null ? (r.score.toFixed ? r.score.toFixed(3) : r.score) : "-"}</span>
                            </div>
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </div>
          <div className="flex flex-col overflow-hidden">
            <div className="text-xs uppercase tracking-wide opacity-70 mb-1">Selected</div>
            <div className="flex-1 overflow-y-auto rounded-lg border border-[#3a454b] p-3">
              {!selected ? (
                <div className="text-sm opacity-70">Click a result to view its full data.</div>
              ) : (
                <pre className="whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-[#c9d1d9]">{JSON.stringify(selected, null, 2)}</pre>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
