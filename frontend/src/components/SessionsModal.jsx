import { useEffect, useState } from "react";

export default function SessionsModal({ apiBase, open, onClose, onLoaded }) {
  const [sessions, setSessions] = useState([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${apiBase}/api/sessions`);
        if (!res.ok) throw new Error("list not ok");
        const data = await res.json();
        if (!cancelled) setSessions(Array.isArray(data.sessions) ? data.sessions : []);
      } catch (_) {
        if (!cancelled) setSessions([]);
      }
    })();
    return () => { cancelled = true; };
  }, [open, apiBase]);

  if (!open) return null;

  const humanBytes = (n) => {
    if (typeof n !== "number" || !isFinite(n) || n < 0) return null;
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n/1024).toFixed(1)} KB`;
    return `${(n/1024/1024).toFixed(1)} MB`;
  };

  const isoToRel = (iso) => {
    if (!iso) return null;
    try {
      const t = new Date(iso);
      const diff = Date.now() - t.getTime();
      const mins = Math.round(diff / 60000);
      if (mins < 1) return "just now";
      if (mins < 60) return `${mins}m ago`;
      const hrs = Math.round(mins/60);
      if (hrs < 24) return `${hrs}h ago`;
      const days = Math.round(hrs/24);
      return `${days}d ago`;
    } catch(_) { return null; }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="sessions-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
    >
      <div className="bg-[#2C3539] border border-[#444] rounded-xl w-[min(92vw,640px)] h-[min(75vh,560px)] shadow-[0_8px_24px_rgba(0,0,0,0.45)] flex flex-col">
        <div className="px-4 py-3 border-b border-[#3a454b] flex items-center justify-between">
          <h2 id="sessions-title" className="text-[#f0f0f0] text-base font-semibold">Sessions</h2>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="px-2 py-1 text-sm rounded-md border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
              onClick={async () => {
                const name = window.prompt("Save current session as", "Session");
                if (!name) return;
                try {
                  setSaving(true);
                  const res = await fetch(`${apiBase}/api/sessions`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name }),
                  });
                  if (res.ok) {
                    const data = await res.json();
                    try {
                      const list = await (await fetch(`${apiBase}/api/sessions`)).json();
                      setSessions(Array.isArray(list.sessions) ? list.sessions : []);
                    } catch(_) {}
                  }
                } catch(_) {
                } finally {
                  setSaving(false);
                }
              }}
              disabled={saving}
            >{saving ? "Saving…" : "Save Current As…"}</button>
            <button
              type="button"
              className="px-2 py-1 text-sm rounded-md border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#FF6600] hover:text-[#FF6600]"
              onClick={onClose}
            >Close</button>
          </div>
        </div>
        <div className="p-3 flex-1 overflow-y-auto">
          {sessions.length === 0 ? (
            <div className="text-sm opacity-70">No sessions found.</div>
          ) : (
            <ul className="divide-y divide-[#3a454b]">
              {sessions.map((it) => (
                <li key={it.id} className="py-2 flex items-center justify-between">
                  <div>
                    <div className="text-[#f0f0f0] text-sm font-medium">{it.name || it.id}</div>
                    <div className="text-xs opacity-60">
                      {(() => {
                        const b = humanBytes(it.bytes);
                        const rel = isoToRel(it.updatedAt || it.createdAt);
                        const parts = [rel, b ? `(${b})` : null].filter(Boolean);
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
                        const next = window.prompt("Rename session", current);
                        if (!next) return;
                        try {
                          const res = await fetch(`${apiBase}/api/sessions/${encodeURIComponent(it.id)}`, {
                            method: "PUT",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ name: next }),
                          });
                          if (res.ok) {
                            const data = await res.json();
                            setSessions((arr) => arr.map((x) => (x.id === it.id ? { ...x, name: data.name } : x)));
                          }
                        } catch(_) {}
                      }}
                    >
                      {/* Pencil icon */}
                      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M12 20h9" />
                        <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
                      </svg>
                    </button>
                    {/* Overwrite */}
                    <button
                      type="button"
                      className="inline-flex items-center justify-center w-7 h-7 rounded-[8px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#f39c12] hover:text-[#f39c12]"
                      title="Overwrite with current state"
                      onClick={async () => {
                        if (!window.confirm(`Overwrite session "${it.name || it.id}" with current state?`)) return;
                        try {
                          const res = await fetch(`${apiBase}/api/sessions/${encodeURIComponent(it.id)}`, {
                            method: "PUT",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ overwrite: true }),
                          });
                          if (res.ok) {
                            const list = await (await fetch(`${apiBase}/api/sessions`)).json();
                            setSessions(Array.isArray(list.sessions) ? list.sessions : []);
                          }
                        } catch(_) {}
                      }}
                    >
                      {/* Save icon */}
                      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                        <path d="M17 21v-8H7v8" />
                        <path d="M7 3v5h8" />
                      </svg>
                    </button>
                    {/* Load */}
                    <button
                      type="button"
                      className="inline-flex items-center justify-center w-7 h-7 rounded-[8px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#26343b] hover:border-[#2ecc71] hover:text-[#2ecc71]"
                      title="Load session"
                      onClick={async () => {
                        try {
                          const res = await fetch(`${apiBase}/api/sessions/${encodeURIComponent(it.id)}/load?mode=merge`, { method: "POST" });
                          if (res.ok) {
                            const info = await res.json();
                            if (typeof onLoaded === "function") onLoaded({ id: it.id, name: it.name || it.id, info });
                            if (typeof onClose === "function") onClose();
                          }
                        } catch(_) {}
                      }}
                    >
                      {/* Play icon */}
                      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                    </button>
                    {/* Delete (confirm) */}
                    <button
                      type="button"
                      className="inline-flex items-center justify-center w-7 h-7 rounded-[8px] border border-[#4a555c] bg-[#1e2a30] text-[#c9d1d9] hover:bg-[#3a2a2a] hover:border-[#c0392b] hover:text-[#c0392b]"
                      title="Delete session"
                      onClick={async () => {
                        if (!window.confirm(`Delete session "${it.name || it.id}"?`)) return;
                        try {
                          const res = await fetch(`${apiBase}/api/sessions/${encodeURIComponent(it.id)}`, { method: "DELETE" });
                          if (res.ok) setSessions((arr) => arr.filter((x) => x.id !== it.id));
                        } catch(_) {}
                      }}
                    >
                      {/* Trash icon */}
                      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M3 6h18" />
                        <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                      </svg>
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
