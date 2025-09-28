// frontend/src/pages/Dashboard.jsx
import React, { useEffect, useState, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
let ShowdownConverter = null;
try {
  // Attempt to import showdown if available in node_modules (recommended)
  // If not present, we will fall back to plain-text rendering.
  // Note: bundlers may tree-shake this import; try/catch prevents build-time crash if not installed.
  // If you prefer to use CDN, include showdown in public/index.html and use window.showdown.
  // npm: npm install showdown
  // eslint-disable-next-line import/no-extraneous-dependencies
  // Showdown supports both default and named exports; safe to access like below:
  // (if bundler fails, just rely on fallback)
  // eslint-disable-next-line
  const sd = require("showdown");
  ShowdownConverter = sd.Converter;
} catch (e) {
  // no-op: we'll fallback to simple display
}

export default function Dashboard() {
  const [dashboard, setDashboard] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  // IPO states
  const [ipoAnalyzerOpen, setIpoAnalyzerOpen] = useState(false);
  const [ipoResultHtml, setIpoResultHtml] = useState("");
  const [ipoReports, setIpoReports] = useState([]);
  const [ipoModalOpen, setIpoModalOpen] = useState(false);
  const [ipoModalContent, setIpoModalContent] = useState("");
  const [ipoModalMeta, setIpoModalMeta] = useState("");
  const [ipoLoading, setIpoLoading] = useState(false);
  const [selectedIpoIds, setSelectedIpoIds] = useState(new Set());
  const [analyzing, setAnalyzing] = useState(false);

  const fileRef = useRef(null);
  const allocChartRef = useRef(null);
  const lineChartRef = useRef(null);
  const allocChartInstance = useRef(null);
  const lineChartInstance = useRef(null);
  const ipoFileRef = useRef(null);
  const ipoTextRef = useRef(null);

  const navigate = useNavigate();

  useEffect(() => {
    loadDashboard();
    loadPortfolioHistory(6);
    loadIpoReports();
    const iv = setInterval(loadDashboard, 60 * 1000);
    return () => {
      clearInterval(iv);
      // destroy charts on unmount
      try {
        if (allocChartInstance.current) allocChartInstance.current.destroy();
        if (lineChartInstance.current) lineChartInstance.current.destroy();
      } catch (e) {
        /* ignore */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadDashboard() {
    try {
      const res = await fetch("/portfolio", { credentials: "same-origin" });
      if (res.status === 401) {
        // not logged in — redirect to login
        window.location.href = "/login";
        return;
      }
      const j = await res.json();
      setDashboard(j);
      setLoading(false);
      renderAllocChart(j?.allocations || {});
    } catch (e) {
      console.error("loadDashboard error", e);
      setLoading(false);
    }
  }

  async function loadPortfolioHistory(months = 6) {
    try {
      const res = await fetch(`/portfolio/history?months=${encodeURIComponent(months)}`, { credentials: "same-origin" });
      if (!res.ok) return;
      const j = await res.json();
      setHistory(j.history || []);
      renderLineChart(j.history || []);
    } catch (e) {
      console.error("loadPortfolioHistory error", e);
    }
  }

  function renderAllocChart(alloc) {
    try {
      if (!window.Chart) return;
      const labels = Object.keys(alloc);
      const data = Object.values(alloc);
      const canvas = allocChartRef.current;
      if (!canvas) return;
      if (allocChartInstance.current) {
        try { allocChartInstance.current.destroy(); } catch (e) { /* ignore */ }
        allocChartInstance.current = null;
      }
      const ctx = canvas.getContext("2d");
      const palette = ["#4f46e5", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#3b82f6", "#f97316", "#14b8a6"];
      allocChartInstance.current = new Chart(ctx, {
        type: "pie",
        data: { labels, datasets: [{ data, backgroundColor: labels.map((_, i) => palette[i % palette.length]) }] },
        options: { plugins: { legend: { position: "bottom" } }, maintainAspectRatio: false }
      });
    } catch (e) {
      console.warn("alloc chart failed", e);
    }
  }

  function renderLineChart(hist) {
    try {
      if (!window.Chart) return;
      const canvas = lineChartRef.current;
      if (!canvas) return;
      if (lineChartInstance.current) {
        try { lineChartInstance.current.destroy(); } catch (e) {}
        lineChartInstance.current = null;
      }
      const labels = hist.map(r => r.date);
      const values = hist.map(r => Number(r.value || 0));
      const ctx = canvas.getContext("2d");
      lineChartInstance.current = new Chart(ctx, {
        type: "line",
        data: { labels, datasets: [{ label: "Portfolio Value (₹)", data: values, fill: true, tension: 0.15, pointRadius: 2 }] },
        options: { responsive: true, interaction: { mode: "index", intersect: false }, scales: { y: { beginAtZero: false } } }
      });
    } catch (e) {
      console.warn("line chart failed", e);
    }
  }

  async function uploadPortfolioFile() {
    const file = fileRef.current?.files?.[0];
    if (!file) {
      alert("Select a portfolio CSV/XLSX file");
      return;
    }
    setUploading(true);
    try {
      const fd = new FormData(); fd.append("file", file);
      const res = await fetch("/portfolio/upload", { method: "POST", body: fd, credentials: "same-origin" });
      const j = await res.json();
      if (res.ok && j.success) {
        await loadDashboard();
        await loadPortfolioHistory(6);
      } else {
        alert(j.error || JSON.stringify(j));
      }
    } catch (e) {
      console.error("upload error", e);
      alert("Upload failed — see console.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function logout() {
    try {
      const res = await fetch("/auth/logout", { method: "POST", credentials: "include", headers: { "Content-Type": "application/json" } });
      const j = await res.json().catch(() => ({}));
      if (j && j.success) window.location.href = "/login";
      else window.location.href = "/login";
    } catch (e) {
      console.error("logout failed", e);
      window.location.href = "/login";
    }
  }

  // ------------------------------
  // IPO analyzer flows
  // ------------------------------
  function openIpoAnalyzer() {
    setIpoAnalyzerOpen(true);
    setIpoResultHtml("");
    if (ipoFileRef.current) ipoFileRef.current.value = "";
    if (ipoTextRef.current) ipoTextRef.current.value = "";
  }
  function closeIpoAnalyzer() {
    setIpoAnalyzerOpen(false);
    setIpoResultHtml("");
  }

  async function analyzeIpoSubmit(ev) {
    ev?.preventDefault?.();
    setIpoLoading(true);
    setIpoResultHtml("");
    setAnalyzing(true);
    try {
      const file = ipoFileRef.current?.files?.[0];
      const pasted = (ipoTextRef.current?.value || "").trim();
      let res;
      if (file) {
        const fd = new FormData();
        fd.append("ipoFile", file);
        res = await fetch("/ipo/analyze", { method: "POST", body: fd, credentials: "same-origin" });
      } else if (pasted) {
        res = await fetch("/ipo/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: pasted }),
          credentials: "same-origin"
        });
      } else {
        alert("Please upload a PDF or paste IPO content.");
        setIpoLoading(false);
        setAnalyzing(false);
        return;
      }

      if (!res.ok) {
        const t = await res.text().catch(() => null);
        throw new Error(t || `HTTP ${res.status}`);
      }
      const j = await res.json();
      if (j.error) throw new Error(j.error || "Unknown error from server");

      const md = j.ipo_report_md || j.ipo_report || "";
      let html = "";
      if (ShowdownConverter) {
        try {
          const conv = new ShowdownConverter({ tables: true, tasklists: true, ghCompatibleHeaderId: true });
          html = conv.makeHtml(md || "");
        } catch (e) {
          html = "<pre style='white-space:pre-wrap;'>" + escapeHtml(md || "") + "</pre>";
        }
      } else if (typeof window !== "undefined" && window.showdown) {
        try {
          const conv = new window.showdown.Converter({ tables: true, tasklists: true });
          html = conv.makeHtml(md || "");
        } catch (e) {
          html = "<pre style='white-space:pre-wrap;'>" + escapeHtml(md || "") + "</pre>";
        }
      } else {
        html = "<pre style='white-space:pre-wrap;'>" + escapeHtml(md || "") + "</pre>";
      }
      setIpoResultHtml(html);

      // If server saved and returned report_id, refresh list
      if (j.report_id) {
        // show saved hint and refresh reports
        await loadIpoReports();
      }
    } catch (err) {
      console.error("ipo analyze error", err);
      setIpoResultHtml(`<div class="text-sm text-red-600">Analysis failed: ${escapeHtml(err.message || String(err))}</div>`);
    } finally {
      setIpoLoading(false);
      setAnalyzing(false);
    }
  }

  function escapeHtml(s) {
    if (!s) return "";
    return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
  }

  async function loadIpoReports() {
    try {
      const res = await fetch("/ipo/list", { credentials: "same-origin" });
      if (res.status === 401) {
        setIpoReports([]);
        return;
      }
      const j = await res.json();
      setIpoReports(j.reports || []);
    } catch (e) {
      console.error("loadIpoReports error", e);
      setIpoReports([]);
    }
  }

  async function viewIpoReport(id) {
    try {
      const res = await fetch(`/ipo/get/${id}`, { credentials: "same-origin" });
      if (!res.ok) {
        const j = await res.json().catch(() => null);
        alert(j?.error || "Failed to load report");
        return;
      }
      const j = await res.json();
      const md = j.content_md || "";
      let html = "";
      if (ShowdownConverter) {
        try {
          const conv = new ShowdownConverter({ tables: true, tasklists: true });
          html = conv.makeHtml(md || "");
        } catch (e) {
          html = "<pre style='white-space:pre-wrap;'>" + escapeHtml(md || "") + "</pre>";
        }
      } else if (typeof window !== "undefined" && window.showdown) {
        try {
          const conv = new window.showdown.Converter({ tables: true, tasklists: true });
          html = conv.makeHtml(md || "");
        } catch (e) {
          html = "<pre style='white-space:pre-wrap;'>" + escapeHtml(md || "") + "</pre>";
        }
      } else {
        html = "<pre style='white-space:pre-wrap;'>" + escapeHtml(md || "") + "</pre>";
      }
      setIpoModalContent(html);
      setIpoModalMeta(j.created_at ? `Created: ${new Date(j.created_at).toLocaleString()}` : "");
      setIpoModalOpen(true);
    } catch (e) {
      console.error("viewIpoReport err", e);
      alert("Failed to fetch report — see console.");
    }
  }

  function closeIpoModal() {
    setIpoModalOpen(false);
    setIpoModalContent("");
    setIpoModalMeta("");
  }

  function toggleIpoCheckbox(id) {
    const copy = new Set(selectedIpoIds);
    if (copy.has(id)) copy.delete(id);
    else copy.add(id);
    setSelectedIpoIds(copy);
  }

  async function compareSelectedIpos() {
    if (!selectedIpoIds || selectedIpoIds.size === 0) {
      alert("Select at least 1 report to compare.");
      return;
    }
    const ids = Array.from(selectedIpoIds).join(",");
    try {
      const res = await fetch(`/ipo/compare?ids=${encodeURIComponent(ids)}`, { credentials: "same-origin" });
      if (!res.ok) {
        const j = await res.json().catch(() => null);
        alert(j?.error || "Failed to compare");
        return;
      }
      const j = await res.json();
      // show simple comparison modal: we'll reuse ipoModal to show an HTML table
      const rows = j.comparison || [];
      let table = `<div style="overflow:auto;"><table style="width:100%;border-collapse:collapse">`;
      table += `<thead><tr style="background:#f3f4f6"><th style="padding:8px">Title</th><th style="padding:8px">Created At</th><th style="padding:8px">Revenue</th><th style="padding:8px">Profit</th><th style="padding:8px">Debt</th><th style="padding:8px">Promoter</th><th style="padding:8px">Score</th></tr></thead><tbody>`;
      rows.forEach(r => {
        table += `<tr><td style="padding:8px">${escapeHtml(r.title || "")}</td><td style="padding:8px">${r.created_at ? new Date(r.created_at).toLocaleString() : ""}</td><td style="padding:8px">${escapeHtml(String(r.revenue || "—"))}</td><td style="padding:8px">${escapeHtml(String(r.profit || "—"))}</td><td style="padding:8px">${escapeHtml(String(r.debt || "—"))}</td><td style="padding:8px">${r.promoter_mentioned ? "Yes":"No"}</td><td style="padding:8px">${r.overall_score ?? "N/A"}</td></tr>`;
      });
      table += `</tbody></table></div>`;
      setIpoModalContent(table);
      setIpoModalMeta("Comparison Results");
      setIpoModalOpen(true);
    } catch (e) {
      console.error("compareSelectedIpos error", e);
      alert("Compare failed — see console.");
    }
  }

  // ------------------------------
  // UI
  // ------------------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 p-6">
      <header className="flex items-center justify-between max-w-7xl mx-auto mb-6">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-xl flex items-center justify-center text-white font-bold">S</div>
          <div>
            <h1 className="text-xl font-semibold">SEBI Saathi</h1>
            <p className="text-sm text-gray-600">Portfolio Dashboard</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Link to="/" className="px-4 py-2 bg-gray-100 rounded-md">Chat Assistant</Link>
          <button onClick={() => { loadDashboard(); loadPortfolioHistory(6); }} className="px-4 py-2 bg-gray-100 rounded-md">Refresh</button>
          <button onClick={logout} className="px-4 py-2 bg-red-500 text-white rounded-md">Logout</button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 xl:grid-cols-4 gap-6">
        <section className="xl:col-span-3 bg-white rounded-2xl border p-6 shadow-sm">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold">Portfolio Overview</h2>
              <p className="text-sm text-gray-500">Your holdings and performance</p>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-600">Total Portfolio Value</p>
              <p className="text-2xl font-bold">{dashboard?.total_value ? `₹${Number(dashboard.total_value).toLocaleString('en-IN')}` : "—"}</p>
            </div>
          </div>

          <div className="overflow-auto rounded-xl border mb-4">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs font-semibold text-gray-700">
                <tr>
                  <th className="p-4 text-left">Symbol</th>
                  <th className="p-4 text-left">Quantity</th>
                  <th className="p-4 text-left">Avg Price</th>
                  <th className="p-4 text-left">Last Price</th>
                  <th className="p-4 text-left">Current Value</th>
                </tr>
              </thead>
              <tbody>
                {(dashboard?.holdings || []).map(h => (
                  <tr key={h.symbol} className="hover:bg-gray-50">
                    <td className="p-4 font-medium">{h.symbol}</td>
                    <td className="p-4">{h.quantity}</td>
                    <td className="p-4">₹{h.avg_price ?? "—"}</td>
                    <td className="p-4">₹{h.last_price ?? "—"}</td>
                    <td className="p-4 font-semibold">₹{Number(h.current_value || 0).toLocaleString('en-IN')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-white rounded-lg p-4 border">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h4 className="font-semibold">Portfolio value — last 6 months</h4>
                <p className="text-xs text-gray-500">Backend route: /portfolio/history?months=6</p>
              </div>
              <button onClick={() => loadPortfolioHistory(6)} className="text-xs px-3 py-1 bg-gray-100 rounded-md">Refresh</button>
            </div>
            <div style={{ height: 320 }}>
              <canvas ref={lineChartRef} style={{ width: "100%", height: "100%" }}></canvas>
            </div>
          </div>
        </section>

        <aside className="bg-white rounded-2xl border p-6 shadow-sm">
          <h3 className="font-semibold mb-4">Portfolio Actions</h3>

          <div className="mb-6 p-4 bg-gray-50 rounded-lg border">
            <label className="block text-sm font-semibold mb-2">Upload Portfolio</label>
            <input ref={fileRef} type="file" accept=".csv,.xls,.xlsx" className="block w-full text-sm mb-3" />
            <button onClick={uploadPortfolioFile} disabled={uploading} className="w-full bg-indigo-600 text-white py-2 rounded-lg">
              {uploading ? "Uploading..." : "Upload & Analyze"}
            </button>
          </div>

          <div className="p-4 bg-gradient-to-br from-gray-50 to-slate-50 rounded-lg border">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-semibold">Asset Allocation</h4>
              <button onClick={() => renderAllocChart(dashboard?.allocations || {})} className="text-xs px-2 py-1 bg-gray-100 rounded-md">Refresh</button>
            </div>
            <div className="h-48">
              <canvas ref={allocChartRef} width="280" height="280" style={{ width: "100%", height: "100%" }}></canvas>
            </div>
          </div>

          <div className="mt-4">
            <button onClick={openIpoAnalyzer} className="w-full bg-blue-600 text-white py-2 rounded-lg">Open IPO Analyzer</button>
          </div>
        </aside>

        <section className="xl:col-span-3 mt-4">
          <div className="bg-white rounded-xl p-6 border">
            <h3 className="font-semibold mb-3">Market Alerts</h3>
            <div className="space-y-3">
              {(dashboard?.alerts || []).map(a => (
                <div key={a.symbol} className="p-3 rounded-lg bg-amber-50 border border-amber-200">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-semibold text-amber-900">{a.symbol} — {a.sentiment}</div>
                      <div className="text-xs text-amber-700">Est move: {a.estimated_pct_move}%</div>
                    </div>
                  </div>
                  <ul className="list-disc list-inside mt-2 text-amber-700 text-sm">
                    {(a.headlines || []).slice(0, 5).map((h, i) => <li key={i}>{h}</li>)}
                  </ul>
                </div>
              ))}
              {(!dashboard || !dashboard.alerts || dashboard.alerts.length === 0) && <div className="text-sm text-gray-500">No alerts</div>}
            </div>
          </div>
        </section>

        <section className="xl:col-span-3 mt-6">
          <div className="bg-white rounded-xl p-6 border">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold">IPO Reports</h3>
              <div className="flex items-center gap-2">
                <button onClick={loadIpoReports} className="px-3 py-1 bg-gray-100 rounded-md">Refresh</button>
                <button onClick={compareSelectedIpos} className="px-3 py-1 bg-amber-500 text-white rounded-md">Compare Selected</button>
              </div>
            </div>
            <div id="ipoList" className="space-y-3">
              {ipoReports.length === 0 && <div className="text-sm text-gray-600">No saved IPO reports yet.</div>}
              {ipoReports.map(r => (
                <div key={r.id} className="p-3 border border-gray-100 rounded-lg flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <input type="checkbox" checked={selectedIpoIds.has(r.id)} onChange={() => toggleIpoCheckbox(r.id)} className="mt-2" />
                    <div>
                      <div className="font-semibold text-gray-900">{r.title || `Report #${r.id}`}</div>
                      <div className="text-sm text-gray-500 mt-1">{r.created_at ? new Date(r.created_at).toLocaleString() : ''}</div>
                      <div className="mt-2 text-xs text-gray-500">Score: {r.sebi_score ?? r.llm_score ?? 'N/A'}</div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <button onClick={() => viewIpoReport(r.id)} className="px-3 py-1 bg-indigo-600 text-white rounded-md text-sm">View</button>
                    <a href={`/ipo/get/${r.id}`} target="_blank" rel="noopener noreferrer" className="text-xs text-gray-600 hover:underline">Open JSON</a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* IPO Analyzer Modal */}
      {ipoAnalyzerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black opacity-40" onClick={closeIpoAnalyzer}></div>
          <div className="relative bg-white rounded-2xl p-6 z-50 w-full max-w-3xl max-h-[90vh] overflow-auto border">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold">IPO Analyzer</h3>
                <p className="text-sm text-gray-600">Upload a prospectus PDF or paste content. Results will be saved to your account.</p>
              </div>
              <div className="flex gap-2">
                <button onClick={closeIpoAnalyzer} className="px-3 py-1 bg-gray-100 rounded-lg text-sm">Close</button>
              </div>
            </div>

            <form onSubmit={analyzeIpoSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Upload IPO Document (PDF)</label>
                <input ref={ipoFileRef} type="file" accept=".pdf,.txt" className="w-full border rounded-lg px-3 py-2" />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Or paste IPO content</label>
                <textarea ref={ipoTextRef} className="w-full border rounded-lg px-3 py-2 h-36" placeholder="Paste prospectus text (first 100k chars recommended)"></textarea>
              </div>

              <div className="flex gap-3">
                <button type="submit" disabled={ipoLoading} className="bg-indigo-600 text-white px-4 py-2 rounded-lg">
                  {ipoLoading ? "Analyzing..." : "Analyze"}
                </button>
                <button type="button" onClick={() => { /* save handled automatically on analyze if backend returns id */ }} className="bg-green-600 text-white px-4 py-2 rounded-lg hidden">Save report</button>
                <button type="button" onClick={() => { loadIpoReports(); }} className="bg-gray-100 px-4 py-2 rounded-lg">Refresh Reports</button>
              </div>
            </form>

            <div className="mt-4 prose max-w-none text-sm" dangerouslySetInnerHTML={{ __html: ipoResultHtml || "<div class='text-sm text-gray-500'>No result yet</div>" }}></div>
          </div>
        </div>
      )}

      {/* IPO View / Compare Modal */}
      {ipoModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black opacity-40" onClick={closeIpoModal}></div>
          <div className="relative bg-white rounded-2xl p-6 z-50 w-full max-w-4xl max-h-[85vh] overflow-auto border">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold">IPO Report</h3>
                <p className="text-sm text-gray-600">{ipoModalMeta}</p>
              </div>
              <div className="flex gap-2 items-center">
                <button onClick={() => { window.open("", "_blank"); }} className="text-sm text-gray-600 hover:text-gray-900 hidden">Open raw</button>
                <button onClick={closeIpoModal} className="px-3 py-1 bg-gray-100 rounded-lg text-sm">Close</button>
              </div>
            </div>

            <div className="prose max-w-none text-sm" dangerouslySetInnerHTML={{ __html: ipoModalContent || "<div class='text-sm text-gray-500'>No content</div>" }}></div>
          </div>
        </div>
      )}
    </div>
  );
}
