// frontend/src/pages/ChatAssistant.jsx
import React, { useEffect, useRef, useState, useContext } from "react";
import { AuthContext } from "../context/AuthContext";

/**
 * ChatAssistant page — uses backend endpoints:
 * /ask, /ask_user, /market/live, /sebi/circulars, /get_user_library, /ipo/analyze,
 * /auth/logout, /quiz/next_question, /get_myth, /calculate_sip, /sources, /analyze
 */

export default function ChatAssistant() {
  const { user } = useContext(AuthContext);

  // chat state
  const [chatMessages, setChatMessages] = useState([
    { sender: "bot", text: "Welcome to SEBI Saathi!" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // data state
  const [market, setMarket] = useState({});
  const [sebi, setSebI] = useState([]);
  const [userLibrary, setUserLibrary] = useState([]);
  const [sources, setSources] = useState([]);

  // feature toggles & modals
  const [scamOpen, setScamOpen] = useState(false);
  const [sipOpen, setSipOpen] = useState(false);
  const [mythOpen, setMythOpen] = useState(false);
  const [ipoOpen, setIpoOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);

  // quiz / myth / sip state
  const [scamQuestion, setScamQuestion] = useState(null);
  const [scamFeedback, setScamFeedback] = useState("");
  const [mythQuestion, setMythQuestion] = useState(null);
  const [mythFeedback, setMythFeedback] = useState("");

  const [sipGoal, setSipGoal] = useState("");
  const [sipAmount, setSipAmount] = useState("");
  const [sipYears, setSipYears] = useState("");
  const [sipResult, setSipResult] = useState(null);

  // IPO
  const ipoFileRef = useRef(null);
  const [ipoContent, setIpoContent] = useState("");
  const [ipoResultHtml, setIpoResultHtml] = useState("");

  // misc
  const [searchMyLibrary, setSearchMyLibrary] = useState(false);
  const [lastContextPreview, setLastContextPreview] = useState(null); // show context preview returned by backend
  const [statusMsg, setStatusMsg] = useState(null); // small transient messages

  // refs
  const chatContainerRef = useRef(null);
  const sipChartRef = useRef(null);

  useEffect(() => {
    // auto-scroll chat
    const el = chatContainerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chatMessages]);

  useEffect(() => {
    loadLiveMarket();
    loadSebicirculars();
    loadUserLibrary();
    const mi = setInterval(loadLiveMarket, 180000); // 3m
    const si = setInterval(loadSebicirculars, 24 * 60 * 60 * 1000);
    return () => { clearInterval(mi); clearInterval(si); };
  }, []);

  async function loadLiveMarket() {
    try {
      const res = await fetch("/market/live");
      if (!res.ok) return;
      setMarket(await res.json());
    } catch (e) { console.error("loadLiveMarket", e); }
  }

  async function loadSebicirculars() {
    try {
      const res = await fetch("/sebi/circulars");
      if (!res.ok) return;
      const j = await res.json();
      setSebI(j.circulars || []);
    } catch (e) { console.error("loadSebicirculars", e); }
  }

  async function loadUserLibrary() {
    try {
      const res = await fetch("/get_user_library", { credentials: "same-origin" });
      if (!res.ok) return setUserLibrary([]);
      const j = await res.json();
      setUserLibrary(j.user_library || []);
    } catch (e) { console.error("loadUserLibrary", e); }
  }

  async function loadSources() {
    try {
      const res = await fetch("/sources", { credentials: "same-origin" });
      const txt = await res.text();
      let j = {};
      try { j = txt ? JSON.parse(txt) : {}; } catch { j = {}; }
      setSources(j.sources || []);
    } catch (e) { console.error("loadSources", e); }
  }

  // ------------------------------
  // Chat / Ask
  // ------------------------------
  async function handleSend() {
    const q = (input || "").trim();
    if (!q) return;
    // prevent double sends
    if (loading) return;
    pushMessage({ sender: "user", text: q });
    setInput("");
    setLoading(true);
    setLastContextPreview(null);
    setStatusMsg(searchMyLibrary ? "Searching your library..." : "Searching knowledge base...");

    try {
      // If user asked to search their library, call the dedicated endpoint
      const endpoint = searchMyLibrary ? "/ask_user" : "/ask";

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ question: q })
      });

      const text = await res.text();
      let j = {};
      try { j = text ? JSON.parse(text) : {}; } catch (e) {
        // server returned non-JSON — present readable fallback
        j = { answer: text || `Server returned HTTP ${res.status}` };
      }

      if (!res.ok) {
        const err = j.error || `HTTP ${res.status}`;
        pushMessage({ sender: "bot", text: `Error: ${err}` });
        console.warn("ask error payload:", j);
      } else {
        const answer = j.answer || j.error || "No answer.";
        pushMessage({ sender: "bot", text: answer });

        // show context preview if backend returned it (useful for debugging RAG)
        if (j.context_preview) {
          setLastContextPreview(j.context_preview);
          // also push a small bot message summarizing whether context existed
          const hasContext = (j.context_preview || "").trim() && !(j.context_preview || "").startsWith("No relevant information");
          pushMessage({ sender: "bot", text: hasContext ? "(Answer based on your library context — preview available below.)" : "(No relevant context found in your library.)" });
        } else {
          setLastContextPreview(null);
        }

        // debug object if available
        if (j.debug) {
          console.debug("ask debug:", j.debug);
        }
      }
    } catch (e) {
      pushMessage({ sender: "bot", text: `Sorry, an error occurred: ${e.message || e}` });
      console.error("handleSend error", e);
    } finally {
      setLoading(false);
      setStatusMsg(null);
    }
  }

  function pushMessage(m) { setChatMessages(prev => [...prev, m]); }

  // ------------------------------
  // Scam / Myth handlers
  // ------------------------------
  async function loadScamQuestion() {
    setScamFeedback(""); setScamQuestion(null);
    try {
      const res = await fetch("/quiz/next_question");
      if (!res.ok) throw new Error("no question");
      setScamQuestion(await res.json());
    } catch (e) {
      console.error("loadScamQuestion", e);
      setScamQuestion({ message: "Loading failed." });
    }
  }

  function checkScam(choice) {
    if (!scamQuestion) return;
    const isCorrect = choice.toLowerCase() === (scamQuestion.type || "").toLowerCase();
    setScamFeedback(`${isCorrect ? "Correct!" : "Incorrect."} It was ${scamQuestion.type}. ${scamQuestion.explanation || ""}`);
  }

  async function loadMyth() {
    setMythFeedback(""); setMythQuestion(null);
    try {
      const res = await fetch("/get_myth");
      if (!res.ok) throw new Error("no myth");
      setMythQuestion(await res.json());
    } catch (e) {
      console.error("loadMyth", e);
      setMythQuestion({ statement: "Failed to load myth." });
    }
  }

  function checkMyth(choice) {
    if (!mythQuestion) return;
    const isCorrect = choice === mythQuestion.type;
    setMythFeedback(`${isCorrect ? "Correct!" : "Not quite."} It's a ${mythQuestion.type}. ${mythQuestion.explanation || ""}`);
  }

  // ------------------------------
  // SIP calculation & chart
  // ------------------------------
  async function handleCalculateSip() {
    setSipResult(null);
    if (!sipAmount || !sipYears) { setSipResult({ error: "Please fill fields" }); return; }
    try {
      const res = await fetch("/calculate_sip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ amount: Number(sipAmount), years: Number(sipYears) }),
        credentials: "same-origin"
      });
      const j = await res.json();
      if (j.error) setSipResult({ error: j.error }); else setSipResult(j);
    } catch (e) {
      console.error("handleCalculateSip", e);
      setSipResult({ error: e.message || "Request failed" });
    }
  }

  // Render SIP chart when sipResult arrives
  useEffect(() => {
    if (!sipResult || !Array.isArray(sipResult.growth_data) || sipResult.growth_data.length === 0) return;

    let destroyFn = null;
    (async () => {
      try {
        // dynamic import of chart.js/auto - OK in React apps
        const ChartModule = await import("chart.js/auto");
        const Chart = ChartModule && (ChartModule.default || ChartModule.Chart || ChartModule);

        const ctx = sipChartRef.current && sipChartRef.current.getContext && sipChartRef.current.getContext("2d");
        if (!ctx) return;

        // destroy existing
        if (window._sipChart) {
          try { window._sipChart.destroy(); } catch (e) { /* ignore */ }
          window._sipChart = null;
        }

        const labels = sipResult.growth_data.map(d => `Y${d.year}`);
        const invested = sipResult.growth_data.map(d => Number(d.invested) || 0);
        const value = sipResult.growth_data.map(d => Number(d.value) || 0);

        window._sipChart = new Chart(ctx, {
          type: "bar",
          data: {
            labels,
            datasets: [
              { label: "Invested", data: invested, backgroundColor: "rgba(59,130,246,0.8)" },
              { label: "Value",    data: value,    backgroundColor: "rgba(16,185,129,0.8)" }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true } }
          }
        });

        destroyFn = () => { if (window._sipChart) try { window._sipChart.destroy(); window._sipChart = null; } catch {} };

      } catch (err) {
        console.error("Failed to render SIP chart", err);
      }
    })();

    return () => { if (destroyFn) destroyFn(); };
  }, [sipResult]);

  // ------------------------------
  // IPO analyze
  // ------------------------------
  async function handleIpoAnalyze(ev) {
    ev?.preventDefault();
    setIpoResultHtml("Analyzing IPO — please wait...");
    try {
      const file = ipoFileRef.current?.files?.[0];
      let res;
      if (file) {
        const fd = new FormData(); fd.append("ipoFile", file);
        res = await fetch("/ipo/analyze", { method: "POST", body: fd, credentials: "same-origin" });
      } else if (ipoContent && ipoContent.trim()) {
        res = await fetch("/ipo/analyze", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: ipoContent }), credentials: "same-origin"
        });
      } else {
        setIpoResultHtml("Please upload a PDF or paste IPO content.");
        return;
      }

      const text = await res.text();
      let j = {};
      try { j = text ? JSON.parse(text) : {}; } catch { j = { error: "Unexpected server response" }; }
      if (j.error) setIpoResultHtml(`Analysis failed: ${j.error}`); else setIpoResultHtml(j.ipo_report_md || j.ipo_report || JSON.stringify(j));
    } catch (e) {
      console.error("handleIpoAnalyze", e);
      setIpoResultHtml(`Error: ${e.message || e}`);
    }
  }

  // ------------------------------
  // Portfolio analyze (CSV/XLSX)
  // ------------------------------
  async function handlePortfolioAnalyze(file) {
    if (!file) return;
    pushMessage({ sender: "user", text: `Analyzing portfolio: ${file.name}` });
    setLoading(true);

    const fd = new FormData();
    fd.append("portfolioFile", file);

    try {
      const res = await fetch("/analyze", {
        method: "POST",
        body: fd,
        credentials: "same-origin",
      });

      const text = await res.text();
      let j = {};
      try { j = text ? JSON.parse(text) : {}; } catch {
        pushMessage({ sender: "bot", text: "Server returned unexpected response. You may need to login." });
        if (res.status === 401) window.location.href = "/login";
        return;
      }

      if (j.error) {
        pushMessage({ sender: "bot", text: `Analysis failed: ${j.error}` });
        return;
      }

      const html = createPortfolioPage(j.analysis_markdown || "", j.chart_data || {});
      const win = window.open("", "_blank");
      if (!win) { pushMessage({ sender: "bot", text: "Could not open analysis window. Please allow popups." }); return; }
      win.document.open(); win.document.write(html); win.document.close();

      pushMessage({ sender: "bot", text: "Portfolio analyzed — report opened in a new tab." });
      loadUserLibrary();
    } catch (err) {
      console.error("Portfolio analyze error", err);
      pushMessage({ sender: "bot", text: `Portfolio analysis failed: ${err.message || err}` });
    } finally { setLoading(false); }
  }

  // ------------------------------
  // Upload doc to user library
  // ------------------------------
  async function handleDocUploadFile(file) {
    if (!file) return;
    pushMessage({ sender: "user", text: `Uploading ${file.name}` });

    try {
      const fd = new FormData();
      fd.append('userFile', file);

      const res = await fetch('/upload_and_ingest', {
        method: 'POST',
        body: fd,
        credentials: 'same-origin'
      });

      const text = await res.text();
      let j = {};
      try { j = text ? JSON.parse(text) : {}; } catch { /* non-json response */ }

      if (res.status === 401) {
        pushMessage({ sender: "bot", text: 'You must be logged in to upload documents. Redirecting to login...' });
        setTimeout(() => { window.location.href = '/login'; }, 700);
        return;
      }

      if (!res.ok) {
        const errMsg = j.error || `HTTP ${res.status}`;
        pushMessage({ sender: "bot", text: `Upload failed: ${errMsg}` });
        console.error('Upload failed:', res.status, text);
        return;
      }

      if (j.success) {
        pushMessage({ sender: "bot", text: `${file.name} uploaded to library.` });
        await loadUserLibrary();
      } else {
        pushMessage({ sender: "bot", text: `Upload failed: ${j.error || 'unknown'}` });
        console.warn('Upload returned success=false:', j);
      }
    } catch (err) {
      console.error('Upload exception', err);
      pushMessage({ sender: "bot", text: `Upload failed: ${err.message || err}` });
    }
  }

  // ------------------------------
  // Delete file
  // ------------------------------
  async function handleDeleteFile(filename) {
    if (!confirm(`Delete ${filename}?`)) return;
    pushMessage({ sender: "bot", text: `Deleting ${filename}...` });
    try {
      const res = await fetch('/delete_user_file', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });
      if (res.status === 401) { pushMessage({ sender: "bot", text: 'You must be logged in to delete files.' }); return; }
      const j = await res.json();
      if (j.success) {
        pushMessage({ sender: "bot", text: `${filename} deleted.` });
        setUserLibrary(j.user_library || []);
      } else throw new Error(j.error || 'Delete failed');
    } catch (e) {
      console.error('delete error', e);
      pushMessage({ sender: "bot", text: `Delete failed: ${e.message}` });
    }
  }

  // ------------------------------
  // small helpers & UI renderers
  // ------------------------------
  function formatMarketItem(id, info) {
    if (!info || info.price === null || info.price === undefined) {
      return <div className="p-2 bg-blue-50 rounded-md text-sm" key={id}>{id.replace('-', ' ')} — --</div>;
    }
    const arrow = info.change >= 0 ? "🔺" : "🔻";
    const color = info.change >= 0 ? "text-green-600" : "text-red-600";
    return (
      <div className="p-2 bg-blue-50 rounded-md text-sm flex justify-between" key={id}>
        <span className="uppercase">{id.replace('-', ' ')}</span>
        <span className={`${color} font-bold`}>₹{info.price} {arrow} ({info.pct_change}%)</span>
      </div>
    );
  }

  // ------------------------------
  // JSX
  // ------------------------------
  return (
    <div className="min-h-screen bg-white font-sans text-gray-800">
      <div className="flex h-screen overflow-hidden">

        {/* Left sidebar */}
        <aside className="w-72 bg-blue-50 border-r border-gray-200 flex flex-col shadow-sm fixed left-0 top-0 h-full overflow-y-auto p-6">
          <div className="border-b border-gray-200 pb-4 mb-4">
            <h1 className="text-2xl font-display font-bold text-primary">SEBI Saathi</h1>
            <p className="text-sm text-muted font-medium mt-1">Financial Copilot</p>
          </div>

          <nav className="flex-1 space-y-4">
            <div>
              <h3 className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">Financial Tools</h3>
              <div className="space-y-2">
                <button onClick={() => { setScamOpen(true); loadScamQuestion(); }} className="w-full flex items-center gap-3 py-3 px-4 rounded-lg hover:bg-white hover:shadow-sm transition text-left">
                  <span className="text-lg">🛡️</span><span className="font-medium text-gray-700">Scam Simulator</span>
                </button>
                <button onClick={() => setSipOpen(true)} className="w-full flex items-center gap-3 py-3 px-4 rounded-lg hover:bg-white hover:shadow-sm transition text-left">
                  <span className="text-lg">📈</span><span className="font-medium text-gray-700">SIP Planner</span>
                </button>
                <button onClick={() => { setMythOpen(true); loadMyth(); }} className="w-full flex items-center gap-3 py-3 px-4 rounded-lg hover:bg-white hover:shadow-sm transition text-left">
                  <span className="text-lg">🔍</span><span className="font-medium text-gray-700">Myth Buster</span>
                </button>
                <button onClick={() => setIpoOpen(true)} className="w-full flex items-center gap-3 py-3 px-4 rounded-lg hover:bg-white hover:shadow-sm transition text-left">
                  <span className="text-lg">📊</span><span className="font-medium text-gray-700">IPO Analyzer</span>
                </button>
              </div>
            </div>

            <div>
              <h3 className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">Navigation</h3>
              <div className="space-y-2">
                <button onClick={() => { setSourcesOpen(true); loadSources(); }} className="w-full flex items-center gap-3 py-3 px-4 rounded-lg hover:bg-white hover:shadow-sm transition text-left">
                  <span className="text-lg">📚</span><span className="font-medium text-gray-700">Sources</span>
                </button>
                <a href="/dashboard" className="w-full flex items-center gap-3 py-3 px-4 rounded-lg hover:bg-white hover:shadow-sm transition text-left">
                  <span className="text-lg">📊</span><span className="font-medium text-gray-700">Portfolio Dashboard</span>
                </a>
              </div>
            </div>
          </nav>

          <div className="mt-auto pt-6 border-t border-gray-200">
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <h3 className="text-sm font-semibold text-gray-800 mb-2">My Library</h3>

              <div className="text-xs text-muted mb-3">
                {userLibrary.length === 0 ? (
                  <div className="text-xs text-gray-400">Your library is empty.</div>
                ) : (
                  <ul className="space-y-2">
                    {userLibrary.map((f) => (
                      <li key={f} className="flex items-center justify-between">
                        <span className="truncate text-sm">{f}</span>
                        <div className="flex items-center gap-2">
                          <a
                            className="text-indigo-600 text-xs hover:underline"
                            href={`/user_source_view/${encodeURIComponent(f)}`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Open
                          </a>
                          <button
                            onClick={() => handleDeleteFile(f)}
                            className="text-red-500 text-xs hover:underline"
                            title={`Delete ${f}`}
                          >
                            Delete
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <label className="w-full inline-block bg-primary hover:bg-blue-700 text-white py-2.5 px-4 rounded-lg text-sm font-medium text-center cursor-pointer">
                📄 Upload Documents
                <input
                  type="file"
                  id="user-file-input"
                  name="userFile"
                  accept=".pdf"
                  className="hidden"
                  onChange={async (e) => {
                    const f = e.target.files && e.target.files[0];
                    if (!f) return;
                    await handleDocUploadFile(f);
                    e.target.value = "";
                  }}
                />
              </label>
            </div>

            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center"><span className="text-white text-sm font-semibold">S</span></div>
                <span className="text-sm font-medium text-gray-800">{user || "guest"}</span>
              </div>
              <button onClick={async () => { await fetch("/auth/logout", { method: "POST", credentials: "same-origin" }); window.location.href = "/login"; }} className="text-danger hover:text-red-700 text-sm font-medium">Logout</button>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 flex flex-col w-[calc(100%-18rem)] ml-72">
          <header className="bg-gradient-to-r from-blue-50 to-blue-100 border-b border-gray-200 px-6 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-display font-bold text-gray-900">Welcome to <span className="text-primary">SEBI Saathi</span></h1>
                <p className="text-muted text-sm mt-1">Your AI-powered financial literacy and compliance companion</p>
              </div>
            </div>
          </header>

          <div className="flex flex-1 overflow-hidden">
            <section className="w-full flex flex-col p-6 h-full">
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col h-full">
                <div className="p-4 border-b border-gray-200 flex-shrink-0">
                  <h2 className="font-display font-semibold text-gray-900">Financial Assistant</h2>
                  <p className="text-sm text-muted">Ask me anything about investments, regulations, or financial planning</p>
                </div>

                <div ref={chatContainerRef} id="chat-container" className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
                  {chatMessages.map((m, idx) => (
                    <div key={idx} className={`flex mb-2 ${m.sender === "user" ? "justify-end" : "justify-start"}`}>
                      <div className={`p-3 rounded-2xl max-w-md shadow ${m.sender === "user" ? "bg-blue-700 text-white" : "bg-indigo-500 text-white"}`}>
                        <p>{m.text}</p>
                      </div>
                    </div>
                  ))}

                  {/* show context preview (muted box) if available */}
                  {lastContextPreview && (
                    <div className="text-xs text-gray-500 border border-gray-100 bg-gray-50 rounded-md p-3">
                      <div className="font-semibold text-gray-700 mb-2">Context preview (from your library)</div>
                      <pre className="whitespace-pre-wrap text-[13px] max-h-48 overflow-auto">{lastContextPreview}</pre>
                    </div>
                  )}
                </div>

                <div className={`${loading ? "block" : "hidden"} px-4 py-2 flex-shrink-0`}>
                  <div className="flex items-center gap-2 text-muted">
                    <div className="animate-spin w-4 h-4 border-2 border-primary border-t-transparent rounded-full"></div>
                    <span className="text-sm">Thinking...</span>
                  </div>
                </div>

                <div className="p-4 bg-white border-t border-gray-200 rounded-b-xl flex-shrink-0">
                  <div className="flex items-center gap-3">
                    <input
                      id="user-input"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleSend()}
                      type="text"
                      placeholder="Ask about investments, SEBI regulations, portfolio analysis..."
                      className="flex-1 border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-primary transition-all duration-200"
                      disabled={loading}
                    />

                    <label className="flex items-center gap-2 text-sm ml-2">
                      <input
                        id="search-scope-toggle"
                        type="checkbox"
                        className="h-4 w-4"
                        checked={searchMyLibrary}
                        onChange={(e) => {
                          setSearchMyLibrary(e.target.checked);
                          // show small transient status
                          setStatusMsg(e.target.checked ? "Search scope: My library" : "Search scope: Knowledge base");
                          setTimeout(() => setStatusMsg(null), 1600);
                        }}
                      />
                      <span className="text-xs text-muted">Search my library</span>
                    </label>

                    <button onClick={handleSend} disabled={loading} className="bg-primary hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium">Send</button>

                    <button onClick={() => document.getElementById("file-input")?.click()} className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-3 rounded-lg font-medium">📊 Analyze</button>

                    <input id="file-input" type="file" className="hidden" accept=".csv,.xlsx,.xls" onChange={(e) => {
                      const f = e.target.files && e.target.files[0];
                      if (f) handlePortfolioAnalyze(f);
                      e.target.value = "";
                    }} />
                  </div>

                  {statusMsg && <div className="mt-2 text-xs text-gray-500">{statusMsg}</div>}
                </div>
              </div>
            </section>

            <aside className="w-80 bg-gray-50 border-l border-gray-200 p-6 flex-col gap-6 sticky top-0 h-screen overflow-y-auto">
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="font-display font-semibold text-gray-900 text-base">Live Market</h2>
                  <span className="text-[10px] bg-success text-white px-2 py-0.5 rounded-full">LIVE</span>
                </div>
                <p className="text-[11px] text-muted mb-3">Updated every few minutes</p>
                <div className="space-y-2">
                  {formatMarketItem("nifty-price", market["NIFTY 50"])}
                  {formatMarketItem("sensex-price", market["SENSEX"])}
                  {formatMarketItem("cdsl-price", market["CDSL"])}
                  {formatMarketItem("KFINTECH-price", market["KFINTECH"])}
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 flex-1">
                <h2 className="font-display font-semibold text-gray-900 text-base mb-3">SEBI Updates</h2>
                <div className="news-section space-y-2 overflow-y-auto flex-1 text-sm text-gray-700">
                  {sebi.length === 0 ? <div className="text-gray-500">No SEBI circulars found.</div> : sebi.map((c,i) => (
                    <div key={i} className="p-3 bg-gray-50 rounded-lg shadow-sm hover:shadow-md transition">
                      <a href={c.url} target="_blank" rel="noreferrer" className="text-indigo-600 hover:underline text-sm flex justify-between">
                        <span>{c.date}</span><span>{c.title}</span>
                      </a>
                    </div>
                  ))}
                </div>
                <div className="mt-3"><button onClick={() => { setSourcesOpen(true); loadSources(); }} className="text-xs text-gray-700 hover:text-primary">View full sources</button></div>
              </div>
            </aside>
          </div>
        </main>
      </div>

      {/* Modals (same as your original but included here for completeness) */}
      {scamOpen && (
        <Modal onClose={() => setScamOpen(false)}>
          <div className="text-center mb-6">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4"><span className="text-2xl">🛡️</span></div>
            <h2 className="text-xl font-display font-bold text-gray-900">Scam or Legit?</h2>
          </div>
          <p id="scam-question" className="text-gray-700 mb-6 text-center">{scamQuestion ? scamQuestion.message : "Loading..."}</p>
          <div className="flex gap-3 mb-4">
            <button onClick={() => checkScam("scam")} className="flex-1 bg-red-500 hover:bg-red-600 text-white py-3 rounded-lg">Scam</button>
            <button onClick={() => checkScam("legit")} className="flex-1 bg-green-500 hover:bg-green-600 text-white py-3 rounded-lg">Legit</button>
          </div>
          <div id="scam-feedback" className="text-sm text-gray-600 mb-4">{scamFeedback}</div>
          <div className="flex gap-2">
            <button onClick={loadScamQuestion} className="w-full bg-gray-800 hover:bg-gray-900 text-white py-3 rounded-lg">Next Question</button>
            <button onClick={() => setScamOpen(false)} className="w-full bg-gray-200 text-gray-800 py-3 rounded-lg">Close</button>
          </div>
        </Modal>
      )}

      {sipOpen && (
        <Modal onClose={() => setSipOpen(false)}>
          <div className="text-center mb-6">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4"><span className="text-2xl">📈</span></div>
            <h2 className="text-xl font-display font-bold text-gray-900">SIP Calculator</h2>
          </div>
          <div className="space-y-4">
            <input value={sipGoal} onChange={(e)=>setSipGoal(e.target.value)} placeholder="Goal (e.g. Retirement)" className="w-full border rounded-lg px-4 py-3" />
            <input value={sipAmount} onChange={(e)=>setSipAmount(e.target.value)} placeholder="Target Amount (₹)" className="w-full border rounded-lg px-4 py-3" />
            <input value={sipYears} onChange={(e)=>setSipYears(e.target.value)} placeholder="Years" className="w-full border rounded-lg px-4 py-3" />
            <button onClick={handleCalculateSip} className="w-full bg-primary text-white py-3 rounded-lg">Calculate SIP</button>

            {sipResult && (
              <div className="mt-2">
                {sipResult.error ? <div className="text-red-600 text-sm">{sipResult.error}</div> : (
                  <>
                    <div className="text-sm">Monthly SIP: <strong>₹{(sipResult.monthly_sip || 0).toLocaleString('en-IN')}</strong></div>
                    {Array.isArray(sipResult.growth_data) && sipResult.growth_data.length > 0 && (
                      <div className="mt-4" style={{ height: 240 }}>
                        <canvas ref={sipChartRef} style={{ width: '100%', height: '100%' }} />
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
          <div className="mt-4"><button onClick={() => setSipOpen(false)} className="w-full bg-gray-200 text-gray-800 py-3 rounded-lg">Close</button></div>
        </Modal>
      )}

      {mythOpen && (
        <Modal onClose={() => setMythOpen(false)}>
          <div className="text-center mb-6">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4"><span className="text-2xl">🔍</span></div>
            <h2 className="text-xl font-display font-bold text-gray-900">Myth Buster</h2>
          </div>
          <p id="myth-statement" className="text-gray-700 mb-6 text-center">{mythQuestion ? mythQuestion.statement : "Loading..."}</p>
          <div className="flex gap-3 mb-4">
            <button onClick={() => checkMyth("Myth")} className="flex-1 bg-red-500 hover:bg-red-600 text-white py-3 rounded-lg">Myth</button>
            <button onClick={() => checkMyth("Fact")} className="flex-1 bg-green-500 hover:bg-green-600 text-white py-3 rounded-lg">Fact</button>
          </div>
          <div id="myth-feedback" className="mt-3 text-sm text-gray-600">{mythFeedback}</div>
          <div className="mt-4">
            <button onClick={loadMyth} className="w-full bg-gray-800 text-white py-3 rounded-lg mb-2">Next Question</button>
            <button onClick={() => setMythOpen(false)} className="w-full bg-gray-200 text-gray-800 py-3 rounded-lg">Close</button>
          </div>
        </Modal>
      )}

      {ipoOpen && (
        <Modal onClose={() => setIpoOpen(false)} wide>
          <div className="text-center mb-6">
            <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4"><span className="text-2xl">📊</span></div>
            <h2 className="text-xl font-display font-bold text-gray-900">IPO Analyzer</h2>
          </div>
          <form onSubmit={handleIpoAnalyze} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Upload IPO Document</label>
              <input ref={ipoFileRef} type="file" id="ipo-file" name="ipoFile" accept=".pdf,.txt" className="block w-full border rounded-lg px-3 py-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Or paste IPO content</label>
              <textarea value={ipoContent} onChange={(e)=>setIpoContent(e.target.value)} id="ipo-content" name="content" placeholder="Paste IPO prospectus text here..." className="w-full border rounded-lg px-4 py-3 h-32" />
            </div>
            <button type="submit" className="w-full bg-primary text-white py-3 rounded-lg">Analyze IPO</button>
          </form>
          <div className="mt-4 prose prose-sm max-w-none">
            {ipoResultHtml ? <div dangerouslySetInnerHTML={{__html: ipoResultHtml.replace(/\n/g, "<br/>")}} /> : <div className="text-sm text-gray-500">No report yet.</div>}
          </div>
          <div className="mt-4"><button onClick={() => setIpoOpen(false)} className="w-full bg-gray-200 text-gray-800 py-3 rounded-lg">Close</button></div>
        </Modal>
      )}

      {sourcesOpen && (
        <Modal onClose={() => setSourcesOpen(false)}>
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Knowledge Sources</h3>
            <p className="text-sm text-muted mt-1">Documents, circulars and trusted links used for answers</p>
          </div>
          <div className="space-y-2 text-sm text-gray-700 max-h-56 overflow-y-auto">
            {sources.length === 0 ? <div className="text-gray-500">No sources found.</div> : sources.map(s => (
              <div key={s.name} className="flex items-center justify-between p-2 border rounded">
                <div className="text-sm truncate pr-3">{s.name}</div>
                <div className="flex items-center gap-2">
                  <a className="text-indigo-600 hover:underline text-sm" href={s.viewer_url || `/source_view/${encodeURIComponent(s.name)}`} target="_blank" rel="noreferrer">Open</a>
                  <a className="text-muted text-xs" href={s.url} target="_blank" rel="noreferrer">Raw</a>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4"><button onClick={() => setSourcesOpen(false)} className="w-full bg-gray-200 text-gray-800 py-3 rounded-lg">Close</button></div>
        </Modal>
      )}
    </div>
  );
}

/* Modal component */
function Modal({ children, onClose, wide = false }) {
  return (
    <div className="modal fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={(e) => { if (e.target.classList.contains('modal')) onClose(); }}>
      <div className={`bg-white p-6 rounded-2xl shadow-2xl ${wide ? "w-[90%] max-h-[90vh] overflow-y-auto relative" : "w-96 relative"}`}>
        <button onClick={onClose} className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 text-xl">✕</button>
        {children}
      </div>
    </div>
  );
}

/* createPortfolioPage: returns full HTML for new window to render analysis_markdown + doughnut chart */
function createPortfolioPage(md, chartData = {}) {
  const mdEscaped = (md || "").replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
  const chartJson = JSON.stringify(chartData || {});
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Portfolio Analysis - SEBI Saathi</title>
<script src="https://cdn.jsdelivr.net/npm/showdown/dist/showdown.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  body{font-family: 'Inter',system-ui,Segoe UI,Roboto,Helvetica,Arial; background:#f8fafc; margin:0; padding:24px;}
  .card{background:#fff;border:1px solid #e6edf3;border-radius:12px;padding:18px;box-shadow:0 6px 18px rgba(12,24,40,0.04);max-width:1100px;margin:0 auto;}
  .chart-box{height:380px}
  .prose{line-height:1.5;color:#111827}
  .legend-swatch{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:8px}
</style>
</head>
<body>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <div><h1 style="margin:0;font-size:20px;">Portfolio Analysis</h1><div style="font-size:13px;color:#6b7280">SEBI Saathi — generated report</div></div>
      <div><button onclick="window.print()" style="background:#10b981;color:#fff;border:0;padding:8px 12px;border-radius:8px;cursor:pointer">Export</button></div>
    </div>

    <div style="display:flex;gap:20px;align-items:flex-start">
      <div style="flex:1;min-width:360px">
        <div id="analysis" class="prose"></div>
      </div>
      <div style="width:360px;flex:0 0 360px">
        <div class="chart-box"><canvas id="chart" style="width:100%;height:100%"></canvas></div>
        <div id="legend" style="margin-top:12px;font-size:13px;color:#374151"></div>
      </div>
    </div>
  </div>

<script>
  (function(){
    const converter = new showdown.Converter({tables:true,tasklists:true});
    const md = \`${mdEscaped}\`;
    document.getElementById('analysis').innerHTML = md ? converter.makeHtml(md) : '<p style="color:#6b7280">No analysis returned.</p>';

    const data = ${chartJson} || {};
    const labels = Object.keys(data);
    const values = Object.values(data).map(v => Number(v) || 0);
    const palette = ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#06b6d4','#f97316','#64748b','#a78bfa','#60a5fa'];
    const colors = labels.map((_,i)=> palette[i % palette.length]);

    const ctx = document.getElementById('chart').getContext('2d');
    try {
      new Chart(ctx, {
        type: 'doughnut',
        data: { labels, datasets: [{ data: values, backgroundColor: colors, borderColor: '#ffffff', borderWidth: 2 }] },
        options: { cutout: '58%', plugins: { legend: { display:false } }, maintainAspectRatio: false, responsive: true }
      });
    } catch (e) { console.error(e); }

    const legend = document.getElementById('legend');
    if (labels.length === 0) {
      legend.innerHTML = '<div style="color:#6b7280">No composition data available.</div>';
    } else {
      legend.innerHTML = labels.map((l,i) => '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px"><span class="legend-swatch" style="background:'+colors[i]+'"></span><span>'+l+': <strong>'+ (Number(values[i]) || 0) +'</strong></span></div>').join('');
    }
  })();
</script>
</body>
</html>`;
}
