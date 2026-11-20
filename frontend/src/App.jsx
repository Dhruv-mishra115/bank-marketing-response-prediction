import { useEffect, useState } from "react";
import {
  Activity, AlertTriangle, BookOpen, ChartColumnIncreasing, Database, FileText,
  Gauge, GitBranch, Layers3, LoaderCircle, Menu, Network, RefreshCw,
  ShieldAlert, Target, Users, WalletCards, X, BriefcaseBusiness,
} from "lucide-react";
import { API_BASE, NAV, displayModel, getJSON } from "./data.js";
import { DatasetPage } from "./pages/Dataset.jsx";
import { DocumentationPage } from "./pages/Documentation.jsx";
import { OverviewPage } from "./pages/Overview.jsx";
import { PerformancePage } from "./pages/Performance.jsx";
import { PredictionPage } from "./pages/Prediction.jsx";
import { ArchitecturePage } from "./pages/Architecture.jsx";
import { WorkflowPage } from "./pages/Workflow.jsx";

const ICONS = { overview: Gauge, prediction: Target, performance: ChartColumnIncreasing, dataset: Database, workflow: BookOpen, architecture: Network, documentation: FileText };
const PAGE_DESCRIPTIONS = {
  overview: "A measured view of the data, validation evidence and current model status.",
  prediction: "Send a feature-complete request to the existing FastAPI model pipeline.",
  performance: "Validation comparisons, threshold trade-offs and the previously exposed final-segment diagnostic.",
  dataset: "A source-grounded look at the UCI Bank Marketing data used by this project.",
  workflow: "How the project moves from a prediction question to a validated inference pipeline.",
  architecture: "A clear boundary between data, the model service and this frontend.",
  documentation: "Project scope, technology choices, limitations and portfolio summary.",
};

export default function App() {
  const [page, setPage] = useState("overview");
  const [metrics, setMetrics] = useState(null);
  const [apiState, setApiState] = useState("checking");
  const [loadError, setLoadError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let alive = true;
    setLoadError("");
    setApiState("checking");
    Promise.allSettled([getJSON(API_BASE + "/healthz"), getJSON(API_BASE + "/dashboard/metrics")]).then(([health, dashboard]) => {
      if (!alive) return;
      setApiState(health.status === "fulfilled" ? (health.value.model_loaded ? "connected" : "model-missing") : "offline");
      if (dashboard.status === "fulfilled") setMetrics(dashboard.value);
      else setLoadError(dashboard.reason?.message || "Dashboard data is unavailable.");
    });
    return () => { alive = false; };
  }, [refreshKey]);

  useEffect(() => {
    const listener = (event) => setPage(event.detail);
    window.addEventListener("navigate-page", listener);
    return () => window.removeEventListener("navigate-page", listener);
  }, []);

  const current = NAV.find((item) => item.id === page) || NAV[0];
  const PageIcon = ICONS[page] || Gauge;
  const navigate = (id) => {
    setPage(id);
    setSidebarOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const renderPage = () => {
    if (loadError && !metrics) return <ErrorState message={loadError} onRetry={() => setRefreshKey((key) => key + 1)} />;
    if (!metrics) return <LoadingState />;
    switch (page) {
      case "prediction": return <PredictionPage metrics={metrics} apiState={apiState} />;
      case "performance": return <PerformancePage metrics={metrics} />;
      case "dataset": return <DatasetPage metrics={metrics} />;
      case "workflow": return <WorkflowPage metrics={metrics} />;
      case "architecture": return <ArchitecturePage metrics={metrics} />;
      case "documentation": return <DocumentationPage metrics={metrics} />;
      default: return <OverviewPage metrics={metrics} onNavigate={navigate} />;
    }
  };

  return <div className="app-shell">
    {sidebarOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
    <aside className={"sidebar " + (sidebarOpen ? "sidebar-open" : "")}>
      <div className="brand-lockup">
        <div className="brand-mark"><Activity size={19} strokeWidth={2.2} /></div>
        <div><strong>Northstar</strong><span>MODEL STUDIO</span></div>
        <button className="icon-button sidebar-close" aria-label="Close menu" onClick={() => setSidebarOpen(false)}><X size={18} /></button>
      </div>
      <div className="workspace-label">WORKSPACE</div>
      <div className="project-switcher"><div className="project-initial">BM</div><div><strong>Bank Marketing</strong><span>Response prediction</span></div><Layers3 size={15} /></div>
      <div className="workspace-label nav-label">ANALYSIS</div>
      <nav aria-label="Main navigation" className="side-nav">
        {NAV.map(({ id, label }) => {
          const Icon = ICONS[id] || Gauge;
          return <button key={id} className={"nav-item " + (page === id ? "active" : "")} onClick={() => navigate(id)} aria-current={page === id ? "page" : undefined}>
            <Icon size={17} strokeWidth={1.8} /><span>{label}</span>{id === "prediction" && <span className="nav-dot" />}
          </button>;
        })}
      </nav>
      <div className="sidebar-bottom">
        <div className="side-health"><span className={"health-dot " + apiState} /><div><strong>{apiState === "connected" ? "API connected" : apiState === "checking" ? "Checking API" : apiState === "model-missing" ? "Model unavailable" : "API offline"}</strong><span>Local inference service</span></div></div>
        <div className="sidebar-foot"><span>Portfolio project</span><span>v1.0</span></div>
      </div>
    </aside>
    <main className="main-shell">
      <header className="topbar">
        <button className="icon-button menu-toggle" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Menu size={19} /></button>
        <div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>{current.label}</strong></div>
        <div className="topbar-right"><div className={"api-pill " + apiState}><span className="health-dot" />{apiState === "connected" ? "Backend online" : apiState === "checking" ? "Connecting" : apiState === "model-missing" ? "Model missing" : "Backend offline"}</div><div className="avatar" aria-label="Portfolio workspace">BM</div></div>
      </header>
      <div className="page-wrap">
        <div className="page-heading"><div><div className="eyebrow"><PageIcon size={14} /> BANK MARKETING ANALYTICS <span className="eyebrow-dot">/</span> {current.label.toUpperCase()}</div><h1>{page === "overview" ? "Response intelligence" : current.label}</h1><p>{PAGE_DESCRIPTIONS[page]}</p></div><button className="button button-quiet refresh-button" onClick={() => setRefreshKey((key) => key + 1)}><RefreshCw size={15} /> Refresh data</button></div>
        {metrics?.project?.status && <div className="status-banner"><span className="status-icon"><ShieldAlert size={17} /></span><span><strong>Experimental model</strong><span>Not approved for real customer targeting. Validation did not meet the requested precision floor.</span></span><span className="status-tag">RESEARCH ONLY</span></div>}
        {loadError && metrics && <div className="inline-alert"><AlertTriangle size={16} /> Some dashboard data could not refresh: {loadError}</div>}
        {renderPage()}
        <footer className="page-footer"><span>Bank Marketing Response Prediction</span><span>Experimental portfolio project · Metrics served by FastAPI</span></footer>
      </div>
    </main>
  </div>;
}

function LoadingState() { return <div className="loading-panel"><LoaderCircle className="spin" size={24} /><strong>Loading project evidence</strong><span>Connecting to the local FastAPI service…</span></div>; }
function ErrorState({ message, onRetry }) { return <div className="empty-panel"><div className="empty-icon"><AlertTriangle size={21} /></div><h2>Dashboard data unavailable</h2><p>{message}</p><p className="muted">Start the FastAPI service, then refresh this page.</p><button className="button button-primary" onClick={onRetry}><RefreshCw size={15} /> Retry connection</button></div>; }
