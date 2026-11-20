import { AlertTriangle, LoaderCircle } from "lucide-react";
import { count } from "./data.js";

export function SectionTitle({ title, subtitle, action }) {
  return <div className="section-title"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>;
}

export function Card({ className = "", children }) {
  return <section className={"card " + className}>{children}</section>;
}

export function KpiCard({ label, value, sub, icon: Icon, tone = "teal", tag }) {
  return <Card className="kpi-card"><div className="kpi-top"><span>{label}</span><span className={"kpi-icon " + tone}><Icon size={17} /></span></div><div className="kpi-value">{value}</div><div className="kpi-bottom"><span>{sub}</span>{tag && <span className="kpi-tag">{tag}</span>}</div></Card>;
}

export function MiniLabel({ children, color }) {
  return <span className="mini-label"><i style={{ background: color }} />{children}</span>;
}

export function ConfusionHeatmap({ matrix, compact = false }) {
  const max = Math.max(...matrix.flat());
  const labels = [["True negative", matrix[0][0]], ["False positive", matrix[0][1]], ["False negative", matrix[1][0]], ["True positive", matrix[1][1]]];
  return <div className={"confusion-wrap " + (compact ? "confusion-compact" : "")}><div className="confusion-axis-title">Predicted class</div><div className="confusion-table"><div className="confusion-y">Actual<br />class</div><div className="confusion-body"><div className="confusion-cols"><span>No</span><span>Yes</span></div><div className="confusion-row"><span className="confusion-rowlabel">No</span>{matrix[0].map((value, i) => <div className="heat-cell" key={i} style={{ "--heat": Math.max(.13, value / max) }}><strong>{count(value)}</strong><small>{i === 0 ? "True negative" : "False positive"}</small></div>)}</div><div className="confusion-row"><span className="confusion-rowlabel">Yes</span>{matrix[1].map((value, i) => <div className="heat-cell positive" key={i} style={{ "--heat": Math.max(.13, value / max) }}><strong>{count(value)}</strong><small>{i === 0 ? "False negative" : "True positive"}</small></div>)}</div></div></div><div className="confusion-axis-foot">{labels.map(([label, value]) => <span key={label}>{label}: {count(value)}</span>)}</div></div>;
}

export function ErrorState({ message, onRetry }) {
  return <div className="empty-panel"><div className="empty-icon"><AlertTriangle size={21} /></div><h2>Dashboard data unavailable</h2><p>{message}</p><p className="muted">Start the FastAPI service, then refresh this page.</p><button className="button button-primary" onClick={onRetry}>Retry connection</button></div>;
}

export function LoadingState() {
  return <div className="loading-panel"><LoaderCircle className="spin" size={24} /><strong>Loading project evidence</strong><span>Connecting to the local FastAPI service…</span></div>;
}
