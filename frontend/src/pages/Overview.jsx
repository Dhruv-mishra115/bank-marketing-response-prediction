import { AlertTriangle, Activity, Database, Target, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, KpiCard, MiniLabel, SectionTitle } from "../components.jsx";
import { CHART_TIP, COLORS, count, decimal, displayModel, percent, shortModel } from "../data.js";

export function OverviewPage({ metrics, onNavigate }) {
  const d = metrics.dataset, model = metrics.model;
  const validation = model.validation_comparison || [];
  const bestRoc = [...validation].sort((a, b) => b.roc_auc - a.roc_auc)[0];
  const bestAp = [...validation].sort((a, b) => b.average_precision - a.average_precision)[0];
  const pie = [{ name: "Subscribed", value: d.target_counts.yes, fill: COLORS.teal }, { name: "Not subscribed", value: d.target_counts.no, fill: COLORS.blue }];
  const modelChart = validation.slice(0, 6).map((row) => ({ name: shortModel(row.model), roc: row.roc_auc, ap: row.average_precision }));
  const monthly = d.monthly_rates.map((row) => ({ month: row.month.toUpperCase(), rate: row.subscription_rate * 100 }));
  return <>
    <div className="kpi-grid">
      <KpiCard label="Dataset records" value={count(d.records)} sub="UCI bank-additional-full" icon={Database} tone="teal" />
      <KpiCard label="Subscribers" value={count(d.target_counts.yes)} sub={percent(d.target_rates.yes) + " of records"} icon={Users} tone="blue" />
      <KpiCard label="Best validation ROC-AUC" value={decimal(bestRoc?.roc_auc, 4)} sub={displayModel(bestRoc?.model)} icon={Activity} tone="violet" />
      <KpiCard label="Best validation AP" value={decimal(bestAp?.average_precision, 4)} sub={displayModel(bestAp?.model)} icon={Target} tone="amber" />
    </div>
    <div className="content-grid overview-grid">
      <Card className="chart-card span-7"><SectionTitle title="Validation model comparison" subtitle="ROC-AUC and Average Precision · validation only" action={<button className="text-link" onClick={() => onNavigate("performance")}>Explore models <span>→</span></button>} />
        <div className="chart-legend"><MiniLabel color={COLORS.teal}>ROC-AUC</MiniLabel><MiniLabel color={COLORS.blue}>Average Precision</MiniLabel><span className="legend-note">Higher is better</span></div>
        <div className="chart-box chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={modelChart} layout="vertical" margin={{ top: 8, right: 24, bottom: 0, left: 10 }} barGap={4}><CartesianGrid stroke="#25323e" horizontal={false} /><XAxis type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} stroke="#8492a1" tickLine={false} axisLine={false} /><YAxis dataKey="name" type="category" width={154} stroke="#aebac7" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} /><Tooltip {...CHART_TIP} formatter={(value) => decimal(value, 4)} /><Bar dataKey="roc" name="ROC-AUC" fill={COLORS.teal} radius={[0, 4, 4, 0]} barSize={8} /><Bar dataKey="ap" name="Average Precision" fill={COLORS.blue} radius={[0, 4, 4, 0]} barSize={8} /></BarChart></ResponsiveContainer></div>
      </Card>
      <Card className="chart-card span-5"><SectionTitle title="Target distribution" subtitle={count(d.records) + " source records"} />
        <div className="distribution-layout"><div className="donut-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie isAnimationActive={false} data={pie} dataKey="value" nameKey="name" innerRadius="70%" outerRadius="94%" paddingAngle={3} stroke="none">{pie.map((entry) => <Cell key={entry.name} fill={entry.fill} />)}</Pie><Tooltip {...CHART_TIP} formatter={(value) => count(value)} /></PieChart></ResponsiveContainer><div className="donut-center"><strong>{percent(d.target_rates.yes)}</strong><span>subscriber rate</span></div></div><div className="distribution-key"><div><MiniLabel color={COLORS.teal}>Yes</MiniLabel><strong>{count(d.target_counts.yes)}</strong><span>{percent(d.target_rates.yes)}</span></div><div><MiniLabel color={COLORS.blue}>No</MiniLabel><strong>{count(d.target_counts.no)}</strong><span>{percent(d.target_rates.no)}</span></div><p>Imbalanced target classes are a central evaluation consideration.</p></div></div>
      </Card>
      <Card className="chart-card span-7"><SectionTitle title="Response by contact month" subtitle="Observed rate in the full source file" /><div className="chart-box chart-small"><ResponsiveContainer width="100%" height="100%"><BarChart data={monthly} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}><CartesianGrid stroke="#25323e" vertical={false} /><XAxis dataKey="month" stroke="#8492a1" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} /><YAxis stroke="#8492a1" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} tickFormatter={(v) => v + "%"} /><Tooltip {...CHART_TIP} formatter={(value) => Number(value).toFixed(1) + "%"} /><Bar dataKey="rate" name="Observed subscription rate" fill={COLORS.teal} radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div></Card>
      <Card className="insight-card span-5"><div className="insight-badge"><AlertTriangle size={17} /></div><span className="eyebrow">MODEL READINESS</span><h3>Validation precision constraint not met</h3><p>No tested candidate reached the requested {percent(model.validation_precision_floor)} precision floor while predicting any subscribers. The best validation AP candidate is shown for learning and diagnostics only.</p><button className="text-link" onClick={() => onNavigate("performance")}>Review validation evidence <span>→</span></button></Card>
    </div>
    <div className="overview-bottom"><div className="meta-strip"><span><Database size={15} /> Selected diagnostic candidate</span><strong>{displayModel(model.selected_candidate)}</strong><span className="meta-divider" /><span>Prediction timing</span><strong>Before current call</strong></div><div className="source-note">Source: <a href={metrics.project.dataset_url} target="_blank" rel="noreferrer">UCI Machine Learning Repository ↗</a></div></div>
  </>;
}
