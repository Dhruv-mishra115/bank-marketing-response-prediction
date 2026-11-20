export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export const NAV = [
  { id: "overview", label: "Overview" },
  { id: "prediction", label: "Customer prediction" },
  { id: "performance", label: "Model performance" },
  { id: "dataset", label: "Dataset insights" },
  { id: "workflow", label: "ML workflow" },
  { id: "architecture", label: "Architecture" },
  { id: "documentation", label: "Project notes" },
];

export const COLORS = { teal: "#69dfc4", blue: "#8ba7ff", red: "#fa817f", amber: "#e9bc69", muted: "#8c9aa9" };
export const CHART_TIP = { contentStyle: { background: "#111a24", border: "1px solid #293746", borderRadius: 10, color: "#edf3f8" }, labelStyle: { color: "#aebbc9" }, itemStyle: { color: "#edf3f8" } };

export const FEATURE_HELP = {
  age: "Client age in years.", job: "Occupation group recorded by the campaign.", marital: "Recorded marital status.",
  education: "Education category; unknown is retained as a dataset category.", default: "Whether the client has credit in default.",
  housing: "Whether the client has a housing loan.", loan: "Whether the client has a personal loan.",
  contact: "Contact communication channel.", month: "Month of the current marketing contact.",
  day_of_week: "Weekday of the current marketing contact.", pdays: "Days since a previous campaign contact; 999 denotes no previous contact in this dataset.",
  previous: "Number of contacts before the current campaign.", poutcome: "Outcome of the previous marketing campaign.",
  "emp.var.rate": "Employment variation rate at campaign time.", "cons.price.idx": "Consumer price index recorded in the source data.",
  "cons.conf.idx": "Consumer confidence index recorded in the source data.", euribor3m: "Three-month Euribor rate recorded in the source data.",
  "nr.employed": "Number of employees recorded in the source data.",
};

export const LABELS = {
  age: "Age", job: "Job", marital: "Marital status", education: "Education", default: "Credit in default",
  housing: "Housing loan", loan: "Personal loan", contact: "Contact channel", month: "Contact month",
  day_of_week: "Day of week", pdays: "Days since prior contact", previous: "Prior campaign contacts",
  poutcome: "Previous campaign outcome", "emp.var.rate": "Employment variation rate",
  "cons.price.idx": "Consumer price index", "cons.conf.idx": "Consumer confidence index",
  euribor3m: "Euribor 3 month", "nr.employed": "Number employed",
};

export const MODEL_LABELS = {
  random_forest_balanced: "Random Forest · class balanced",
  random_forest_unweighted: "Random Forest · unweighted",
  random_forest_balanced_larger_leaf: "Random Forest · larger leaf",
  random_forest_balanced_shallower: "Random Forest · shallower",
  decision_tree: "Decision Tree",
  logistic_regression: "Logistic Regression",
  class_weighted_logistic: "Logistic Regression · weighted",
  calibrated_random_forest: "Random Forest · calibrated",
};

export const percent = (value, digits = 1) => Number.isFinite(Number(value)) ? (Number(value) * 100).toFixed(digits) + "%" : "Not available";
export const decimal = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "Not available";
export const count = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "Not available";
export const displayModel = (name) => MODEL_LABELS[name] || name || "Not available";

export async function getJSON(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "The service returned an error.");
  return body;
}

export function shortModel(name) {
  return ({ random_forest_balanced: "RF balanced", random_forest_unweighted: "RF unweighted", random_forest_balanced_larger_leaf: "RF larger leaf", random_forest_balanced_shallower: "RF shallower", decision_tree: "Decision Tree", logistic_regression: "Logistic Regression", class_weighted_logistic: "Weighted Logistic", calibrated_random_forest: "RF calibrated" })[name] || name;
}
