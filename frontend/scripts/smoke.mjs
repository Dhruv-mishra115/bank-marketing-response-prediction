import { strict as assert } from "node:assert";
import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.CHROME_EXECUTABLE_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe",
  args: ["--no-sandbox"],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const consoleErrors = [];
page.on("pageerror", (error) => consoleErrors.push(error.message));

try {
  const response = await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle" });
  assert.equal(response.status(), 200, "Vite app should load");
  await page.getByRole("heading", { name: "Response intelligence" }).waitFor();
  await page.getByText("41,188", { exact: true }).first().waitFor();
  const chartState = await page.evaluate(() => ({
    piePaths: document.querySelectorAll(".recharts-sector").length,
  }));
  assert.equal(chartState.piePaths, 2, "Overview class-balance chart should render both classes");
  await page.screenshot({ path: "../reports/figures/dashboard-overview.png", fullPage: true });
  console.log("Overview rendered with API metrics.");

  await page.getByRole("button", { name: "Customer prediction" }).click();
  await page.getByRole("heading", { name: "Client profile" }).waitFor();
  await page.getByRole("button", { name: "Predict response" }).click();
  await page.getByText("MODEL RESPONSE").waitFor();
  await page.getByText("Subscription probability").waitFor();
  const predictionText = await page.locator(".result-card").innerText();
  assert.match(predictionText, /LIVE API/);
  assert.match(predictionText, /Decision threshold/);
  console.log("Prediction form received a backend response.");

  await page.context().setOffline(true);
  await page.getByRole("button", { name: "Predict response" }).click();
  const errorAlert = page.getByRole("alert");
  await errorAlert.waitFor();
  const errorText = await errorAlert.innerText();
  assert.match(errorText, /fetch|network|failed/i);
  await page.context().setOffline(false);
  console.log("Backend-unavailable form state rendered.");
  await page.getByRole("button", { name: "Model performance" }).click();
  await page.getByRole("heading", { name: "Model comparison" }).waitFor();
  await page.getByText("PREVIOUSLY EXPOSED").first().waitFor();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("button", { name: "Dataset insights" }).click();
  await page.getByRole("heading", { name: "Unknown categorical values" }).waitFor();
  await page.waitForFunction(() => !document.querySelector(".sidebar")?.classList.contains("sidebar-open"));
  await page.waitForTimeout(250);
  assert.equal(await page.locator(".recharts-sector").count(), 2, "Dataset class-balance chart should render both classes");
  const horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  assert.equal(horizontalOverflow, false, "Mobile layout should not overflow horizontally");
  const layout = await page.evaluate(() => ({
    gridColumns: getComputedStyle(document.querySelector(".kpi-grid")).gridTemplateColumns,
    kpiCards: document.querySelectorAll(".kpi-card").length,
    donutSectors: document.querySelectorAll(".recharts-sector").length,
  }));
  assert.equal(layout.donutSectors, 2, "Mobile class-balance chart should render both classes");
  await mkdir("../reports/figures", { recursive: true });
  await page.screenshot({ path: "../reports/figures/dashboard-mobile.png", fullPage: true });

  assert.deepEqual(consoleErrors, [], "Browser should not report uncaught page errors");
  console.log("Browser smoke test passed: overview metrics, live prediction, offline handling, model performance, mobile navigation and layout.");
  console.log("Prediction response rendered:", predictionText.replace(/\s+/g, " ").slice(0, 220));
} finally {
  await browser.close();
}
