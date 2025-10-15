#!/usr/bin/env node

import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");

const rawBaseUrl = process.env.AE_DASHBOARD_URL ?? "http://127.0.0.1:8000/ui/";
const baseUrl = rawBaseUrl.endsWith("/") ? rawBaseUrl : `${rawBaseUrl}/`;
const outputDir = process.env.AE_SCREENSHOT_DIR
  ? path.resolve(process.env.AE_SCREENSHOT_DIR)
  : path.join(repoRoot, "artifacts", "latest", "screenshots");
fs.mkdirSync(outputDir, { recursive: true });
const overviewPath = path.join(outputDir, "pipeline-overview.png");
const settingsPath = path.join(outputDir, "settings-presets.png");
const manifestPath = path.join(outputDir, "manifest.json");
for (const filePath of [overviewPath, settingsPath, manifestPath]) {
  if (fs.existsSync(filePath)) {
    fs.rmSync(filePath, { force: true });
  }
}

const log = (message) => {
  console.log(`[screenshots] ${message}`);
};

async function captureScreenshots() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on("console", (msg) => log(`console:${msg.type()} ${msg.text()}`));
  page.on("pageerror", (error) => log(`pageerror ${error}`));
  page.on("response", (response) => {
    const status = response.status();
    if (status >= 400) {
      log(`response ${status} ${response.url()}`);
    }
  });

  try {
    log(`Opening ${baseUrl}`);
    const response = await page.goto(baseUrl, { waitUntil: "networkidle" });
    if (!response || !response.ok()) {
      const status = response ? `${response.status()} ${response.statusText()}` : "no response";
      log(`Warning: dashboard responded with ${status}`);
    }

    await page.waitForSelector('[data-test="overview-layout"]', { state: "visible", timeout: 20000 });
    await page.waitForTimeout(600);
    await page.screenshot({ path: overviewPath, fullPage: true });
    log(`Saved overview screenshot to ${overviewPath}`);

    await page.click('[data-test="nav-settings"]');
    await page.waitForSelector('[data-test="settings-grid"] .settings-table', { state: "visible", timeout: 20000 });
    await page.waitForTimeout(600);
    await page.screenshot({ path: settingsPath, fullPage: true });
    log(`Saved settings screenshot to ${settingsPath}`);

    const manifest = {
      timestamp: new Date().toISOString(),
      baseUrl,
      outputDir: path.relative(repoRoot, outputDir),
      files: {
        overview: path.relative(repoRoot, overviewPath),
        settings: path.relative(repoRoot, settingsPath),
      },
    };
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf8");
    log(`Wrote manifest ${manifestPath}`);
  } finally {
    await browser.close();
  }
}

captureScreenshots().catch((error) => {
  console.error("[screenshots] Failed to capture screenshots:", error);
  process.exitCode = 1;
});
