import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const updateScript = path.resolve(__dirname, "scripts", "updateArtifacts.mjs");
const DEV_CAPTURE_URL = process.env.AE_DEV_CAPTURE_URL ?? "http://127.0.0.1:5173/";
const autoCaptureEnabled = Boolean(process.env.AE_ENABLE_AUTOCAPTURE) && !process.env.AE_SKIP_AUTOCAPTURE;

if (!autoCaptureEnabled) {
  console.info("[artifacts] Auto capture disabled (set AE_ENABLE_AUTOCAPTURE=1 to enable).");
}

function runUpdateArtifactsCLI(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [updateScript, ...args], {
      cwd: repoRoot,
      stdio: "inherit",
      env: process.env,
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`updateArtifacts exited with status ${code}`));
      }
    });
  });
}

function artifactsAutoCapturePlugin() {
  let scheduleTrigger: ((reason: string) => void) | undefined;

  return {
    name: "alpha-evolve-artifacts-autocapture",
    apply: "serve",
    configureServer(server) {
      let running = false;
      let pendingReason: string | null = null;
      let timer: NodeJS.Timeout | null = null;

      const invoke = (reason: string) => {
        if (running) {
          pendingReason = reason;
          return;
        }
        running = true;
        server.config.logger.info(`[artifacts] Refreshing (${reason})…`);
        runUpdateArtifactsCLI(["--skip-build", "--reuse-server", "--dashboard-url", DEV_CAPTURE_URL, "--timeout", "45"])
          .catch((error) => {
            server.config.logger.error(`[artifacts] Refresh failed: ${error.message}`);
          })
          .finally(() => {
            running = false;
            if (pendingReason) {
              const next = pendingReason;
              pendingReason = null;
              schedule(next);
            }
          });
      };

      const schedule = (reason: string) => {
        if (timer) {
          clearTimeout(timer);
        }
        timer = setTimeout(() => {
          timer = null;
          invoke(reason);
        }, 600);
      };

      scheduleTrigger = schedule;

      const onListening = () => schedule("dev server start");
      server.httpServer?.once("listening", onListening);

      server.httpServer?.on("close", () => {
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
      });
    },
    handleHotUpdate(ctx) {
      if (ctx.file.includes(`${path.sep}artifacts${path.sep}`)) {
        return;
      }
      const reason = `hmr:${path.relative(ctx.server.config.root, ctx.file)}`;
      scheduleTrigger?.(reason);
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: autoCaptureEnabled ? [react(), artifactsAutoCapturePlugin()] : [react()],
  build: {
    sourcemap: true,
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/ui-meta": "http://127.0.0.1:8000"
    }
  }
});
