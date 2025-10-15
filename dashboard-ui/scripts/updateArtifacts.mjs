import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(here), "..", "..");
const iterationScript = path.join(repoRoot, "scripts", "dev", "run_iteration.py");

function commandExists(command) {
  try {
    const result = spawnSync(command, ["--version"], { stdio: "ignore" });
    return result.error === undefined;
  } catch (error) {
    return false;
  }
}

function resolvePythonInvoker() {
  const preferred = process.env.AE_PYTHON;
  if (preferred) {
    if (preferred === "uv") {
      return { command: "uv", prefix: ["run"] };
    }
    return { command: preferred, prefix: [] };
  }

  if (commandExists("uv")) {
    return { command: "uv", prefix: ["run"] };
  }

  return { command: "python3", prefix: [] };
}

function parseArgs(argv) {
  const options = {
    skipBuild: false,
    reuseServer: false,
    dashboardUrl: undefined,
    host: undefined,
    port: undefined,
    timeout: undefined,
    extraEnv: [],
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--skip-build":
        options.skipBuild = true;
        break;
      case "--reuse-server":
        options.reuseServer = true;
        break;
      case "--dashboard-url":
        i += 1;
        options.dashboardUrl = argv[i];
        break;
      case "--host":
        i += 1;
        options.host = argv[i];
        break;
      case "--port":
        i += 1;
        options.port = argv[i];
        break;
      case "--timeout":
        i += 1;
        options.timeout = argv[i];
        break;
      case "--env":
        i += 1;
        if (argv[i]) {
          options.extraEnv.push(argv[i]);
        }
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function buildPythonArgs(options) {
  const args = [iterationScript];
  if (options.host) {
    args.push("--host", options.host);
  }
  if (options.port) {
    args.push("--port", String(options.port));
  }
  if (options.dashboardUrl) {
    args.push("--dashboard-url", options.dashboardUrl);
  }
  if (options.skipBuild) {
    args.push("--skip-build");
  }
  if (options.reuseServer) {
    args.push("--reuse-server");
  }
  if (options.timeout) {
    args.push("--timeout", String(options.timeout));
  }
  for (const entry of options.extraEnv) {
    args.push("--env", entry);
  }
  return args;
}

export function runUpdateArtifacts(options = {}) {
  if (process.env.AE_SKIP_AUTOCAPTURE) {
    console.log("[update:artifacts] Skipping artefact refresh (AE_SKIP_AUTOCAPTURE is set).");
    return Promise.resolve();
  }

  const { command, prefix } = resolvePythonInvoker();
  const pythonArgs = buildPythonArgs(options);
  const finalArgs = [...prefix, ...pythonArgs];

  return new Promise((resolve, reject) => {
    const child = spawn(command, finalArgs, {
      cwd: repoRoot,
      stdio: "inherit",
      env: process.env,
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`run_iteration.py exited with status ${code}`));
      }
    });
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  let options;
  try {
    options = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error("[update:artifacts] Failed to parse arguments:", error.message);
    process.exit(1);
  }
  runUpdateArtifacts(options).catch((error) => {
    console.error("[update:artifacts] Artefact refresh failed:", error);
    process.exit(1);
  });
}
