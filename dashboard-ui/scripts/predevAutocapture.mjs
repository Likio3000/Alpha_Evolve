#!/usr/bin/env node

import { runUpdateArtifacts } from "./updateArtifacts.mjs";

async function main() {
  if (!process.env.AE_ENABLE_AUTOCAPTURE || process.env.AE_SKIP_AUTOCAPTURE) {
    console.log("[predev] Auto capture skipped (set AE_ENABLE_AUTOCAPTURE=1 to enable).");
    return;
  }

  try {
    await runUpdateArtifacts();
  } catch (error) {
    console.error("[predev] Auto capture failed:", error);
    process.exit(1);
  }
}

main();
