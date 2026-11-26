// Stateless worker: holds nothing the coordinator cannot rebuild, so killing the
// tab at any point loses at most one shard of work.
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";

const status = document.getElementById("status")!;
const params = new URLSearchParams(location.search);
const label = params.get("worker") ?? "tab";
const slowMs = Number(params.get("slow") ?? 0); // chaos harness only: delay per training step

async function initBackend(): Promise<string> {
  if (await tf.setBackend("webgpu").catch(() => false)) return "webgpu";
  await tf.setBackend("webgl");
  return "webgl";
}

async function main() {
  const backend = await initBackend();
  status.textContent = `worker ${label} on ${backend} (slow=${slowMs}ms)`;
  // TODO(milestone 2): single-tab MNIST training (784 -> 128 -> 10, batch 32, SGD lr 0.05).
  // TODO(milestone 3): connect to /ws/worker, hello/welcome, heartbeats every 2 s,
  //   handle assign/revoke, send result frames, reconnect with backoff 1/2/4 s capped at 10 s.
}

main();
