// Stateless worker: holds nothing the coordinator cannot rebuild, so killing the
// tab at any point loses at most one shard of work.
//
// Milestone 2 runs the tab on its own: it walks every shard in order, trains one
// pass over each, and evaluates on the test set after each epoch. Milestone 3
// replaces the loop with assignments from the coordinator; the training and
// evaluation code underneath it does not change.
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";

import { createModel, DEFAULT_TRAIN_CONFIG, disposeBatch, NUM_PARAMS, toBatch } from "./model";
import { ShardCache, TEST_SHARD_ID } from "./shard";
import { evaluate, trainShard } from "./train";

const status = document.getElementById("status")!;
const logEl = document.getElementById("log");

const params = new URLSearchParams(location.search);
const label = params.get("worker") ?? "tab";
const slowMs = Number(params.get("slow") ?? 0); // chaos harness only: delay per training step
const epochs = Number(params.get("epochs") ?? 5);
const shardCount = Number(params.get("shards") ?? 100);

function log(line: string): void {
  const stamped = `${new Date().toLocaleTimeString()}  ${line}`;
  console.log(stamped);
  if (logEl) logEl.textContent = `${stamped}\n${logEl.textContent ?? ""}`;
}

async function initBackend(): Promise<string> {
  if (await tf.setBackend("webgpu").catch(() => false)) return "webgpu";
  await tf.setBackend("webgl");
  return "webgl";
}

async function main() {
  const backend = await initBackend();
  await tf.ready();
  status.textContent = `worker ${label} on ${backend}: loading MNIST…`;
  log(`worker ${label} on ${backend}, ${NUM_PARAMS} parameters, slow=${slowMs}ms`);

  const shards = new ShardCache();
  const model = createModel(DEFAULT_TRAIN_CONFIG);
  const test = toBatch(await shards.get(TEST_SHARD_ID));
  log(`test set: ${test.n} images`);

  const before = await evaluate(model, test);
  log(`epoch 0 (untrained): accuracy ${pct(before.accuracy)}, loss ${before.loss.toFixed(4)}`);

  const started = performance.now();
  let accuracy = before.accuracy;
  try {
    for (let epoch = 1; epoch <= epochs; epoch++) {
      let samples = 0;
      for (let id = 0; id < shardCount; id++) {
        const shard = await shards.get(id);
        const result = await trainShard(model, shard, { ...DEFAULT_TRAIN_CONFIG, slowMs });
        samples += result.samples;
        status.textContent =
          `worker ${label} on ${backend}\n` +
          `epoch ${epoch}/${epochs}, shard ${id + 1}/${shardCount}, ${samples} images this epoch\n` +
          `last shard: ${result.steps} steps, loss ${result.loss.toFixed(4)}, ${result.ms.toFixed(0)} ms\n` +
          `tensors ${tf.memory().numTensors}, shard cache ${(shards.bytes() / 1e6).toFixed(1)} MB`;
        // Yield to the event loop so the page stays responsive between shards.
        await new Promise((r) => setTimeout(r, 0));
      }
      const result = await evaluate(model, test);
      accuracy = result.accuracy;
      const loss = result.loss;
      const elapsed = (performance.now() - started) / 1000;
      log(
        `epoch ${epoch}/${epochs}: accuracy ${pct(accuracy)}, loss ${loss.toFixed(4)}, ` +
          `${samples} images, ${elapsed.toFixed(1)} s elapsed, ${tf.memory().numTensors} tensors`,
      );
    }
  } finally {
    disposeBatch(test);
  }

  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  status.textContent =
    `worker ${label} on ${backend}: done\n` +
    `final test accuracy ${pct(accuracy)} after ${epochs} epochs in ${seconds} s`;
  log(`done: ${pct(accuracy)} after ${epochs} epochs in ${seconds} s`);

  // TODO(milestone 3): replace the loop above with the coordinator protocol:
  //   connect to /ws/worker, hello/welcome, heartbeats every 2 s, handle
  //   assign/revoke, send result frames, reconnect with backoff 1/2/4 s capped at 10 s.
}

const pct = (x: number) => `${(100 * x).toFixed(2)}%`;

main().catch((err) => {
  status.textContent = `worker ${label} failed: ${err instanceof Error ? err.message : String(err)}`;
  console.error(err);
});
