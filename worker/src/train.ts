// One pass of local SGD over one shard, plus evaluation. Everything here is
// pure compute: no coordinator, no sockets, so it can be driven by milestone 2's
// single-tab loop and by the assignment handler later.
import * as tf from "@tensorflow/tfjs";

import { type Batch, DEFAULT_TRAIN_CONFIG, disposeBatch, toBatch, type TrainConfig } from "./model";
import { type Shard, shardSize } from "./shard";

export interface TrainOptions extends TrainConfig {
  /** Chaos harness only: extra delay per training step, simulating a weak device. */
  slowMs?: number;
}

export interface TrainResult {
  /** Images trained on; the coordinator weights this worker's delta by it. */
  samples: number;
  /** Gradient steps taken (ceil(samples / batchSize)). */
  steps: number;
  /** Mean training loss over the pass. */
  loss: number;
  /** Wall-clock milliseconds, including any `slowMs` delay. */
  ms: number;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Trains the model for one pass over `shard`, starting from whatever weights it
 * currently holds. Tensors built here are disposed before returning, so a tab
 * that trains hundreds of shards does not grow.
 */
export async function trainShard(
  model: tf.LayersModel,
  shard: Shard,
  opts: TrainOptions = DEFAULT_TRAIN_CONFIG,
): Promise<TrainResult> {
  const batch = toBatch(shard);
  const started = performance.now();
  try {
    const slowMs = opts.slowMs ?? 0;
    const history = await model.fit(batch.xs, batch.ys, {
      batchSize: opts.batchSize,
      epochs: 1,
      shuffle: true,
      verbose: 0,
      callbacks: slowMs > 0 ? { onBatchEnd: async () => void (await sleep(slowMs)) } : undefined,
    });
    return {
      samples: batch.n,
      steps: Math.ceil(batch.n / opts.batchSize),
      loss: lastMetric(history, "loss"),
      ms: performance.now() - started,
    };
  } finally {
    disposeBatch(batch);
  }
}

export interface EvalResult {
  loss: number;
  accuracy: number;
  samples: number;
}

/**
 * Evaluates on a held-out batch. In the full system the coordinator's Go
 * evaluation is the source of truth, because it must not depend on a killable
 * tab; this is the single-tab readout and a cross-check on it.
 */
export async function evaluate(model: tf.LayersModel, test: Batch, batchSize = 512): Promise<EvalResult> {
  const out = model.evaluate(test.xs, test.ys, { batchSize });
  const scalars = Array.isArray(out) ? out : [out];
  try {
    const [loss, accuracy] = await Promise.all(scalars.map(async (s) => (await s.data())[0]));
    return { loss, accuracy, samples: test.n };
  } finally {
    scalars.forEach((s) => s.dispose());
  }
}

/** Reads the last value of a metric out of a tfjs History. */
function lastMetric(history: tf.History, name: string): number {
  const values = history.history[name];
  if (!values || values.length === 0) return NaN;
  const last = values[values.length - 1];
  return typeof last === "number" ? last : NaN;
}

/** Images in a shard; re-exported so callers need not import both modules. */
export { shardSize };
