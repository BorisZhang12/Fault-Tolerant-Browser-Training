// The model and the flat weight vector, mirrored from
// coordinator/internal/model/layout.go: 784 -> 128 (ReLU) -> 10 (softmax),
// 101,770 parameters in the fixed wire order W1, b1, W2, b2.
import * as tf from "@tensorflow/tfjs";

import { PIXELS, type Shard, shardSize } from "./shard";

export const INPUTS = PIXELS;
export const HIDDEN = 128;
export const OUTPUTS = 10;

export interface TensorSpec {
  name: string;
  shape: number[];
  offset: number;
  size: number;
}

/** Tensors in their wire order, with each one's offset into the flat vector. */
export const LAYOUT: TensorSpec[] = (() => {
  const specs: Array<[string, number[]]> = [
    ["W1", [INPUTS, HIDDEN]],
    ["b1", [HIDDEN]],
    ["W2", [HIDDEN, OUTPUTS]],
    ["b2", [OUTPUTS]],
  ];
  let offset = 0;
  return specs.map(([name, shape]) => {
    const size = shape.reduce((a, b) => a * b, 1);
    const spec = { name, shape, offset, size };
    offset += size;
    return spec;
  });
})();

/** Length of the flat weight vector (101,770). */
export const NUM_PARAMS = LAYOUT[LAYOUT.length - 1].offset + LAYOUT[LAYOUT.length - 1].size;

export interface TrainConfig {
  batchSize: number;
  learningRate: number;
}

export const DEFAULT_TRAIN_CONFIG: TrainConfig = { batchSize: 32, learningRate: 0.05 };

/**
 * Builds the model once per tab. Its layers produce weights in LAYOUT order,
 * since a tfjs dense layer reports [kernel, bias] and the kernel of the first
 * layer is [784, 128].
 */
export function createModel(cfg: TrainConfig = DEFAULT_TRAIN_CONFIG): tf.Sequential {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [INPUTS], units: HIDDEN, activation: "relu" }),
      tf.layers.dense({ units: OUTPUTS, activation: "softmax" }),
    ],
  });
  model.compile({
    optimizer: tf.train.sgd(cfg.learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  assertLayout(model);
  return model;
}

/** Fails loudly if a tfjs version ever changes the order or shape of weights. */
export function assertLayout(model: tf.LayersModel): void {
  const weights = model.getWeights();
  if (weights.length !== LAYOUT.length) {
    throw new Error(`model has ${weights.length} weight tensors, want ${LAYOUT.length}`);
  }
  LAYOUT.forEach((spec, i) => {
    const got = weights[i].shape;
    if (got.length !== spec.shape.length || got.some((d, j) => d !== spec.shape[j])) {
      throw new Error(`weight ${i} (${spec.name}) has shape [${got}], want [${spec.shape}]`);
    }
  });
}

/** Reads the model's weights into one flat Float32Array in wire order. */
export async function getFlatWeights(model: tf.LayersModel): Promise<Float32Array> {
  const weights = model.getWeights();
  const parts = await Promise.all(weights.map((t) => t.data<"float32">()));
  const flat = new Float32Array(NUM_PARAMS);
  LAYOUT.forEach((spec, i) => flat.set(parts[i], spec.offset));
  return flat;
}

/** Loads a flat weight vector into the model, replacing its current weights. */
export function setFlatWeights(model: tf.LayersModel, flat: Float32Array): void {
  if (flat.length !== NUM_PARAMS) {
    throw new Error(`got ${flat.length} parameters, want ${NUM_PARAMS}`);
  }
  // slice, not subarray: on the CPU and WebGL backends tf.tensor keeps the array
  // it is handed, which would make the model alias the caller's buffer. The
  // worker computes its delta against the base weights it still holds, so that
  // aliasing would make every delta zero.
  const tensors = LAYOUT.map((spec) =>
    tf.tensor(flat.slice(spec.offset, spec.offset + spec.size), spec.shape, "float32"),
  );
  try {
    model.setWeights(tensors);
  } finally {
    tensors.forEach((t) => t.dispose());
  }
}

/** The update a worker sends back: local weights minus the base version's. */
export function delta(local: Float32Array, base: Float32Array): Float32Array {
  if (local.length !== base.length) {
    throw new Error(`cannot subtract ${base.length} parameters from ${local.length}`);
  }
  const out = new Float32Array(local.length);
  for (let i = 0; i < local.length; i++) out[i] = local[i] - base[i];
  return out;
}

export interface Batch {
  /** [n, 784] pixels scaled to 0-1. */
  xs: tf.Tensor2D;
  /** [n, 10] one-hot labels. */
  ys: tf.Tensor2D;
  n: number;
}

/**
 * Turns a shard's raw bytes into tensors. The caller owns them and must call
 * disposeBatch; one shard is 1.9 MB as float32, the test set 31 MB.
 */
export function toBatch(shard: Shard): Batch {
  const n = shardSize(shard);
  const [xs, ys] = tf.tidy(() => [
    tf.tensor2d(shard.images, [n, PIXELS], "float32").div<tf.Tensor2D>(255),
    tf.oneHot(tf.tensor1d(shard.labels, "int32"), OUTPUTS).toFloat() as tf.Tensor2D,
  ]);
  return { xs, ys, n };
}

export function disposeBatch(batch: Batch): void {
  batch.xs.dispose();
  batch.ys.dispose();
}
