import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";

import {
  createModel,
  delta,
  disposeBatch,
  getFlatWeights,
  HIDDEN,
  INPUTS,
  LAYOUT,
  NUM_PARAMS,
  OUTPUTS,
  setFlatWeights,
  toBatch,
} from "./model";
import { PIXELS, type Shard } from "./shard";
import { evaluate, trainShard } from "./train";

beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});

/** A shard of `n` images of a single label, with that digit's row lit up. */
function fakeShard(n: number, label: number): Shard {
  const images = new Uint8Array(n * PIXELS);
  for (let i = 0; i < n; i++) images.fill(255, i * PIXELS + label * 28, i * PIXELS + label * 28 + 28);
  return { id: 0, images, labels: new Uint8Array(n).fill(label) };
}

describe("layout", () => {
  it("matches the Go layout: W1, b1, W2, b2 over 101,770 parameters", () => {
    expect(LAYOUT.map((t) => t.name)).toEqual(["W1", "b1", "W2", "b2"]);
    expect(LAYOUT.map((t) => t.offset)).toEqual([0, INPUTS * HIDDEN, INPUTS * HIDDEN + HIDDEN, 101760]);
    expect(NUM_PARAMS).toBe(INPUTS * HIDDEN + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS);
    expect(NUM_PARAMS).toBe(101770);
  });

  it("builds a model whose weights are in that order and shape", () => {
    const model = createModel(); // createModel calls assertLayout
    expect(model.getWeights().map((w) => w.shape)).toEqual([[INPUTS, HIDDEN], [HIDDEN], [HIDDEN, OUTPUTS], [OUTPUTS]]);
    expect(model.countParams()).toBe(NUM_PARAMS);
    model.dispose();
  });
});

describe("flat weights", () => {
  it("round-trips through the model", async () => {
    const model = createModel();
    const flat = new Float32Array(NUM_PARAMS);
    for (let i = 0; i < flat.length; i++) flat[i] = (i % 17) * 0.01 - 0.08;
    setFlatWeights(model, flat);
    const out = await getFlatWeights(model);
    expect(out.length).toBe(NUM_PARAMS);
    expect(Array.from(out.subarray(0, 8))).toEqual(Array.from(flat.subarray(0, 8)));
    // Every tensor boundary, where an off-by-one offset would show up.
    for (const t of LAYOUT) {
      expect(out[t.offset]).toBeCloseTo(flat[t.offset], 6);
      expect(out[t.offset + t.size - 1]).toBeCloseTo(flat[t.offset + t.size - 1], 6);
    }
    model.dispose();
  });

  it("does not keep a reference to the array it was given", async () => {
    const model = createModel();
    const flat = new Float32Array(NUM_PARAMS).fill(0.25);
    setFlatWeights(model, flat);
    flat.fill(99); // reusing the buffer must not change the model
    expect((await getFlatWeights(model))[0]).toBeCloseTo(0.25, 6);
    model.dispose();
  });

  it("rejects a vector of the wrong length", () => {
    const model = createModel();
    expect(() => setFlatWeights(model, new Float32Array(NUM_PARAMS - 1))).toThrow(/want 101770/);
    model.dispose();
  });

  it("leaks no tensors across a set and get", async () => {
    const model = createModel();
    const flat = new Float32Array(NUM_PARAMS).fill(0.1);
    const before = tf.memory().numTensors;
    setFlatWeights(model, flat);
    await getFlatWeights(model);
    expect(tf.memory().numTensors).toBe(before);
    model.dispose();
  });
});

describe("delta", () => {
  it("is local minus base", () => {
    const got = delta(new Float32Array([1, 2, 3]), new Float32Array([0.5, 2, -1]));
    expect(Array.from(got)).toEqual([0.5, 0, 4]);
  });

  it("rejects mismatched lengths", () => {
    expect(() => delta(new Float32Array(2), new Float32Array(3))).toThrow(/cannot subtract/);
  });
});

describe("toBatch", () => {
  it("scales pixels to 0-1 and one-hots the labels", async () => {
    const batch = toBatch(fakeShard(2, 3));
    expect(batch.xs.shape).toEqual([2, PIXELS]);
    expect(batch.ys.shape).toEqual([2, OUTPUTS]);
    const xs = await batch.xs.data();
    expect(Math.max(...xs)).toBeCloseTo(1, 6);
    expect(Array.from(await batch.ys.data()).slice(0, OUTPUTS)).toEqual([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
    disposeBatch(batch);
  });
});

describe("trainShard", () => {
  it("takes ceil(n / batchSize) steps and reports its sample count", async () => {
    const model = createModel();
    const result = await trainShard(model, fakeShard(50, 1), { batchSize: 32, learningRate: 0.05 });
    expect(result.samples).toBe(50);
    expect(result.steps).toBe(2);
    expect(Number.isFinite(result.loss)).toBe(true);
    model.dispose();
  });

  it("changes the weights and leaks no tensors", async () => {
    const model = createModel();
    const before = await getFlatWeights(model);
    const tensorsBefore = tf.memory().numTensors;
    await trainShard(model, fakeShard(64, 7), { batchSize: 32, learningRate: 0.05 });
    const after = await getFlatWeights(model);
    expect(delta(after, before).some((d) => d !== 0)).toBe(true);
    expect(tf.memory().numTensors).toBe(tensorsBefore);
    model.dispose();
  });

  it("learns a separable shard, so the training path is wired up correctly", async () => {
    const model = createModel();
    const shard: Shard = {
      id: 0,
      images: new Uint8Array(64 * PIXELS),
      labels: new Uint8Array(64),
    };
    // Half the images are blank with label 0, half are lit with label 1.
    for (let i = 32; i < 64; i++) {
      shard.images.fill(255, i * PIXELS, (i + 1) * PIXELS);
      shard.labels[i] = 1;
    }
    const test = toBatch(shard);
    const first = await trainShard(model, shard, { batchSize: 32, learningRate: 0.5 });
    for (let i = 0; i < 20; i++) await trainShard(model, shard, { batchSize: 32, learningRate: 0.5 });
    const after = await evaluate(model, test, 64);
    expect(after.loss).toBeLessThan(first.loss);
    expect(after.accuracy).toBeGreaterThan(0.9);
    disposeBatch(test);
    model.dispose();
  }, 30_000);

  it("adds the chaos harness's per-step delay", async () => {
    const model = createModel();
    const slow = await trainShard(model, fakeShard(64, 2), { batchSize: 32, learningRate: 0.05, slowMs: 40 });
    expect(slow.steps).toBe(2);
    expect(slow.ms).toBeGreaterThanOrEqual(70); // two steps, 40 ms each
    model.dispose();
  });
});

describe("evaluate", () => {
  it("reports loss and accuracy over the batch", async () => {
    const model = createModel();
    const batch = toBatch(fakeShard(16, 4));
    const result = await evaluate(model, batch, 16);
    expect(result.samples).toBe(16);
    expect(result.accuracy).toBeGreaterThanOrEqual(0);
    expect(result.accuracy).toBeLessThanOrEqual(1);
    expect(Number.isFinite(result.loss)).toBe(true);
    disposeBatch(batch);
    model.dispose();
  });

  it("leaves no tensors behind", async () => {
    const model = createModel();
    const batch = toBatch(fakeShard(8, 0));
    const before = tf.memory().numTensors;
    await evaluate(model, batch, 8);
    expect(tf.memory().numTensors).toBe(before);
    disposeBatch(batch);
    model.dispose();
  });
});
