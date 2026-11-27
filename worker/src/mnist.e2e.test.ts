// End-to-end check of the single-tab training path (milestone 2), against a
// running coordinator: Go encodes the shards, TypeScript decodes them, trains on
// them, and evaluates on the real test set. It proves the data path, the weight
// layout and the hyperparameters, but not the WebGPU backend — this runs on the
// CPU backend in Node, so it is slower per step than a real tab.
//
// Skipped unless a coordinator is running:
//
//	./coordinator/bin/coordinator -addr :8080 -run-id m2   # in another shell
//	cd worker && COORDINATOR_URL=http://localhost:8080 npm test
//
// Environment: E2E_EPOCHS (default 5), E2E_SHARDS (default 100, all of MNIST),
// E2E_MIN_ACCURACY (default 0.95, the milestone's done-line).
import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";

import { createModel, DEFAULT_TRAIN_CONFIG, disposeBatch, toBatch } from "./model";
import { ShardCache, TEST_SHARD_ID } from "./shard";
import { evaluate, trainShard } from "./train";

const url = process.env.COORDINATOR_URL;
const epochs = Number(process.env.E2E_EPOCHS ?? 5);
const shardCount = Number(process.env.E2E_SHARDS ?? 100);
const minAccuracy = Number(process.env.E2E_MIN_ACCURACY ?? 0.95);

describe.skipIf(!url)("single-tab MNIST training", () => {
  beforeAll(async () => {
    await tf.setBackend("cpu");
    await tf.ready();
  });

  it(
    `reaches ${minAccuracy * 100}% test accuracy over ${epochs} epochs of ${shardCount} shards`,
    async () => {
      const shards = new ShardCache(url);
      const model = createModel(DEFAULT_TRAIN_CONFIG);
      const test = toBatch(await shards.get(TEST_SHARD_ID));
      expect(test.n).toBe(10_000);

      const started = Date.now();
      let accuracy = 0;
      try {
        for (let epoch = 1; epoch <= epochs; epoch++) {
          for (let id = 0; id < shardCount; id++) {
            const shard = await shards.get(id);
            expect(shard.labels.length).toBe(600);
            await trainShard(model, shard, DEFAULT_TRAIN_CONFIG);
          }
          const result = await evaluate(model, test);
          accuracy = result.accuracy;
          console.log(
            `epoch ${epoch}/${epochs}: accuracy ${(100 * accuracy).toFixed(2)}%, ` +
              `loss ${result.loss.toFixed(4)}, ${((Date.now() - started) / 1000).toFixed(1)} s, ` +
              `${tf.memory().numTensors} tensors`,
          );
        }
      } finally {
        disposeBatch(test);
        model.dispose();
      }
      expect(accuracy).toBeGreaterThanOrEqual(minAccuracy);
    },
    30 * 60_000,
  );
});
