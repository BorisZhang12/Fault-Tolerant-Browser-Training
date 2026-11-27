import { describe, expect, it, vi } from "vitest";

import { decodeShard, PIXELS, SHARD_HEADER_SIZE, ShardCache, shardSize, TEST_SHARD_ID } from "./shard";

/** Builds a shard the way coordinator/internal/data.EncodeShard does. */
function encodeShard(id: number, count: number, pixel = 0, label = 0): ArrayBuffer {
  const buf = new ArrayBuffer(SHARD_HEADER_SIZE + count * PIXELS + count);
  const dv = new DataView(buf);
  dv.setUint32(0, count, true);
  dv.setUint32(4, PIXELS, true);
  dv.setUint32(8, id, true);
  new Uint8Array(buf, SHARD_HEADER_SIZE, count * PIXELS).fill(pixel);
  new Uint8Array(buf, SHARD_HEADER_SIZE + count * PIXELS, count).fill(label);
  return buf;
}

describe("decodeShard", () => {
  it("reads the header and splits images from labels", () => {
    const s = decodeShard(encodeShard(7, 3, 42, 5));
    expect(s.id).toBe(7);
    expect(shardSize(s)).toBe(3);
    expect(s.images.length).toBe(3 * PIXELS);
    expect(s.images[0]).toBe(42);
    expect(Array.from(s.labels)).toEqual([5, 5, 5]);
  });

  it("reads the test set's shard ID", () => {
    expect(decodeShard(encodeShard(TEST_SHARD_ID, 1)).id).toBe(TEST_SHARD_ID);
  });

  it("rejects a frame shorter than the header", () => {
    expect(() => decodeShard(new ArrayBuffer(8))).toThrow(/shorter than header/);
  });

  it("rejects the wrong pixel count", () => {
    const buf = encodeShard(0, 1);
    new DataView(buf).setUint32(4, 783, true);
    expect(() => decodeShard(buf)).toThrow(/783 pixels/);
  });

  it("rejects a truncated body", () => {
    expect(() => decodeShard(encodeShard(0, 2).slice(0, SHARD_HEADER_SIZE + PIXELS))).toThrow(/bytes, want/);
  });
});

describe("ShardCache", () => {
  function stubFetch(buf: (path: string) => ArrayBuffer) {
    const fetchMock = vi.fn(async (url: string) => ({
      ok: true,
      status: 200,
      statusText: "OK",
      arrayBuffer: async () => buf(url),
    }));
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  it("fetches a shard once and serves it from memory after that", async () => {
    const fetchMock = stubFetch(() => encodeShard(3, 2));
    const cache = new ShardCache();
    const first = await cache.get(3);
    const second = await cache.get(3);
    expect(second).toBe(first);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith("/shards/3");
    expect(cache.bytes()).toBe(2 * PIXELS + 2);
  });

  it("asks for the test set by name", async () => {
    const fetchMock = stubFetch(() => encodeShard(TEST_SHARD_ID, 1));
    await new ShardCache().get(TEST_SHARD_ID);
    expect(fetchMock).toHaveBeenCalledWith("/shards/test");
  });

  it("shares one request between concurrent callers", async () => {
    const fetchMock = stubFetch(() => encodeShard(5, 1));
    const cache = new ShardCache();
    const [a, b] = await Promise.all([cache.get(5), cache.get(5)]);
    expect(a).toBe(b);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects a shard whose ID does not match the request", async () => {
    stubFetch(() => encodeShard(9, 1));
    await expect(new ShardCache().get(4)).rejects.toThrow(/asked for shard 4, got 9/);
  });

  it("retries after a failed fetch instead of caching the failure", async () => {
    const fetchMock = vi.fn(async () => ({ ok: false, status: 404, statusText: "Not Found" }));
    vi.stubGlobal("fetch", fetchMock);
    const cache = new ShardCache();
    await expect(cache.get(1)).rejects.toThrow(/404/);
    await expect(cache.get(1)).rejects.toThrow(/404/);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
