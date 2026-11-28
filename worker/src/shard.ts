// MNIST shards as served by the coordinator, mirrored from
// coordinator/internal/data/mnist.go. All header fields little-endian.
//
//   bytes      field
//   4          count (uint32): images in this shard
//   4          pixels (uint32): pixels per image, always 784
//   4          shard (uint32): shard ID, or 0xFFFFFFFF for the test set
//   count*784  image bytes (uint8, row-major, 0-255)
//   count      labels (uint8, 0-9)

export const PIXELS = 784;
export const SHARD_HEADER_SIZE = 12;
export const TEST_SHARD_ID = 0xffffffff;

export interface Shard {
  /** Shard ID, or TEST_SHARD_ID for the test set. */
  id: number;
  /** count * PIXELS raw pixel bytes; image i starts at i * PIXELS. */
  images: Uint8Array;
  /** One label per image, 0-9. */
  labels: Uint8Array;
}

export function decodeShard(buf: ArrayBuffer): Shard {
  if (buf.byteLength < SHARD_HEADER_SIZE) throw new Error("shard shorter than header");
  const dv = new DataView(buf);
  const count = dv.getUint32(0, true);
  const pixels = dv.getUint32(4, true);
  const id = dv.getUint32(8, true);
  if (pixels !== PIXELS) throw new Error(`shard says ${pixels} pixels per image, want ${PIXELS}`);
  const want = SHARD_HEADER_SIZE + count * PIXELS + count;
  if (buf.byteLength !== want) {
    throw new Error(`shard is ${buf.byteLength} bytes, want ${want} for ${count} images`);
  }
  return {
    id,
    images: new Uint8Array(buf, SHARD_HEADER_SIZE, count * PIXELS),
    labels: new Uint8Array(buf, SHARD_HEADER_SIZE + count * PIXELS, count),
  };
}

/** Number of images in a shard. */
export function shardSize(s: Shard): number {
  return s.labels.length;
}

/**
 * Fetches shards from the coordinator and keeps the raw bytes in memory, so a
 * shard assigned twice is only downloaded once. Raw bytes, not tensors: 100
 * shards are 47 MB as uint8 but 188 MB as float32.
 */
export class ShardCache {
  private readonly cache = new Map<number, Shard>();
  private readonly inflight = new Map<number, Promise<Shard>>();

  constructor(private readonly baseUrl = "") {}

  /** Fetches shard `id`, or the test set when id is TEST_SHARD_ID. */
  async get(id: number): Promise<Shard> {
    const cached = this.cache.get(id);
    if (cached) return cached;
    // Two assignments for the same shard must share one request.
    const pending = this.inflight.get(id);
    if (pending) return pending;

    const path = id === TEST_SHARD_ID ? "test" : String(id);
    const load = (async () => {
      const res = await fetch(`${this.baseUrl}/shards/${path}`);
      if (!res.ok) throw new Error(`GET /shards/${path}: ${res.status} ${res.statusText}`);
      const shard = decodeShard(await res.arrayBuffer());
      if (shard.id !== id) throw new Error(`asked for shard ${id}, got ${shard.id}`);
      this.cache.set(id, shard);
      return shard;
    })();
    this.inflight.set(id, load);
    try {
      return await load;
    } finally {
      this.inflight.delete(id);
    }
  }

  /** Total bytes held, for the memory line on the worker page. */
  bytes(): number {
    let n = 0;
    for (const s of this.cache.values()) n += s.images.byteLength + s.labels.byteLength;
    return n;
  }
}
