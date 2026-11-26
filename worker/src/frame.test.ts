import { describe, expect, it } from "vitest";
import { decode, encode, KIND_RESULT } from "./frame";

describe("frame", () => {
  it("round-trips", () => {
    const leaseId = new Uint8Array(16);
    leaseId[0] = 0xab;
    const f = { kind: KIND_RESULT, version: 7, leaseId, sampleCount: 600, values: new Float32Array([1.5, -2, 0, 3.25]) };
    const out = decode(encode(f), 4);
    expect(out.version).toBe(7);
    expect(out.sampleCount).toBe(600);
    expect(out.leaseId[0]).toBe(0xab);
    expect(Array.from(out.values)).toEqual([1.5, -2, 0, 3.25]);
  });
});
// TODO(milestone 3): decode a frame written by the Go encoder (shared fixture) to prove cross-language compatibility.
