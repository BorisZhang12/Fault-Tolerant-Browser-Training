// Binary frame format, mirrored from coordinator/internal/protocol/frame.go.
// All fields little-endian. Header: kind(1) version(4) leaseId(16) samples(4).

export const KIND_WEIGHTS = 1;
export const KIND_RESULT = 2;
export const HEADER_SIZE = 25;

export interface Frame {
  kind: number;
  version: number;
  leaseId: Uint8Array; // 16 bytes
  sampleCount: number;
  values: Float32Array;
}

export function encode(f: Frame): ArrayBuffer {
  const buf = new ArrayBuffer(HEADER_SIZE + 4 * f.values.length);
  const dv = new DataView(buf);
  dv.setUint8(0, f.kind);
  dv.setUint32(1, f.version, true);
  new Uint8Array(buf, 5, 16).set(f.leaseId.subarray(0, 16));
  dv.setUint32(21, f.sampleCount, true);
  for (let i = 0; i < f.values.length; i++) dv.setFloat32(HEADER_SIZE + 4 * i, f.values[i], true);
  return buf;
}

export function decode(buf: ArrayBuffer, wantValues: number): Frame {
  if (buf.byteLength < HEADER_SIZE) throw new Error("frame shorter than header");
  if (buf.byteLength !== HEADER_SIZE + 4 * wantValues) throw new Error("frame has wrong length");
  const dv = new DataView(buf);
  const values = new Float32Array(wantValues);
  for (let i = 0; i < wantValues; i++) values[i] = dv.getFloat32(HEADER_SIZE + 4 * i, true);
  return {
    kind: dv.getUint8(0),
    version: dv.getUint32(1, true),
    leaseId: new Uint8Array(buf.slice(5, 21)),
    sampleCount: dv.getUint32(21, true),
    values,
  };
}
