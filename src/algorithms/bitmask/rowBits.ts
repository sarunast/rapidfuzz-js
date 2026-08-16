export function rowBitSet(
  rows: Int32Array,
  words: number,
  row: number,
  pos: number,
): boolean {
  return (rows[row * words + (pos >>> 5)] & (1 << (pos & 31))) !== 0
}

export function shiftedRowBitSet(
  rows: Int32Array,
  stride: number,
  row: number,
  offset: number,
  pos: number,
): boolean {
  const relative = pos - offset
  if (relative < 0) return false

  // By word rather than by bit: `stride << 5` is an int32 shift, so a stride
  // past 2^26 would wrap and answer for a band that does contain the position.
  const word = relative >>> 5
  if (word >= stride) return false
  return (rows[row * stride + word] & (1 << (relative & 31))) !== 0
}
