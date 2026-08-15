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
  if (relative < 0 || relative >= stride << 5) return false
  return (rows[row * stride + (relative >>> 5)] & (1 << (relative & 31))) !== 0
}
