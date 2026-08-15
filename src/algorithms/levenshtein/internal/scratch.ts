export const ALIGNMENT_MATRIX_LIMIT: number = 1024 * 1024
export let hirschbergLeft: Uint32Array = new Uint32Array(0)
export let hirschbergRight: Uint32Array = new Uint32Array(0)

let weightedFloatScratch: Float64Array | null = null
let weightedIntegerScratch: Int32Array | null = null

export function resetWeightedScratch(): void {
  weightedFloatScratch = null
  weightedIntegerScratch = null
  hirschbergLeft = new Uint32Array(0)
  hirschbergRight = new Uint32Array(0)
}

export function weightedFloatRow(needed: number): Float64Array {
  const held = weightedFloatScratch
  if (held !== null && held.length >= needed) return held
  let size = held === null ? 64 : held.length
  while (size < needed) size *= 2
  weightedFloatScratch = new Float64Array(size)
  return weightedFloatScratch
}

export function weightedIntegerRow(needed: number): Int32Array {
  const held = weightedIntegerScratch
  if (held !== null && held.length >= needed) return held
  let size = held === null ? 64 : held.length
  while (size < needed) size *= 2
  weightedIntegerScratch = new Int32Array(size)
  return weightedIntegerScratch
}

export function growHirschbergRows(needed: number): void {
  if (hirschbergLeft.length >= needed) return
  let size = Math.max(128, hirschbergLeft.length)
  while (size < needed) size *= 2
  hirschbergLeft = new Uint32Array(size)
  hirschbergRight = new Uint32Array(size)
}
