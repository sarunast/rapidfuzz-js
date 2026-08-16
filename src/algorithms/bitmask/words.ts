export const WORD_BITS = 32
export const WORD_SHIFT = 5
export const WORD_MASK = 31

export function wordCount(length: number): number {
  return Math.ceil(length / WORD_BITS)
}
