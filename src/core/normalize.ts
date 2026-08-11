import type { Sequence } from './types.js'

const NON_ALNUM = /[^\p{L}\p{N}_]/gu

export function normalizeText(value: Sequence): string {
  if (typeof value !== 'string') throw new TypeError('normalizeText expects a string')
  return value.replace(NON_ALNUM, ' ').trim().toLowerCase()
}
