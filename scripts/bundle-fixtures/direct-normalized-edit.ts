import { normalizedSimilarity } from 'rapidfuzz-js/levenshtein'

export const run = (a: string, b: string): number => normalizedSimilarity(a, b)
