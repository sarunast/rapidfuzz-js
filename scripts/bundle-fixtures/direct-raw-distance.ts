import { distance } from 'rapidfuzz-js/levenshtein'

export const run = (a: string, b: string): number => distance(a, b)
