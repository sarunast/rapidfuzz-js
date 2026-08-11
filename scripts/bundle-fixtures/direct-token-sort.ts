import { tokenSortSimilarity } from 'rapidfuzz-js/fuzz'

export const run = (a: string, b: string): number => tokenSortSimilarity(a, b)
