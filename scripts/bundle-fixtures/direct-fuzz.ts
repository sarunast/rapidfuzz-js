import { similarity } from 'rapidfuzz-js/fuzz'

export const run = (a: string, b: string): number => similarity(a, b)
