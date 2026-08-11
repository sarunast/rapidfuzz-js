/**
 * Shared bit-parallel scoring kernels: Hyyrö's LCS and Myers' Levenshtein, over 32-bit
 * words.
 *
 * These replace the O(|s1| * |s2|) dynamic programs with O(|s1| * |s2| / 32),
 * which is where nearly all of this library's time goes — `fuzz.ratio`,
 * `WRatio` and therefore `process.extract` all bottom out in the LCS kernel.
 *
 * Distinct from `_bitParallel.ts`, which computes the same recurrences but
 * keeps every intermediate row, because `editops` has to walk them back. Here
 * only the final vector is needed, so a row can live in the shared scratch
 * buffers instead of in a matrix.
 *
 * ## Why 32 bits and not 64
 *
 * JavaScript's bitwise operators coerce to int32. `BigInt` has no such limit but
 * is roughly an order of magnitude slower, so both files spell out the carries
 * between words by hand rather than paying for it.
 *
 * ## What is where
 *
 * This file is only the entry point. The kernels live under `_bitVector/`, one
 * module per algorithm family plus the state they share:
 *
 * - `shared.ts` — the mask table, the scratch buffers and the affix trimming,
 *   and the canonical definitions of the word constants. Read its header before
 *   changing anything the kernels hoist.
 * - `pattern.ts` — the immutable per-query {@link PatternMask}. Imports nothing.
 * - `lcs.ts`, `levenshtein.ts`, `osa.ts` — one algorithm family each, kernels
 *   and dispatcher together.
 *
 * Each kernel module declares its own copies of the constants it reads inside a
 * loop; `shared.ts` explains why, and is the definition they must agree with.
 */

import { resetOsaScratch } from './osa.js'
import { resetBitVectorScratch } from './shared.js'

export {
  lcsLength,
  lcsLengthPrepared,
  lcsLengthPreparedBounded,
  lcsLengthRange,
} from './lcs.js'
export {
  levenshteinPrepared,
  levenshteinPreparedRow,
  levenshteinSmallBand,
  levenshteinUniform,
} from './levenshtein.js'
export {
  osaManyWords,
  osaOneWord,
  osaOneWordPrepared,
  osaOneWordRange,
  osaPrepared,
} from './osa.js'
export { type PatternMask, preparePattern } from './pattern.js'
export { UNBOUNDED_MISSES, WORD_LIMIT } from './shared.js'

/**
 * Drop every shared buffer and put the direct lookup table back at Latin-1.
 *
 * For the benchmarks, which need each case to start where the last one started.
 * The buffers grow on demand and never shrink, and `directLimit` in particular
 * grows permanently, so without this a case measured after a sixteen-thousand-
 * element pair never pays for the allocation that pair paid — and its number
 * depends on what ran before it rather than on its own work.
 *
 * Composed rather than re-exported: `osa.ts` owns its own scratch, so resetting
 * everything means resetting both. Correctness does not depend on it — every
 * buffer is refilled before it is read, so dropping one only costs the next call
 * an allocation. Nothing in `src` calls it, and no entry point re-exports it.
 */
export function resetSharedScratch(): void {
  resetBitVectorScratch()
  resetOsaScratch()
}
