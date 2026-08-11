import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from '../shared/editops/index.js'
import { conv, type EditopsOptions, type Sequence } from '../shared/scorerSupport.js'
import { alignHirschberg } from './internal/alignment.js'
import { distance_, UNIFORM } from './internal/engine.js'

/**
 * Options for {@link levenshteinEditops}, which alone among the metrics that
 * expose edit operations takes a hint — the others have no band to narrow.
 */
export interface LevenshteinEditopsOptions extends EditopsOptions {
  /**
   * Estimate of the distance, which buys a narrower alignment band.
   *
   * It cannot change the *length* of the edit script. It can change which
   * optimal script comes back, but only for inputs large enough to need the
   * recursive path: a hint that pays for itself replaces the assumed distance
   * with the real one, that shrinks the matrix the dispatch is sizing, and a
   * pair that then fits takes the exact matrix instead of being split. Both
   * answers are optimal; only one of them is upstream's.
   */
  scoreHint?: number | undefined
}

/**
 * Edit operations that turn `s1` into `s2`.
 *
 * The alignment follows Hyyrö's bit-parallel algorithm, and over a matrix it
 * recovers the operations upstream produces, one for one. Uniform weights only,
 * as upstream.
 *
 * Upstream always builds that matrix, at a bit per cell. This does too while it
 * fits in {@link ALIGNMENT_MATRIX_LIMIT}, and splits the alignment in half
 * recursively when it does not — an alignment of two 16k inputs is 33MB of
 * matrix. Past that point the operations are still an optimal edit script of
 * the same length, but not always upstream's choice among the optimal ones.
 *
 * That is not a split rule waiting to be fixed. Upstream's recovery walks the
 * whole matrix backwards, preferring a deletion at each step, and which script
 * that yields is a property of the whole matrix rather than of any one cell the
 * path passes through. Splitting at exactly the column where upstream's path
 * crosses the middle row is not enough: measured over large random pairs, of
 * the cases that disagree, more than a third split on upstream's own path and
 * still recovered a different script, because each half is then recovered from
 * its own matrix, whose row differences are not the ones upstream read.
 */
export function levenshteinEditops(
  s1: Sequence,
  s2: Sequence,
  options: LevenshteinEditopsOptions = {},
): Editops {
  const [full1, full2] = conv(s1, s2, options.processor)
  const ops: Editop[] = []
  let maximum = Math.max(full1.length, full2.length)

  // A hint buys a narrower alignment band, but only by finding the distance
  // first — so the alignment is computed twice over. Upstream takes that trade
  // only when the hint promises to more than halve the second pass, and this
  // follows it: without a hint nothing extra runs.
  const hint =
    options.scoreHint == null ? null : Math.max(31, Math.floor(options.scoreHint))
  if (hint !== null && 2 * hint < maximum) {
    maximum = distance_(full1, full2, UNIFORM, maximum, hint)
  }

  alignHirschberg(ops, full1, 0, full1.length, full2, 0, full2.length, maximum)
  return editopsFromValidated(ops, full1.length, full2.length)
}

/** {@link levenshteinEditops} expressed as blocks. */
export function levenshteinOpcodes(
  s1: Sequence,
  s2: Sequence,
  options: LevenshteinEditopsOptions = {},
): Opcodes {
  return levenshteinEditops(s1, s2, options).toOpcodes()
}

export { levenshteinEditops as editops, levenshteinOpcodes as opcodes }
