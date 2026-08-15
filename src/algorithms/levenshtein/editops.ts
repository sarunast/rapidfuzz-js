import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from '../../core/editops/index.js'
import { convPair } from '../../core/sequence.js'
import type { Sequence } from '../../core/types.js'
import { alignHirschberg } from './internal/alignment.js'

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
export function levenshteinEditops(s1: Sequence, s2: Sequence): Editops {
  const [full1, full2] = convPair(s1, s2)
  const ops: Editop[] = []
  const maximum = Math.max(full1.length, full2.length)

  alignHirschberg(ops, full1, 0, full1.length, full2, 0, full2.length, maximum)
  return editopsFromValidated(ops, full1.length, full2.length)
}

/**
 * {@link levenshteinEditops} as contiguous ranges rather than single operations.
 *
 * Opcodes cover the whole of both inputs, including the `equal` stretches
 * between edits, which is usually what a diff view or a highlighter wants —
 * `editops` lists only the changes. The two convert into each other with
 * `toEditops()` and `toOpcodes()`.
 */
export function levenshteinOpcodes(s1: Sequence, s2: Sequence): Opcodes {
  return levenshteinEditops(s1, s2).toOpcodes()
}

export { levenshteinEditops as editops, levenshteinOpcodes as opcodes }
