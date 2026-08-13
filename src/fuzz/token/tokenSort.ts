import type { PatternMask } from '../../algorithms/shared/bitmask/pattern.js'
import { asSequence, isMissing } from '../../algorithms/shared/scorerSupport.js'
import { indelNormSimHeld, ratioConverted } from '../partialWindow.js'
import type { FuzzInput, FuzzOptions } from '../types.js'
import {
  canonicalLengthOf,
  sortedOf,
  tokenPair,
  tokenViewOf,
  type PreparedTokenChoice,
} from './tokens.js'

/**
 * `sortedPatternA` must build the masks of `sortedA` — the caller holding one
 * is the only way to know that, since a mask cannot be checked against the
 * sequence it came from. It is a thunk for the same reason
 * `tokenRatioConverted` takes one: arguments are evaluated before the callee
 * runs, so a caller passing the built masks would build them behind the cutoff
 * guard below rather than in front of it.
 *
 * It is scored through {@link indelNormSimHeld} rather than `ratioHeld`
 * because the sorted forms are built here and their lengths are already in
 * hand; either way the arithmetic is
 * `indelNormSimRange`'s with the LCS kernel swapped, which is what keeps a
 * prepared query from disagreeing with an unprepared one in the last ULP.
 */
export function tokenSortRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
  sortedPatternA?: () => PatternMask,
): number {
  // Before the sorted forms are built, not after. These used to be default
  // parameter values, which JavaScript evaluates ahead of the body — so the
  // guard sat behind the very work it exists to skip. `wRatio` reaches a cutoff
  // above 100 whenever its base ratio clears 95, by dividing the running best by
  // 0.95, and that is the common case in best-match search rather than a corner one.
  if (scoreCutoff > 100) return 0

  const choiceA = viewA ?? tokenViewOf(a)
  const choiceB = viewB ?? tokenViewOf(b)

  // The Indel length ceiling, applied before the sorted forms exist rather than
  // inside the kernel that receives them. Sorting and joining cannot change how
  // long the canonical sequence is, so the ceiling is decidable off the raw
  // inputs — and a pair it rejects is one that never needed splitting, sorting
  // or joining at all. Best-match search raises its cutoff to the running best,
  // which is what makes this the common answer rather than a corner one.
  //
  // The expression and the comparison are `indelNormSimRange`'s, character for
  // character, including the division of the cutoff: a lookalike predicate would
  // part company with it in the last ULP exactly when a cutoff lands on a score.
  //
  // Behind `scoreCutoff > 0`, because the ceiling is never negative and so can
  // never fall below a cutoff of zero — measuring the lengths to discover that
  // is a scan of both inputs that cannot repay itself, and unguarded it cost
  // 1.17x on a cutoff-free scan. A `NaN` cutoff fails this test and reaches the
  // kernel, which is where it answered before.
  if (scoreCutoff > 0) {
    const lengthA = canonicalLengthOf(choiceA)
    const lengthB = canonicalLengthOf(choiceB)
    const maximum = lengthA + lengthB
    // Two empty canonical forms are identical, and the ceiling below would
    // divide by zero. `indelNormSimRange` weighs its 1 against the cutoff here;
    // this cannot, because the guard above has already refused everything over
    // 100 and nothing at or below it rejects a perfect score.
    if (maximum === 0) return 100
    if (1 - Math.abs(lengthA - lengthB) / maximum < scoreCutoff / 100) return 0
  }

  const sortedA = sortedOf(choiceA)
  const sortedB = sortedOf(choiceB)

  if (sortedPatternA !== undefined) {
    return (
      indelNormSimHeld(
        sortedPatternA(),
        sortedA.length,
        sortedB,
        0,
        sortedB.length,
        scoreCutoff / 100,
      ) * 100
    )
  }

  return ratioConverted(sortedA, sortedB, scoreCutoff)
}

/** Sorts the words in both inputs, then takes the `ratio`. */
export function tokenSortRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = tokenPair(asSequence(s1), asSequence(s2))

  // Through the core rather than straight to `ratioConverted`, so an impossible
  // cutoff is refused before either input is sorted.
  return tokenSortRatioConverted(a, b, options.scoreCutoff ?? 0)
}
