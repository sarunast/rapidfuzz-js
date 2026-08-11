import type { PatternMask } from '../../algorithms/shared/bitmask/pattern.js'
import { asSequence, convPair, isMissing } from '../../algorithms/shared/scorerSupport.js'
import type { FuzzInput, FuzzOptions } from '../types.js'
import { indelNormSimHeld, ratioConverted } from './partialWindow.js'
import { sortedOf, tokenViewOf, type PreparedTokenChoice } from './tokens.js'

/**
 * `sortedPatternA` must be the masks of `sortedA` — the caller holding one is
 * the only way to know that, since a mask cannot be checked against the
 * sequence it came from.
 *
 * It is scored through {@link indelNormSimHeld} rather than `ratioPrepared` on
 * purpose: the two differ in where they scale by 100, and a prepared query must
 * not disagree with an unprepared one in the last ULP. `indelNormSimHeld` is
 * `indelNormSimRange` with the LCS kernel swapped, so the arithmetic is the same
 * expression either way.
 */
export function tokenSortRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
  sortedPatternA?: PatternMask,
): number {
  // Before the sorted forms are built, not after. These used to be default
  // parameter values, which JavaScript evaluates ahead of the body — so the
  // guard sat behind the very work it exists to skip. `wRatio` reaches a cutoff
  // above 100 whenever its base ratio clears 95, by dividing the running best by
  // 0.95, and that is the common case in best-match search rather than a corner one.
  if (scoreCutoff > 100) return 0

  const sortedA = sortedOf(viewA ?? tokenViewOf(a))
  const sortedB = sortedOf(viewB ?? tokenViewOf(b))

  if (sortedPatternA !== undefined) {
    return (
      indelNormSimHeld(
        sortedPatternA,
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

  const [a, b] = convPair(asSequence(s1), asSequence(s2))

  // Through the core rather than straight to `ratioConverted`, so an impossible
  // cutoff is refused before either input is sorted.
  return tokenSortRatioConverted(a, b, options.scoreCutoff ?? 0)
}
