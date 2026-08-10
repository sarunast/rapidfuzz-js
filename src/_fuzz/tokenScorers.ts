/**
 * The six token scorers: token-sort, token-set and their combination, each in a
 * whole-input and a partial-match form.
 *
 * ## The two halves
 *
 * Each scorer comes in two: a public `*_impl` that validates its inputs and
 * converts them, and a `*Converted` core over the converted sequences. The
 * composite scorers — `wRatio`, `tokenRatio`, `partialTokenRatio` — call the
 * cores, so one `wRatio` of two strings expands them into code points once
 * rather than once per component scorer. That expansion was the single largest
 * cost in `process.extract`, whose default scorer is `wRatio`.
 *
 * ## Why all six live together
 *
 * They compose heavily. `tokenRatioConverted` runs the token-set core first and
 * returns early on a perfect score before it will sort anything; the partial
 * family shares `difference`, `intersects` and the sorted/joined forms. Splitting
 * them one-per-file would put a module boundary through the middle of that
 * control flow and buy nothing.
 *
 * `normDistance` and `indelDist` are here rather than anywhere shared because
 * `tokenSetRatioConverted` is their only caller.
 */
import { asSequence, isNone } from '../_common.js'
import type { PatternMask } from '../distance/_bitVector/index.js'
import { lcsSeqLengthRange } from '../distance/lcsSeq.js'
import {
  type CharSet,
  convertProcessedPair,
  indelNormSimHeld,
  partialAlignmentConverted,
  partialRatioConverted,
  ratioConverted,
} from './basic.js'
import {
  difference,
  intersects,
  joinTokens,
  sortedOf,
  sortTokens,
  splitOf,
  tokenViewOf,
  uniqueOf,
  type PreparedTokenChoice,
} from './tokens.js'
import type { FuzzInput, FuzzOptions } from './types.js'

/** `100 - 100 * dist / lensum`, gated on `scoreCutoff`. Port of `_norm_distance`. */
function normDistance(dist: number, lensum: number, scoreCutoff: number): number {
  const score = 100 - (100 * dist) / lensum
  return score >= scoreCutoff ? score : 0
}

/**
 * Indel distance, exact whenever it comes out at or below `budget`.
 *
 * Callers only ever use the result when it is within their own cutoff, so a
 * pair further apart than that may come back overstated.
 */
function indelDist(a: ArrayLike<unknown>, b: ArrayLike<unknown>, budget: number): number {
  return (
    a.length + b.length - 2 * lcsSeqLengthRange(a, 0, a.length, b, 0, b.length, budget)
  )
}

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
  // 0.95, and that is the common case in `extractOne` rather than a corner one.
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
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = convertProcessedPair(asSequence(s1), asSequence(s2), options.processor)

  // Through the core rather than straight to `ratioConverted`, so an impossible
  // cutoff is refused before either input is sorted. The processor has already
  // run, so nothing observable is skipped by answering early.
  return tokenSortRatioConverted(a, b, options.scoreCutoff ?? 0)
}

/** Compares the inputs by their common and differing words, using `ratio`. */
export function tokenSetRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = convertProcessedPair(asSequence(s1), asSequence(s2), options.processor)

  return tokenSetRatioConverted(a, b, options.scoreCutoff ?? 0)
}

export function tokenSetRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
): number {
  // Resolved after the guard rather than as default parameter values, which
  // JavaScript evaluates ahead of the body. See `tokenSortRatioConverted`.
  if (scoreCutoff > 100) return 0

  const tokensA = uniqueOf(viewA ?? tokenViewOf(a))
  const tokensB = uniqueOf(viewB ?? tokenViewOf(b))

  // FuzzyWuzzy returns 0 here; kept for compatibility. See RapidFuzz issue 110.
  if (tokensA.size === 0 || tokensB.size === 0) return 0

  // One pass over each side rather than one per set operation: walking `tokensA`
  // decides the shared tokens and `diffAb` together.
  //
  // The shared ones are counted, not collected. Nothing below reads a token out
  // of the intersection — only how many there are and how long they are joined —
  // so the array that used to hold them, and the traversal that measured it,
  // were both pure overhead.
  let sectCount = 0
  let sectPayload = 0
  const diffAb: unknown[][] = []
  const diffBa: unknown[][] = []

  for (const [key, token] of tokensA.packed) {
    if (tokensB.packed.has(key)) {
      sectCount++
      sectPayload += token.length
    } else diffAb.push(token)
  }
  for (const [key, bucket] of tokensA.mixed) {
    for (const token of bucket) {
      if (tokensB.has(key, token)) {
        sectCount++
        sectPayload += token.length
      } else diffAb.push(token)
    }
  }

  // `a` is wholly contained in `b`: every one of its tokens was found, and it
  // has at least one. What `b` holds beyond them cannot change that, so the
  // second walk never happens — the common shape of a search where every query
  // token appears somewhere in the candidate. Containment of the token *set*,
  // not a prefix: `'react senior'` against `'zurich senior typescript react'`
  // takes this path too.
  if (sectCount !== 0 && diffAb.length === 0) return 100

  for (const [key, token] of tokensB.packed) {
    if (!tokensA.packed.has(key)) diffBa.push(token)
  }
  for (const [key, bucket] of tokensB.mixed) {
    for (const token of bucket) {
      if (!tokensA.has(key, token)) diffBa.push(token)
    }
  }

  // Containment the other way round.
  if (sectCount !== 0 && diffBa.length === 0) return 100

  const diffAbJoined = joinTokens(sortTokens(diffAb))
  const diffBaJoined = joinTokens(sortTokens(diffBa))

  const abLen = diffAbJoined.length
  const baLen = diffBaJoined.length
  // What `joinedLength` would have returned: one separator between each pair.
  const sectLen = sectCount === 0 ? 0 : sectPayload + sectCount - 1

  const sectAbLen = sectLen + (sectLen !== 0 ? 1 : 0) + abLen
  const sectBaLen = sectLen + (sectLen !== 0 ? 1 : 0) + baLen

  let result = 0
  const cutoffDistance = Math.ceil((sectAbLen + sectBaLen) * (1 - scoreCutoff / 100))
  // The distance is discarded above `cutoffDistance`, so the kernel is free to
  // stop being exact there.
  const dist = indelDist(diffAbJoined, diffBaJoined, cutoffDistance)

  if (dist <= cutoffDistance) {
    result = normDistance(dist, sectAbLen + sectBaLen, scoreCutoff)
  }

  // The remaining ratios are 0 when there is no common section.
  if (!sectLen) return result

  // Only `sect` is shared, so these distances follow from the length difference.
  const sectAbRatio = normDistance(1 + abLen, sectLen + sectAbLen, scoreCutoff)
  const sectBaRatio = normDistance(1 + baLen, sectLen + sectBaLen, scoreCutoff)

  return Math.max(result, sectAbRatio, sectBaRatio)
}

export function tokenRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
  // A thunk, not the masks: JavaScript evaluates arguments before the callee
  // runs, so a caller passing the built masks would build them even on the
  // token-set-perfect path below, which is the one that returns without ever
  // scoring a token-sort.
  preparedSortedPatternA?: () => PatternMask,
): number {
  // The whole combination is unreachable above 100 — token-set answers 0 at its
  // own guard and token-sort at its length ceiling — so this returns before any
  // splitting, hashing, sorting or joining. `wRatio` asks for exactly this
  // whenever its base ratio clears 95.
  if (scoreCutoff > 100) return 0

  // The views carry the memo, so the sorted forms below are built only if the
  // token-set score leaves something to beat — and, when they are, a caller that
  // already has them supplies them rather than paying twice.
  const tokensViewA = viewA ?? tokenViewOf(a)
  const tokensViewB = viewB ?? tokenViewOf(b)

  const setScore = tokenSetRatioConverted(a, b, scoreCutoff, tokensViewA, tokensViewB)

  // Only the larger of the two is reported, so token-sort matters solely if it
  // can beat what token-set already scored — which lets its own cutoff be
  // raised to that score. Both scorers return either 0 or a value at their
  // cutoff, so the two Math.max branches come out the same as before: a
  // token-sort score the higher cutoff rejects is one this Math.max discards.
  //
  // A perfect token-set score leaves nothing to beat at all, and returning here
  // skips sorting and joining both token lists as well as the LCS run.
  if (setScore === 100) return 100

  return Math.max(
    setScore,
    tokenSortRatioConverted(
      a,
      b,
      Math.max(scoreCutoff, setScore),
      tokensViewA,
      tokensViewB,
      // The masks describe the *caller's* sorted query. A view built here has a
      // sorted form of its own, and scoring it against the caller's masks would
      // compare against the wrong sequence.
      viewA === undefined ? undefined : preparedSortedPatternA?.(),
    ),
  )
}

/** The larger of `tokenSetRatio` and `tokenSortRatio`. */
export function tokenRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = convertProcessedPair(asSequence(s1), asSequence(s2), options.processor)

  return tokenRatioConverted(a, b, options.scoreCutoff ?? 0)
}

/** Sorts the words in both inputs, then takes the `partialRatio`. */
export function partialTokenSortRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = convertProcessedPair(asSequence(s1), asSequence(s2), options.processor)

  // Refused before either input is sorted, for the same reason as
  // `tokenSortRatio_impl`. `partialRatioConverted` would answer 0 anyway, but
  // only after both sorted forms had been built.
  const scoreCutoff = options.scoreCutoff ?? 0
  if (scoreCutoff > 100) return 0

  return partialRatioConverted(
    sortedOf(tokenViewOf(a)),
    sortedOf(tokenViewOf(b)),
    scoreCutoff,
  )
}

/** Compares the inputs by their differing words, using `partialRatio`. */
export function partialTokenSetRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = convertProcessedPair(asSequence(s1), asSequence(s2), options.processor)

  return partialTokenSetRatioConverted(a, b, options.scoreCutoff ?? 0)
}

export function partialTokenSetRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
): number {
  // Resolved after the guard rather than as default parameter values. See
  // `tokenSortRatioConverted`.
  if (scoreCutoff > 100) return 0

  const tokensA = uniqueOf(viewA ?? tokenViewOf(a))
  const tokensB = uniqueOf(viewB ?? tokenViewOf(b))

  // FuzzyWuzzy returns 0 here; kept for compatibility. See RapidFuzz issue 110.
  if (tokensA.size === 0 || tokensB.size === 0) return 0

  // Any shared word already makes this a perfect partial match.
  if (intersects(tokensA, tokensB)) return 100

  return partialRatioConverted(
    joinTokens(sortTokens(difference(tokensA, tokensB))),
    joinTokens(sortTokens(difference(tokensB, tokensA))),
    scoreCutoff,
  )
}

/** The larger of `partialTokenSetRatio` and `partialTokenSortRatio`. */
export function partialTokenRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = convertProcessedPair(asSequence(s1), asSequence(s2), options.processor)

  return partialTokenRatioConverted(a, b, options.scoreCutoff ?? 0)
}

export function partialTokenRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
  // Thunks, and only for the first of the two comparisons below. That one scores
  // the whole sorted query against the whole sorted candidate, so its left side
  // is the same sequence for every candidate and its masks are worth holding.
  // The second scores the *differences*, which depend on the candidate and so
  // can never be prepared. Thunks rather than values because the shared-word
  // shortcut above returns before either is wanted.
  preparedSortedPatternA?: () => PatternMask,
  preparedSortedCharSetA?: () => CharSet,
): number {
  // The most expensive of these guards to have had behind its defaults: a split,
  // a dedupe, an outer-array copy, a sort and a join on the `a` side alone, all
  // to answer 0. They are resolved below instead, which JavaScript only reaches
  // once the guard has passed. See `tokenSortRatioConverted`.
  if (scoreCutoff > 100) return 0

  const tokensViewA = viewA ?? tokenViewOf(a)
  const tokensViewB = viewB ?? tokenViewOf(b)
  const tokensA = uniqueOf(tokensViewA)
  const tokensB = uniqueOf(tokensViewB)

  // Any shared word already makes this a perfect partial match.
  //
  // Sorting and joining happens below this rather than in a default parameter
  // above it, so the shared-word answer no longer pays for a form it discards.
  if (intersects(tokensA, tokensB)) return 100

  const diffAb = difference(tokensA, tokensB)
  const diffBa = difference(tokensB, tokensA)

  // `partialRatioConverted` is this call with the last two arguments left off,
  // so a caller with nothing prepared gets exactly what it got before.
  //
  // The needle is decided here rather than left to the callee, because the
  // thunks are arguments: JavaScript would run them on the way in, building the
  // sorted query's masks and pruning set even on the calls that go on to ignore
  // them. `partialAlignmentConverted` keeps the caller's state only while the
  // first argument is the shorter side, and `<=` is its test — equal lengths
  // scan `a` first, so the prepared state still applies.
  const sortedA = sortedOf(tokensViewA)
  const sortedB = sortedOf(tokensViewB)
  const preparedApplies = viewA !== undefined && sortedA.length <= sortedB.length

  const result =
    partialAlignmentConverted(
      sortedA,
      sortedB,
      scoreCutoff,
      true,
      preparedApplies ? preparedSortedPatternA?.() : undefined,
      preparedApplies ? preparedSortedCharSetA?.() : undefined,
    )?.score ?? 0

  // Nothing is shared, so the second partialRatio would repeat the first.
  const splitA = splitOf(tokensViewA)
  const splitB = splitOf(tokensViewB)
  if (splitA.length === diffAb.length && splitB.length === diffBa.length) return result

  return Math.max(
    result,
    partialRatioConverted(
      joinTokens(sortTokens(diffAb)),
      joinTokens(sortTokens(diffBa)),
      Math.max(scoreCutoff, result),
    ),
  )
}
