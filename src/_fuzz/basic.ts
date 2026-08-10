/**
 * `ratio` and `partialRatio` — the two scorers that bottom out directly in
 * normalised Indel similarity, plus the processor plumbing every other family
 * builds on.
 *
 * ## This module must not tokenise
 *
 * `basic.ts` and `tokens.ts` are siblings, not a chain: nothing here may import
 * the token engine. `ratio` and `partialRatio` are the lower-level subsystem,
 * and keeping them usable without loading or splitting anything is what makes
 * the dependency graph readable — `tokenScorers.ts` and `composite.ts` sit above
 * both, and a cycle would show up here first.
 *
 * The four helpers exported for those upper layers — {@link indelNormSimHeld},
 * {@link applyProcessor}, {@link convertProcessedPair} and
 * {@link ratioConverted} — live here rather than in a shared `common` module
 * precisely because this one is already upstream of everything that wants them.
 */
import {
  asSequence,
  conv,
  convSequence,
  isNone,
  isSequence,
  type Processor,
  type Sequence,
} from '../_common.js'
import type { PatternMask } from '../distance/_bitVector/index.js'
import {
  lcsSeqLengthPrepared,
  lcsSeqLengthRange,
  prepareLcsPattern,
} from '../distance/lcsSeq.js'
import type { FuzzInput, FuzzOptions, ScoreAlignment } from './types.js'

function indelNormSim(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  return indelNormSimRange(a, 0, a.length, b, 0, b.length, scoreCutoff)
}

/**
 * Normalised Indel similarity of two ranges, so a caller comparing many
 * windows of one sequence — `partialRatio` — never has to copy a window out.
 */
function indelNormSimRange(
  a: ArrayLike<unknown>,
  start1: number,
  len1: number,
  b: ArrayLike<unknown>,
  start2: number,
  len2: number,
  scoreCutoff: number,
): number {
  const maximum = len1 + len2
  if (maximum === 0) return 1 >= scoreCutoff ? 1 : 0

  // A common subsequence cannot be longer than the shorter input, so the indel
  // distance can never fall below the length difference — which makes this an
  // exact ceiling on the score, reachable without touching the LCS kernel.
  // `extractOne` raises its cutoff to the running best, so once a good match is
  // found this rejects every candidate too differently sized to beat it.
  //
  // The ceiling is scored through the same expression and the same comparison
  // as the real result, which is what keeps the prune from ever disagreeing
  // with it: a lookalike predicate would part company in the last ULP exactly
  // when a cutoff lands on a score.

  const ceiling = 1 - Math.abs(len1 - len2) / maximum
  if (ceiling < scoreCutoff) return 0

  // Largest number of unmatched elements that could still score at the cutoff.
  // `sim` below is `1 - misses / maximum`, so the bound is `maximum * (1 -
  // cutoff)`; the extra 1 covers the rounding of that product, and overshooting
  // is free — a budget too large only costs the kernel its shortcut, while one
  // too small could reject a pair that exactly meets the cutoff. Without a
  // cutoff every miss is allowed, which needs no arithmetic to work out.
  const budget =
    scoreCutoff > 0 ? Math.floor(maximum * (1 - scoreCutoff)) + 1 : maximum + 1
  const lcs = lcsSeqLengthRange(a, start1, len1, b, start2, len2, budget)

  const sim = 1 - (maximum - 2 * lcs) / maximum
  return sim >= scoreCutoff ? sim : 0
}

/**
 * Normalised Indel similarity of the held pattern against a range of `text`.
 *
 * The held kernel is always exact, so unlike {@link indelNormSimRange} there is
 * no miss budget to derive — the length-difference ceiling is the only prune.
 */
export function indelNormSimHeld(
  pattern: PatternMask,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  scoreCutoff: number,
): number {
  const maximum = patternLength + textLength
  if (maximum === 0) return 1 >= scoreCutoff ? 1 : 0

  const ceiling = 1 - Math.abs(patternLength - textLength) / maximum
  if (ceiling < scoreCutoff) return 0

  const lcs = lcsSeqLengthPrepared(pattern, text, textStart, textLength)
  const sim = 1 - (maximum - 2 * lcs) / maximum

  return sim >= scoreCutoff ? sim : 0
}

/**
 * Run the processor, and hold its return to the type it declares.
 *
 * The annotation is `unknown` although {@link Processor} says `Sequence`,
 * because the function on the other side of this call is the caller's and may
 * be plain JavaScript. What made the check worth its cost is the shape of the
 * failure: `convSequence` reads a `length` off whatever it is handed, and
 * `new Array(undefined)` is `[undefined]` — so a processor returning a number
 * or an object turned two unrelated inputs into the same one-element sequence
 * and scored them 100. Upstream raises for every one of those returns.
 *
 * The scorers that reach {@link conv} instead are checked there; this is the
 * same boundary for the ones that process and convert in two steps.
 */
export function applyProcessor(
  s: Sequence,
  processor: Processor | null | undefined,
): Sequence {
  if (processor == null) return s

  const processed: unknown = processor(s)
  if (!isSequence(processed)) {
    throw new TypeError('processor must return a string or an array-like sequence')
  }

  return processed
}

export function convertProcessedPair(
  s1: Sequence,
  s2: Sequence,
  processor: Processor | null | undefined,
): [ArrayLike<unknown>, ArrayLike<unknown>] {
  return [
    convSequence(applyProcessor(s1, processor)),
    convSequence(applyProcessor(s2, processor)),
  ]
}

/** Normalised Indel similarity as a percentage, over converted inputs. */
export function ratioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  return indelNormSim(a, b, scoreCutoff / 100) * 100
}

/**
 * Normalised Indel similarity as a percentage.
 *
 * @example
 * ratio('this is a test', 'this is a test!') // => 96.55172413793103
 */
export function ratio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  return ratioConverted(a, b, options.scoreCutoff ?? 0)
}

/**
 * The distinct elements of `s1`, which the window scan uses to reject a window
 * whose last element the needle does not hold at all.
 *
 * Split out so a caller scoring one query against many candidates can build it
 * once. Unlike a {@link PatternMask}, this cannot be shared across
 * representations: the scan compares with `===`, and `'a' !== 97`.
 *
 * `NaN` is kept, deliberately. A `Set` matches it against itself where the
 * kernels refuse to, so a window anchored on one is scanned and scores nothing —
 * dropping it here would prune that window instead. That is a *stronger* filter
 * than the one this is documented to be, and it decides which of two equal
 * windows gets reported, so it could move an alignment. Upstream has no `NaN`
 * element to check the answer against, and the input it would speed up is one no
 * caller has: the trade is a real tie-breaking risk for an unreachable win.
 */
export function charSetOf(s: ArrayLike<unknown>): ReadonlySet<unknown> {
  const set = new Set<unknown>()
  for (let i = 0; i < s.length; i++) set.add(s[i])
  return set
}

/** Port of `_partial_ratio_impl`. Assumes `s1.length <= s2.length`. */
export function partialRatioImpl(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff: number,
  prepared: PatternMask = prepareLcsPattern(s1, 0, s1.length),
  scoreOnly = false,
  preparedCharSet?: ReadonlySet<unknown>,
): ScoreAlignment {
  // `s1` is the pattern for every alignment below, so its match masks are built
  // once here instead of once per window. Nothing between the hold and the
  // release scores a different pattern or calls back into user code, which is
  // what the held masks require.
  return partialRatioScan(s1, s2, scoreCutoff, prepared, scoreOnly, preparedCharSet)
}

/**
 * Both inputs must already share an element representation — the pruning set
 * below compares them with `===`. See `alignRepresentation`.
 *
 * `preparedCharSet`, when given, must be {@link charSetOf} over this `s1` in the
 * representation it arrived in. A caller holding one across many candidates is
 * the only way to know that, exactly as with `prepared`.
 */
function partialRatioScan(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff: number,
  pattern: PatternMask,
  scoreOnly: boolean,
  preparedCharSet?: ReadonlySet<unknown>,
): ScoreAlignment {
  const len1 = s1.length
  const len2 = s2.length

  // No normalised similarity can exceed 1, so a cutoff past it rejects every
  // window — but only after each one has been scored, because the rejection
  // happens inside `indelNormSimHeld`'s ceiling test. `wRatio` asks for exactly
  // this: it divides the running best by 0.9 or 0.6 before the partial branch,
  // so a base score over 90 arrives here as a cutoff above 1.
  //
  // The result is what the scan would have returned anyway. `indelNormSimHeld`
  // answers 0 for every window, `consider` never improves on `res.score`, and
  // `res` reaches the end untouched — including `srcEnd`/`destEnd` at `len1`.
  if (scoreCutoff > 1) {
    return { score: 0, srcStart: 0, srcEnd: len1, destStart: 0, destEnd: len1 }
  }

  // An empty needle matches nothing, and every window scores the same 0 — but
  // the interior search does not know that. It sizes itself from `len2`, so
  // `partialRatio('', longText)` allocated a `Uint32Array` the length of the
  // haystack and bisected it, evaluating `1 - distance / (2 * len1)` as `0 / 0`
  // the whole way down. The `NaN` failed every comparison, so the answer came
  // out right by accident; this makes it come out right on purpose.
  if (len1 === 0) {
    return { score: 0, srcStart: 0, srcEnd: 0, destStart: 0, destEnd: 0 }
  }

  const charSet = preparedCharSet ?? charSetOf(s1)

  const res = {
    score: 0,
    srcStart: 0,
    srcEnd: len1,
    destStart: 0,
    destEnd: len1,
  }
  let cutoff = scoreCutoff

  /** Returns true once a perfect alignment is found and the search can stop. */
  const consider = (start: number, end: number): boolean => {
    // Scored as a range of `s2` against the held pattern: the window is only
    // ever read, so neither copying it out nor rebuilding `s1`'s masks — both
    // of which cost more than the scoring itself — is necessary.
    const lsRatio = indelNormSimHeld(pattern, len1, s2, start, end - start, cutoff)
    if (lsRatio <= res.score) return false

    res.score = lsRatio
    cutoff = lsRatio
    res.destStart = start
    res.destEnd = end

    if (res.score === 1) {
      res.score = 100
      return true
    }

    return false
  }

  /** Exact indel distance for a full-length window. */
  const windowDistance = (start: number): number => {
    const lcs = lcsSeqLengthPrepared(pattern, s2, start, len1)
    return 2 * (len1 - lcs)
  }

  /** Windows running off the start of `s2`, shorter than the pattern. */
  const scanPrefix = (): boolean => {
    for (let i = 1; i < len1; i++) {
      if (!charSet.has(s2[i - 1])) continue
      if (consider(0, i)) return true
    }
    return false
  }

  /** Windows running off the end of `s2`, likewise shorter. */
  const scanSuffix = (): boolean => {
    for (let i = len2 - len1; i < len2; i++) {
      if (!charSet.has(s2[i])) continue
      if (consider(i, len2)) return true
    }
    return false
  }

  const scanInterior = (): boolean => {
    const lastInterior = len2 - len1 - 1
    // Upstream bisects unconditionally. Below roughly a word of windows the
    // linear scan wins here, because it can skip a window whose last element
    // the pattern does not hold at all — a test the bisection has no place for,
    // since it needs both endpoints of a range scored to bound what lies
    // between them. Dropping the gate cost 20% at 16 windows.
    if (lastInterior >= 64) {
      // Port of upstream's divide-and-conquer full-window search. Moving a
      // window by one position can change its indel distance by at most two.
      // Distances at both endpoints therefore provide a lower bound for every
      // window between them, allowing unpromising ranges to be discarded.
      const unknown = 0xffff_ffff
      const scores = new Uint32Array(lastInterior + 1)
      scores.fill(unknown)
      // From the running best rather than the caller's cutoff: on the alignment
      // path the shorter windows have already been scanned, and whatever they
      // found is a tighter bound than the one asked for.
      let distanceToBeat =
        Math.floor(2 * len1 * (1 - cutoff) + Number.EPSILON * 2 * len1) + 1
      let windows = [0, lastInterior]
      let nextWindows: number[] = []

      while (windows.length !== 0) {
        for (let window = 0; window < windows.length; window += 2) {
          const first = windows[window]
          const last = windows[window + 1]

          if (scores[first] === unknown) {
            const distance = windowDistance(first)
            scores[first] = distance
            const score = 1 - distance / (2 * len1)
            if (distance < distanceToBeat && score >= cutoff) {
              distanceToBeat = distance
              if (consider(first, first + len1)) return true
            }
          }
          if (scores[last] === unknown) {
            const distance = windowDistance(last)
            scores[last] = distance
            const score = 1 - distance / (2 * len1)
            if (distance < distanceToBeat && score >= cutoff) {
              distanceToBeat = distance
              if (consider(last, last + len1)) return true
            }
          }

          const cellDiff = last - first
          if (cellDiff <= 1) continue

          const knownEdits = Math.abs(scores[first] - scores[last])
          const maxImprovement =
            Math.floor((cellDiff - Math.floor(knownEdits / 2)) / 2) * 2
          const minimumPossible = Math.min(scores[first], scores[last]) - maxImprovement
          if (minimumPossible < distanceToBeat) {
            const center = first + Math.floor(cellDiff / 2)
            nextWindows.push(first, center, center, last)
          }
        }

        const oldWindows = windows
        windows = nextWindows
        nextWindows = oldWindows
        nextWindows.length = 0
      }
    } else {
      for (let i = 0; i < len2 - len1; i++) {
        if (!charSet.has(s2[i + len1 - 1])) continue
        if (consider(i, i + len1)) return true
      }
    }
    return false
  }

  // The full-length windows are the strongest candidates, so scoring them first
  // leaves the cutoff high enough for the length ceiling in `indelNormSimHeld`
  // to reject most of the ~2 * len1 shorter ones without running the kernel.
  //
  // It also decides which of two equally-scoring alignments is reported, since
  // only a strictly better window replaces the one held. `partialRatio` returns
  // a score, which no order can change; `partialRatioAlignment` returns the
  // positions, and there upstream's Python order is the one to match.
  if (scoreOnly) {
    if (scanInterior() || scanPrefix() || scanSuffix()) return res
  } else if (scanPrefix() || scanInterior() || scanSuffix()) {
    return res
  }

  res.score *= 100
  return res
}

function swapAlignment(a: ScoreAlignment): ScoreAlignment {
  return {
    score: a.score,
    srcStart: a.destStart,
    srcEnd: a.destEnd,
    destStart: a.srcStart,
    destEnd: a.srcEnd,
  }
}

/**
 * Best alignment of the shorter input inside the longer one, with the
 * {@link ratio_impl} for that alignment.
 *
 * Returns `null` when either input is missing, or when the score is below
 * `scoreCutoff`.
 */
export function partialRatioAlignment(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): ScoreAlignment | null {
  if (isNone(s1) || isNone(s2)) return null

  // `conv` rather than `convertProcessedPair`: unlike the token scorers, nothing
  // below needs code points specifically — it needs the two inputs to agree,
  // which `conv` gives either way, keeping a pair of BMP strings as strings.
  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)

  return partialAlignmentConverted(a, b, options.scoreCutoff ?? 0)
}

export function partialAlignmentConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  cutoff: number,
  scoreOnly = false,
  preparedA?: PatternMask,
  preparedCharSetA?: ReadonlySet<unknown>,
): ScoreAlignment | null {
  // No alignment can score above 100, so a cutoff past it rejects every pair —
  // including two empty inputs, whose perfect score is awarded below without
  // ever reaching the comparison at the end. `wRatio` divides its cutoff by a
  // scale factor and so does ask for more than 100; every other path already
  // answered 0 by way of the length ceiling, which is why this only shows up on
  // the empty pair.
  if (cutoff > 100) return null

  let scoreCutoff = cutoff

  if (a.length === 0 && b.length === 0) {
    return { score: 100, srcStart: 0, srcEnd: 0, destStart: 0, destEnd: 0 }
  }

  const s1Shorter = a.length <= b.length
  const shorter = s1Shorter ? a : b
  const longer = s1Shorter ? b : a

  // The pattern is whichever input is shorter, so a caller's held masks only
  // serve the first scan, and only when `a` is that one. Equal lengths included:
  // upstream falls back to the uncached scorer when `a` is strictly longer, not
  // when the two match.
  //
  // The held char set rides the same gate for the same reason: both describe
  // `a`, and neither means anything once `b` is the needle.
  const aIsNeedle = shorter === a
  let res = partialRatioImpl(
    shorter,
    longer,
    scoreCutoff / 100,
    aIsNeedle ? preparedA : undefined,
    scoreOnly,
    aIsNeedle ? preparedCharSetA : undefined,
  )

  if (res.score !== 100 && a.length === b.length) {
    scoreCutoff = Math.max(scoreCutoff, res.score)
    // `b` is the needle on this pass, so nothing the caller holds applies.
    const res2 = partialRatioImpl(
      longer,
      shorter,
      scoreCutoff / 100,
      undefined,
      scoreOnly,
    )
    if (res2.score > res.score) res = swapAlignment(res2)
  }

  if (res.score < scoreCutoff) return null

  return s1Shorter ? res : swapAlignment(res)
}

export function partialRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  return partialAlignmentConverted(a, b, scoreCutoff, true)?.score ?? 0
}

/**
 * Searches for the best alignment of the shorter input in the longer one and
 * returns the {@link ratio_impl} for it.
 */
export function partialRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  // Same work as `partialRatioAlignment`, minus its obligation to report *which*
  // alignment won — which is what lets the scan visit the windows in the order
  // that prunes best. See `partialRatioScan`.
  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  return partialAlignmentConverted(a, b, options.scoreCutoff ?? 0, true)?.score ?? 0
}
