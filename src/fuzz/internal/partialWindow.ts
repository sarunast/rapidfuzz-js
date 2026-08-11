import {
  lcsSeqLengthPrepared,
  lcsSeqLengthRange,
  prepareLcsPattern,
} from '../../algorithms/lcs/implementation.js'
import type { PatternMask } from '../../algorithms/shared/bitmask/pattern.js'
/**
 * `ratio` and `partialRatio` — the two scorers that bottom out directly in
 * normalised Indel similarity.
 *
 * ## This module must not tokenise
 *
 * `partialWindow.ts` and `tokens.ts` are siblings, not a chain: nothing here may
 * import the token engine. Basic and partial similarity are the lower-level
 * subsystem, and keeping them usable without loading or splitting anything is
 * what makes the dependency graph readable. Token families and adaptive fuzzy
 * similarity sit above both, and a cycle would show up here first.
 *
 * The two helpers exported for those upper layers — {@link indelNormSimHeld}
 * and {@link ratioConverted} — live here rather than in a shared `common` module
 * precisely because this one is already upstream of everything that wants them.
 */
import { asSequence, convPair, isMissing } from '../../algorithms/shared/scorerSupport.js'
import type { FuzzInput, FuzzOptions, ScoreAlignment } from '../types.js'

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
  // Best-match search raises its cutoff to the running best, so once a good match is
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
  // Two empty inputs are identical, and the division below would answer `NaN`.
  // Not weighed against the cutoff: every caller scales a percentage into
  // `[0, 1]` and answers 0 above 100 before reaching this, so no cutoff that
  // gets here is above the 1 it would be compared against.
  if (maximum === 0) return 1

  const ceiling = 1 - Math.abs(patternLength - textLength) / maximum
  if (ceiling < scoreCutoff) return 0

  const lcs = lcsSeqLengthPrepared(pattern, text, textStart, textLength)
  const sim = 1 - (maximum - 2 * lcs) / maximum

  return sim >= scoreCutoff ? sim : 0
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
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return ratioConverted(a, b, options.scoreCutoff ?? 0)
}

/**
 * The distinct elements of a needle, in the form the window scan probes them.
 *
 * Port of upstream's `detail::CharSet`, which is a direct-indexed table for
 * narrow characters and a hash set for the rest. The table is the point: a
 * plain `Set` has to be probed with `s2[i]`, and indexing a string allocates a
 * one-character string per probe — the whole cost for Latin-1 text, and an
 * uncached allocation above it. `charCodeAt` into a `Uint8Array` allocates
 * nothing, which measured 0.54x of the `Set` scan on ASCII and 0.09x on
 * Cyrillic.
 *
 * Unlike a {@link PatternMask} this cannot be shared across representations:
 * the scan compares with `===`, and `'a' !== 97`. What *is* shared is the
 * numbering — {@link wide} holds code units for a string needle and element
 * values otherwise, and those two agree throughout the BMP, which is as far as
 * a string-backed needle reaches.
 */
export interface CharSet {
  /**
   * Elements the table addresses directly, by code unit for a string needle or
   * by value for a number.
   *
   * 256 entries, except for a string needle that leaves Latin-1 without leaving
   * the low BMP — see {@link HIGH_TABLE_LIMIT}. Read its `length` rather than
   * assuming either size.
   */
  readonly direct: Uint8Array
  /**
   * Everything {@link direct} cannot address. Null when the needle has none,
   * which is every pure Latin-1 input and every needle whose table was widened
   * to cover its script.
   */
  readonly wide: ReadonlySet<unknown> | null
}

/**
 * Highest code unit a string needle's direct table will stretch to cover.
 *
 * Every non-ideographic script sits in one contiguous block of the low BMP:
 * Cyrillic ends at U+04FF, Greek at U+03FF, Hebrew at U+05F4, and a needle
 * written in any of them is answered by a table of a couple of kilobytes
 * however long it is. The ideographs start at U+4E00 and would want eighty, so
 * they keep the `Set` — and keep it *inline* at the probe, which is why the
 * table's size is what varies here rather than a second table being added
 * beside it. A widened table costs one comparison against a local instead of
 * against the constant 256; a second table cost a null test on every probe and
 * two more fields on every {@link CharSet}, which measured 1.02-1.03x on the
 * ASCII and Latin-1 needles that can never use one.
 */
const HIGH_TABLE_LIMIT = 2048

/**
 * Build the pruning set for `s`.
 *
 * Split out so a caller scoring one query against many candidates can build it
 * once.
 *
 * `NaN` is kept, deliberately. It fails the numeric test below and lands in
 * {@link CharSet.wide}, where `Set` matches it against itself although the
 * kernels refuse to — so a window anchored on one is scanned and scores
 * nothing. Dropping it would prune that window instead: a *stronger* filter
 * than this is documented to be, and one that decides which of two equal
 * windows gets reported, so it could move an alignment. Upstream has no `NaN`
 * element to check the answer against, and the input it would speed up is one
 * no caller has.
 */
export function charSetOf(s: ArrayLike<unknown>): CharSet {
  let wide: Set<unknown> | null = null

  // Hoisted, because a needle is one representation throughout and the test is
  // otherwise paid per element. A string yields one-character strings, which
  // `charCodeAt` reads without allocating either — and it is the one form whose
  // table is worth allocating up front, since text without a single character
  // below U+0100 is rarer than the null test to defer it would cost.
  if (typeof s === 'string') {
    const direct = new Uint8Array(256)
    // A `Set<number>` of its own rather than the one below, so that widening the
    // table can read it back without asking what it holds. A string needle's
    // high elements are code units and nothing else.
    let codes: Set<number> | null = null
    // Tracked as the loop runs rather than derived afterwards, so a needle with
    // no high code unit — the common one — never makes a second pass to be told
    // so.
    let highest = -1
    for (let i = 0; i < s.length; i++) {
      const code = s.charCodeAt(i)
      if (code < 256) direct[code] = 1
      else {
        ;(codes ??= new Set<number>()).add(code)
        if (code > highest) highest = code
      }
    }

    if (codes === null || highest >= HIGH_TABLE_LIMIT) return { direct, wide: codes }

    // Cold, and once per needle rather than once per probe: widen the table to
    // reach the script the needle is written in, and the `Set` has no reader
    // left. From zero rather than from the lowest code unit seen, because an
    // offset is a second field to carry and a subtraction on every probe, where
    // the run of zeroes below the block is at most two kilobytes once.
    const wideDirect = new Uint8Array(highest + 1)
    wideDirect.set(direct)
    for (const code of codes) wideDirect[code] = 1
    return { direct: wideDirect, wide: null }
  }

  // A sequence of objects fills no table at all, and allocating one anyway was
  // the whole of that input's cost: 337ns against a 1246ns build, which showed
  // up as 1.11x on `partialRatio` over object elements. The probe needs a table
  // either way, so an empty one stands in rather than a null to test for.
  let direct: Uint8Array | null = null
  for (let i = 0; i < s.length; i++) {
    const value = s[i]
    if (typeof value === 'number' && value >= 0 && value < 256 && (value | 0) === value) {
      ;(direct ??= new Uint8Array(256))[value] = 1
    } else {
      ;(wide ??= new Set<unknown>()).add(value)
    }
  }

  return { direct: direct ?? emptyTable(), wide }
}

/**
 * The all-zero table a needle with nothing to put in one shares.
 *
 * Retained rather than allocated per call, and built on demand rather than at
 * module scope: `sideEffects: false` promises that importing this module does
 * no work. Never written — {@link charSetOf} allocates its own the moment it
 * has a narrow element to record.
 */
let shared: Uint8Array | null = null
function emptyTable(): Uint8Array {
  return (shared ??= new Uint8Array(256))
}

/**
 * Scratch for the window bisection, which is sized on the haystack but reads
 * almost none of it.
 *
 * The search visits O(log) of its windows; the array indexing them is
 * `len2 - len1` long. So the `fill` that marked every entry unknown cost more
 * than the search it prepared for, and it was paid per candidate. A generation
 * stamp answers "has this window been scored" without writing to a cell the
 * search never reaches, which drops both the fill and the allocation.
 *
 * Re-entrancy is what makes module scratch safe here, and the scan has none:
 * input validation and every conversion run before it, and from the moment it
 * starts it reads only strings, converted sequences, masks and its own sets.
 * Nothing it calls can re-enter it.
 *
 * Held only up to {@link BISECTION_SCRATCH_LIMIT} windows. Above that a single
 * comparison against a huge haystack would keep megabytes reachable for as long
 * as nothing else was scored, which is the trade `MASK_PATTERN_LIMIT` draws for
 * the same reason.
 */
const BISECTION_SCRATCH_LIMIT = 1 << 16
let bisectionScores: Uint32Array | null = null
let bisectionStamps: Int32Array | null = null
let bisectionWindows: number[] = []
let bisectionNextWindows: number[] = []
let bisectionGeneration = 0

/**
 * A stamp is an `Int32Array` cell, so the counter cannot exceed what one holds.
 * Reaching the ceiling takes two billion bisections in a single process, which
 * no test is going to sit through — {@link resetPartialRatioScratch} takes a
 * starting generation so the wrap can be driven directly instead.
 */
const BISECTION_GENERATION_LIMIT = 0x7fff_ffff

/** Stamps start at 0 and generations at 1, so a stale slot can never match. */
function nextBisectionGeneration(): number {
  bisectionGeneration++
  if (bisectionGeneration >= BISECTION_GENERATION_LIMIT) {
    // Dropped rather than cleared. A fresh buffer is zeroes, which is the state
    // a clearing fill would have written, and dropping needs no test for a
    // buffer that may not have been built yet — the wrap is reachable on the
    // very first bisection after a reset that starts the counter near it.
    bisectionStamps = null
    bisectionGeneration = 1
  }

  return bisectionGeneration
}

/**
 * The two buffers, grown to `size`.
 *
 * Grown independently, which is safe in the one direction that matters: a fresh
 * stamp buffer is all zeroes and so matches no live generation, leaving every
 * window unscored. That is the conservative answer, and the only one a caller
 * could act on wrongly is the opposite.
 */
function bisectionScoresFor(size: number): Uint32Array {
  const held = bisectionScores
  if (held !== null && held.length >= size) return held

  // Sized from the floor rather than from what is already there: growth is
  // rare, and starting the doubling at the existing length only spells the same
  // answer with a branch on a buffer that may not exist.
  let next = 256
  while (next < size) next *= 2
  bisectionScores = new Uint32Array(next)
  return bisectionScores
}

function bisectionStampsFor(size: number): Int32Array {
  const held = bisectionStamps
  if (held !== null && held.length >= size) return held

  let next = 256
  while (next < size) next *= 2
  bisectionStamps = new Int32Array(next)
  return bisectionStamps
}

/**
 * Drop the bisection's scratch.
 *
 * Correctness does not depend on it: a dropped buffer only costs the next
 * bisection an allocation, and a dropped generation is answered by stamps that
 * match nothing. `startGeneration` is the exception, and the reason this takes
 * an argument at all — the wrap at {@link BISECTION_GENERATION_LIMIT} is two
 * billion bisections away, so starting near it is what lets that path run.
 */
export function resetPartialRatioScratch(startGeneration = 0): void {
  bisectionScores = null
  bisectionStamps = null
  bisectionWindows = []
  bisectionNextWindows = []
  bisectionGeneration = startGeneration
}

/**
 * Shortest window of `s2` that a needle of `len1` could still score at `cutoff`.
 *
 * A window of `m` elements has `maximum = len1 + m` and at most `m` elements in
 * common, so its similarity cannot exceed `2m / (len1 + m)` — the same ceiling
 * {@link indelNormSimHeld} computes and rejects on, reached without calling it.
 * Rearranged, a window is worth visiting when `m >= cutoff * len1 / (2 - cutoff)`.
 * `cutoff` is at most 1 wherever this is called, so the divisor is at least 1.
 *
 * Deliberately **floored** rather than rounded up to the exact threshold, and
 * that is what makes it safe to use as a loop bound: flooring can only answer
 * below the true minimum, never above, so the scans may visit a window the
 * kernel goes on to reject but can never skip one it would have scored. The
 * exact test stays where it was, in the one place that also produces the score,
 * and no second floating-point convention is introduced for the same question.
 * A window of no elements is not a window, hence the lower clamp; there is no
 * upper one, because `cutoff` of 1 already lands exactly on `len1`.
 *
 * Module scope, not a closure over the scan: it is called once per scan and once
 * per improving window, and two more closures per call is a cost the shortest
 * scans could feel.
 */
function minimumWindow(len1: number, cutoff: number): number {
  const estimate = Math.floor((cutoff * len1) / (2 - cutoff))
  return estimate < 1 ? 1 : estimate
}

/** Port of `_partial_ratio_impl`. Assumes `s1.length <= s2.length`. */
export function partialRatioImpl(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff: number,
  prepared: PatternMask = prepareLcsPattern(s1, 0, s1.length),
  scoreOnly = false,
  preparedCharSet?: CharSet,
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
  preparedCharSet?: CharSet,
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
  const direct = charSet.direct
  // Read once: a string needle's table is 256 entries or as wide as its script
  // needs, and the probe below has to compare against whichever it got.
  const narrow = direct.length
  const wide = charSet.wide

  // The needle and the text always share a representation — `convPair` returns a
  // pair, and `alignRepresentation` is applied to both sides — so this settles
  // which store `charSetOf` filled, once per comparison rather than per window.
  const text = typeof s2 === 'string' ? s2 : null

  /**
   * Whether the needle holds the element of `s2` at `index`.
   *
   * One closure rather than one per representation, although the test above is
   * then repeated per probe. Splitting it gives the three scans below two
   * function identities to call, which costs them their inline caches: measured
   * 1.17x against this on a sequence of objects, where the split was supposed to
   * help most. The branch itself is loop-invariant and predicted.
   */
  const holds = (index: number): boolean => {
    if (text === null) {
      const value = s2[index]
      if (
        typeof value === 'number' &&
        value >= 0 &&
        value < 256 &&
        (value | 0) === value
      ) {
        return direct[value] !== 0
      }
      return wide !== null && wide.has(value)
    }

    const code = text.charCodeAt(index)
    return code < narrow ? direct[code] !== 0 : wide !== null && wide.has(code)
  }

  const res = {
    score: 0,
    srcStart: 0,
    srcEnd: len1,
    destStart: 0,
    destEnd: len1,
  }
  let cutoff = scoreCutoff

  /**
   * Shortest window {@link cutoff} still admits — see {@link minimumWindow}.
   *
   * Raised in {@link acceptKnownScore}, which is the only place `cutoff` moves.
   */
  let minWindow = minimumWindow(len1, cutoff)

  /**
   * Take `score` as the window's, without scoring it.
   *
   * Returns true once a perfect alignment is found and the search can stop.
   *
   * Split from {@link consider} for the bisection, which has already run the
   * kernel over the window and worked the score out from what came back. See
   * {@link scanInterior} for why the two agree to the last bit.
   */
  const acceptKnownScore = (score: number, start: number, end: number): boolean => {
    if (score <= res.score) return false

    res.score = score
    cutoff = score
    minWindow = minimumWindow(len1, score)
    res.destStart = start
    res.destEnd = end

    if (res.score === 1) {
      res.score = 100
      return true
    }

    return false
  }

  /** Score the window `[start, end)` of `s2`, and take it if it is the best yet. */
  const consider = (start: number, end: number): boolean => {
    // Scored as a range of `s2` against the held pattern: the window is only
    // ever read, so neither copying it out nor rebuilding `s1`'s masks — both
    // of which cost more than the scoring itself — is necessary.
    const lsRatio = indelNormSimHeld(pattern, len1, s2, start, end - start, cutoff)
    return acceptKnownScore(lsRatio, start, end)
  }

  /** Exact indel distance for a full-length window. */
  const windowDistance = (start: number): number => {
    const lcs = lcsSeqLengthPrepared(pattern, s2, start, len1)
    return 2 * (len1 - lcs)
  }

  /**
   * Windows running off the start of `s2`, shorter than the pattern.
   *
   * Nothing to stop early for, unlike the other two scans. A window of `i < len1`
   * elements has `maximum = len1 + i` and at most `i` elements in common, so its
   * normalised similarity is at most `2i / (len1 + i) < 1` — it can raise the
   * running best but never end the search.
   */
  const scanPrefix = (): void => {
    // Every window here is longer than the last, so a `cutoff` that rises
    // mid-scan can only have made windows already behind us unviable. The start
    // is the whole of the prune.
    for (let i = minWindow; i < len1; i++) {
      if (!holds(i - 1)) continue
      consider(0, i)
    }
  }

  /** Windows running off the end of `s2`, likewise shorter. */
  const scanSuffix = (): boolean => {
    for (let i = len2 - len1; i < len2; i++) {
      // Shortening, so the first window too short to reach the cutoff is the
      // last one worth visiting — including when `consider` below has just
      // raised the cutoff.
      if (len2 - i < minWindow) break
      if (!holds(i)) continue
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
      //
      // Endpoints reach `acceptKnownScore` rather than `consider`, because the
      // score below *is* the one `consider` would arrive at, to the last bit —
      // not an estimate of it. `windowDistance` and `indelNormSimHeld` run the
      // same `lcsSeqLengthPrepared` over the same range, and for a full-length
      // window the two arithmetics coincide: `indelNormSimHeld`'s `maximum` is
      // `2 * len1`, its length-difference ceiling is 1 and so never rejects,
      // and its `1 - (maximum - 2 * lcs) / maximum` is this `1 - distance /
      // (2 * len1)` with `distance` substituted. Every intermediate is a small
      // integer and exact, so the two are the same double, not merely equal to
      // within a tolerance. Its `sim >= scoreCutoff` guard is the `score >=
      // cutoff` test already made below. Routing through `consider` therefore
      // ran the kernel a second time over the window it had just scored.
      const windowCount = lastInterior + 1
      const held = windowCount <= BISECTION_SCRATCH_LIMIT
      const generation = nextBisectionGeneration()
      // A local pair above the cap. Its stamps are zeroes, which match no live
      // generation, so it starts entirely unscored exactly as the held pair does.
      const scores = held ? bisectionScoresFor(windowCount) : new Uint32Array(windowCount)
      const stamps = held ? bisectionStampsFor(windowCount) : new Int32Array(windowCount)
      // From the running best rather than the caller's cutoff: on the alignment
      // path the shorter windows have already been scanned, and whatever they
      // found is a tighter bound than the one asked for.
      let distanceToBeat =
        Math.floor(2 * len1 * (1 - cutoff) + Number.EPSILON * 2 * len1) + 1
      let windows = held ? bisectionWindows : []
      let nextWindows = held ? bisectionNextWindows : []
      windows.length = 0
      nextWindows.length = 0
      windows.push(0, lastInterior)

      while (windows.length !== 0) {
        for (let window = 0; window < windows.length; window += 2) {
          const first = windows[window]
          const last = windows[window + 1]

          if (stamps[first] !== generation) {
            const distance = windowDistance(first)
            scores[first] = distance
            stamps[first] = generation
            const score = 1 - distance / (2 * len1)
            if (distance < distanceToBeat && score >= cutoff) {
              distanceToBeat = distance
              if (acceptKnownScore(score, first, first + len1)) return true
            }
          }
          if (stamps[last] !== generation) {
            const distance = windowDistance(last)
            scores[last] = distance
            stamps[last] = generation
            const score = 1 - distance / (2 * len1)
            if (distance < distanceToBeat && score >= cutoff) {
              distanceToBeat = distance
              if (acceptKnownScore(score, last, last + len1)) return true
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
        if (!holds(i + len1 - 1)) continue
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
  //
  // {@link scanInterior} stops one short of the last full-length window, at
  // `len2 - len1 - 1`; the window at `len2 - len1` is {@link scanSuffix}'s first
  // iteration. So the suffix scan has to come before the prefix scan for the
  // rule above to hold at all — and for equal lengths it is the *only* place a
  // full-length window is scored, since the interior scan has no windows to
  // visit. Scoring it after every prefix window was scoring the strongest
  // candidate last. Measured on 400 equal-length choices, no cutoff:
  // **0.66-0.69x** against edit-distance-1 variants of the query, 0.86-0.90x
  // when a third of them match exactly, and noise on unrelated words of the same
  // length, where no window raises the cutoff enough to prune another. Flat
  // through best-match search, which raises the cutoff to the running best and has
  // already pruned what this reorders.
  if (scoreOnly) {
    if (scanInterior() || scanSuffix()) return res
    scanPrefix()
  } else {
    scanPrefix()
    if (scanInterior() || scanSuffix()) return res
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
  if (isMissing(s1) || isMissing(s2)) return null

  // `convPair` rather than `convertProcessedPair`: unlike the token scorers, nothing
  // below needs code points specifically — it needs the two inputs to agree,
  // which `convPair` gives either way, keeping a pair of BMP strings as strings.
  const [a, b] = convPair(asSequence(s1), asSequence(s2))

  return partialAlignmentConverted(a, b, options.scoreCutoff ?? 0)
}

export function partialAlignmentConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  cutoff: number,
  scoreOnly = false,
  preparedA?: PatternMask,
  preparedCharSetA?: CharSet,
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
  if (isMissing(s1) || isMissing(s2)) return 0

  // Same work as `partialRatioAlignment`, minus its obligation to report *which*
  // alignment won — which is what lets the scan visit the windows in the order
  // that prunes best. See `partialRatioScan`.
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return partialAlignmentConverted(a, b, options.scoreCutoff ?? 0, true)?.score ?? 0
}
