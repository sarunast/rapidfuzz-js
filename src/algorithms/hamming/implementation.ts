import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from '../shared/editops/index.js'
import {
  conv,
  distanceCutoffFor,
  distCutoff,
  normalize,
  normSimCutoff,
  type ScorerOptions,
  type Sequence,
  prepareMetric,
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  type EditopsOptions,
  type Scorer,
} from '../shared/scorerSupport.js'

export interface HammingEditopsOptions extends EditopsOptions {
  /** See {@link HammingOptions.pad}. Defaults to `true`. */
  pad?: boolean | undefined
}

function parsedPad(kwargs: Readonly<Record<string, unknown>>): unknown {
  return Reflect.get(kwargs, 'pad') ?? true
}

function preparedHammingDistance(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  pad: unknown,
  distanceCutoff: number,
): number {
  return distance_(s1, s2, Boolean(pad), distanceCutoff)
}

export interface HammingOptions extends ScorerOptions {
  /**
   * When `true` (the default) the shorter input is treated as padded, so every
   * surplus element of the longer one counts as a mismatch. When `false`,
   * inputs of differing length are an error.
   */
  pad?: boolean | undefined
}

/**
 * Count the differing positions, abandoning the scan once `cutoff` is passed.
 *
 * Hamming has no structure to exploit — every position has to be looked at —
 * so a cutoff is the only thing that can make it sublinear. Under
 * best-match search, where the running best tightens the bound after every
 * candidate, most comparisons are decided within the first few positions
 * instead of after a full pass. The bail-out value is only required to exceed
 * `cutoff`; each caller maps it onto the rejection its convention reports.
 */
function distance_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  pad: boolean,
  cutoff = Number.POSITIVE_INFINITY,
): number {
  const len1 = s1.length
  const len2 = s2.length

  if (!pad && len1 !== len2) {
    throw new Error('Sequences are not the same length.')
  }

  const limit = Math.min(len1, len2)
  const surplus = Math.max(len1, len2) - limit

  // The length difference alone is already a lower bound on the distance, so
  // badly mismatched candidates never reach a loop at all.
  if (surplus > cutoff) return cutoff + 1

  // Two loops rather than one with the bound folded in. Testing the running
  // distance after every mismatch measured ~7% on an unbounded `scoreMatrix`, where
  // the test can never fire and `cutoff` is an infinity that drags the integer
  // comparison into floating point. Most calls have no cutoff, so they get a
  // loop that does not ask.
  return cutoff === Number.POSITIVE_INFINITY
    ? exactMismatches(s1, s2, limit, surplus)
    : boundedMismatches(s1, s2, limit, surplus, cutoff)
}

/**
 * Indexing a string yields a fresh one-character string per position, and
 * comparing two of those is a string comparison. Reading the code units instead
 * compares two integers. Both inputs share a representation by the time they
 * arrive here — `conv` on the direct path, `alignRepresentation` on the
 * prepared one — so this only has to ask the question once.
 */
function exactMismatches(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  limit: number,
  surplus: number,
): number {
  let dist = surplus

  if (typeof s1 === 'string' && typeof s2 === 'string') {
    for (let i = 0; i < limit; i++) {
      if (s1.charCodeAt(i) !== s2.charCodeAt(i)) dist++
    }
    return dist
  }

  for (let i = 0; i < limit; i++) {
    if (s1[i] !== s2[i]) dist++
  }

  return dist
}

/** {@link exactMismatches}, abandoned as soon as the count passes `cutoff`. */
function boundedMismatches(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  limit: number,
  surplus: number,
  cutoff: number,
): number {
  let dist = surplus

  if (typeof s1 === 'string' && typeof s2 === 'string') {
    for (let i = 0; i < limit; i++) {
      if (s1.charCodeAt(i) !== s2.charCodeAt(i) && ++dist > cutoff) return cutoff + 1
    }
    return dist
  }

  for (let i = 0; i < limit; i++) {
    if (s1[i] !== s2[i] && ++dist > cutoff) return cutoff + 1
  }

  return dist
}

function maximum(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return Math.max(s1.length, s2.length)
}

/**
 * Hamming distance: the number of positions at which the inputs differ, plus
 * the length difference when `pad` is enabled.
 *
 * If the distance is greater than `scoreCutoff`, `scoreCutoff + 1` is returned.
 *
 * @throws if `pad` is `false` and the inputs have different lengths.
 */
function hammingDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: HammingOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  const cutoff = distanceCutoffFor('distance', options.scoreCutoff, maximum(a, b))
  return distCutoff(distance_(a, b, options.pad ?? true, cutoff), options.scoreCutoff)
}

/**
 * Hamming similarity normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function hammingNormalizedSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: HammingOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  const max = maximum(a, b)
  const cutoff = distanceCutoffFor('normalizedSimilarity', options.scoreCutoff, max)
  const norm = normalize(distance_(a, b, options.pad ?? true, cutoff), max)
  return normSimCutoff(1 - norm, options.scoreCutoff)
}

/**
 * Edit operations that turn `s1` into `s2` position by position: a `replace`
 * for each differing position, then deletions or insertions for the tail.
 *
 * @throws if `pad` is `false` and the inputs have different lengths.
 */
export function hammingEditops(
  s1: Sequence,
  s2: Sequence,
  options: HammingEditopsOptions = {},
): Editops {
  const [a, b] = conv(s1, s2, options.processor)

  if (options.pad === false && a.length !== b.length) {
    throw new Error('Sequences are not the same length.')
  }

  const ops: Editop[] = []
  const minLen = Math.min(a.length, b.length)

  for (let i = 0; i < minLen; i++) {
    if (a[i] !== b[i]) ops.push({ tag: 'replace', srcPos: i, destPos: i })
  }
  for (let i = minLen; i < a.length; i++)
    ops.push({ tag: 'delete', srcPos: i, destPos: b.length })
  for (let i = minLen; i < b.length; i++)
    ops.push({ tag: 'insert', srcPos: a.length, destPos: i })

  return editopsFromValidated(ops, a.length, b.length)
}

/** {@link hammingEditops} expressed as blocks. */
export function hammingOpcodes(
  s1: Sequence,
  s2: Sequence,
  options: HammingEditopsOptions = {},
): Opcodes {
  return hammingEditops(s1, s2, options).toOpcodes()
}

export const hammingDistance: Scorer<HammingOptions> = /* @__PURE__ */ withPreparedFlags(
  hammingDistance_impl,
  DISTANCE_FLAGS,
  prepareMetric('distance', preparedHammingDistance, maximum, parsedPad),
)
export const hammingNormalizedSimilarity: Scorer<HammingOptions> =
  /* @__PURE__ */ withPreparedFlags(
    hammingNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareMetric('normalizedSimilarity', preparedHammingDistance, maximum, parsedPad),
  )
