import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from '#core/editops/index.js'
import {
  distanceCutoffFor,
  distCutoff,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
} from '#core/scoring/builtIn/cutoff.js'
import {
  type MaybeSequenceMetricImplementation,
  type ConfigurationCanonicalizer,
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
} from '#core/scoring/builtIn/implementation.js'
import type { ScorerOptions } from '#core/scoring/builtIn/options.js'
import { prepareMetric } from '#core/scoring/builtIn/preparation.js'
import { validateSequence, convPair, maxSequenceLength } from '#core/sequence.js'
import type { MaybeSequence, Sequence } from '#core/types.js'

export interface HammingEditopsOptions {
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

  if (surplus > cutoff) return cutoff + 1

  return cutoff === Number.POSITIVE_INFINITY
    ? exactMismatches(s1, s2, limit, surplus)
    : boundedMismatches(s1, s2, limit, surplus, cutoff)
}

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

function hammingDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: HammingOptions = {},
): number {
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const cutoff = distanceCutoffFor(
    'distance',
    options.scoreCutoff,
    maxSequenceLength(a, b),
  )
  return distCutoff(distance_(a, b, options.pad ?? true, cutoff), options.scoreCutoff)
}

function hammingSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: HammingOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const max = maxSequenceLength(a, b)
  const cutoff = distanceCutoffFor('similarity', options.scoreCutoff, max)
  return simCutoff(
    max - distance_(a, b, options.pad ?? true, cutoff),
    options.scoreCutoff,
  )
}

function hammingNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: HammingOptions = {},
): number {
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const max = maxSequenceLength(a, b)
  const cutoff = distanceCutoffFor('normalizedDistance', options.scoreCutoff, max)
  const norm = normalizeDistance(distance_(a, b, options.pad ?? true, cutoff), max)
  return normDistCutoff(norm, options.scoreCutoff)
}

function hammingNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: HammingOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const max = maxSequenceLength(a, b)
  const cutoff = distanceCutoffFor('normalizedSimilarity', options.scoreCutoff, max)
  const norm = normalizeDistance(distance_(a, b, options.pad ?? true, cutoff), max)
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
  const [a, b] = convPair(s1, s2)

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

/**
 * {@link hammingEditops} as contiguous ranges rather than single operations.
 *
 * Opcodes cover the whole of both inputs, including the `equal` stretches
 * between edits, which is usually what a diff view or a highlighter wants —
 * `editops` lists only the changes. The two convert into each other with
 * `toEditops()` and `toOpcodes()`.
 */
export function hammingOpcodes(
  s1: Sequence,
  s2: Sequence,
  options: HammingEditopsOptions = {},
): Opcodes {
  return hammingEditops(s1, s2, options).toOpcodes()
}

const hammingConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) => {
  const pad = Reflect.get(options, 'pad')
  if (pad == null || typeof pad === 'boolean') return options
  throw new TypeError('pad must be a boolean')
}

export const hammingDistance: MaybeSequenceMetricImplementation<HammingOptions> =
  /* @__PURE__ */ withPreparedFlags(
    hammingDistance_impl,
    DISTANCE_FLAGS,
    prepareMetric('distance', preparedHammingDistance, maxSequenceLength, parsedPad),
    { configurationCanonicalizer: hammingConfigurationCanonicalizer },
  )
export const hammingSimilarity: MaybeSequenceMetricImplementation<HammingOptions> =
  /* @__PURE__ */ withPreparedFlags(
    hammingSimilarity_impl,
    SIMILARITY_FLAGS,
    prepareMetric('similarity', preparedHammingDistance, maxSequenceLength, parsedPad),
    { configurationCanonicalizer: hammingConfigurationCanonicalizer },
  )
export const hammingNormalizedDistance: MaybeSequenceMetricImplementation<HammingOptions> =
  /* @__PURE__ */ withPreparedFlags(
    hammingNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareMetric(
      'normalizedDistance',
      preparedHammingDistance,
      maxSequenceLength,
      parsedPad,
    ),
    { configurationCanonicalizer: hammingConfigurationCanonicalizer },
  )
export const hammingNormalizedSimilarity: MaybeSequenceMetricImplementation<HammingOptions> =
  /* @__PURE__ */ withPreparedFlags(
    hammingNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareMetric(
      'normalizedSimilarity',
      preparedHammingDistance,
      maxSequenceLength,
      parsedPad,
    ),
    { configurationCanonicalizer: hammingConfigurationCanonicalizer },
  )
