import { normDistCutoff, normSimCutoff } from '#core/scoring/builtIn/cutoff.js'
import {
  type ConfigurationCanonicalizer,
  type MaybeSequenceMetricImplementation,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  withPreparedFlags,
} from '#core/scoring/builtIn/implementation.js'
import type { ScorerOptions } from '#core/scoring/builtIn/options.js'
import {
  prepareChoiceSequence,
  preparedChoiceSequence,
  type PreparationFactory,
} from '#core/scoring/builtIn/preparation.js'
import type { PreparedKernel } from '#core/scoring/compilation.js'
import {
  alignRepresentation,
  validateSequence,
  convPair,
  queryAligner,
  scorerSequence,
} from '#core/sequence.js'
import type { MaybeSequence, Sequence } from '#core/types.js'

import { commonPrefix } from '../affix.js'
import { preparePattern, type PatternMask } from '../bitmask/pattern.js'
import { jaroSimilarity_, jaroSimilarityPrepared_ } from '../jaro/implementation.js'

interface JaroWinklerOptions extends ScorerOptions {
  prefixWeight?: number | undefined
}

function jaroCutoffFor(prefixSimilarity: number, scoreCutoff: number): number {
  if (scoreCutoff <= 0.7) return scoreCutoff
  if (prefixSimilarity >= 1) return 0.7
  return Math.max(0.7, (prefixSimilarity - scoreCutoff) / (prefixSimilarity - 1))
}

function winklerBonus(
  jaro: number,
  prefixSimilarity: number,
  scoreCutoff: number,
): number {
  const sim = jaro > 0.7 ? Math.min(jaro + prefixSimilarity * (1 - jaro), 1) : jaro
  return sim >= scoreCutoff ? sim : 0
}

function similarity_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  prefixWeight: number,
  scoreCutoff: number,
  prepared: PatternMask,
): number {
  const prefixSimilarity = Math.min(commonPrefix(s1, s2), 4) * prefixWeight
  const jaroCutoff = jaroCutoffFor(prefixSimilarity, scoreCutoff)
  const jaro = jaroSimilarityPrepared_(s1, prepared, s2, jaroCutoff)
  return winklerBonus(jaro, prefixSimilarity, scoreCutoff)
}

function directSimilarity(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  prefixWeight: number,
  scoreCutoff: number,
): number {
  if (prefixWeight > 1 || prefixWeight < 0) {
    throw new RangeError('prefix_weight has to be in the range 0.0 - 1.0')
  }
  const prefixSimilarity = Math.min(commonPrefix(s1, s2), 4) * prefixWeight
  const jaroCutoff = jaroCutoffFor(prefixSimilarity, scoreCutoff)
  return winklerBonus(jaroSimilarity_(s1, s2, jaroCutoff), prefixSimilarity, scoreCutoff)
}

function jaroWinklerSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: JaroWinklerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return normSimCutoff(
    directSimilarity(a, b, options.prefixWeight ?? 0.1, options.scoreCutoff ?? 0),
    options.scoreCutoff,
  )
}

function jaroWinklerDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: JaroWinklerOptions = {},
): number {
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const cutoff = options.scoreCutoff
  return normDistCutoff(
    1 -
      directSimilarity(
        a,
        b,
        options.prefixWeight ?? 0.1,
        cutoff == null ? 0 : 1 - cutoff,
      ),
    cutoff,
  )
}

type PreparedJaroWinklerKind = 'distance' | 'similarity'

function parsePrefixWeight(options: Readonly<Record<string, unknown>>): number {
  const prefixWeight = Reflect.get(options, 'prefixWeight')
  if (prefixWeight == null) return 0.1
  if (typeof prefixWeight !== 'number') {
    throw new TypeError('prefixWeight must be a number')
  }
  if (!(prefixWeight >= 0 && prefixWeight <= 1)) {
    throw new RangeError('prefix_weight has to be in the range 0.0 - 1.0')
  }
  return prefixWeight
}

function prepareJaroWinkler(kind: PreparedJaroWinklerKind): PreparationFactory {
  return (options) => {
    const prefixWeight = parsePrefixWeight(options)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = scorerSequence(query)
      const pattern = preparePattern(a, 0, a.length)
      const alignedQueryFor = queryAligner(a)

      return (rawChoice, rawCutoff) => {
        const b = preparedChoiceSequence(rawChoice)
        const similarityCutoff =
          kind === 'distance'
            ? rawCutoff === null
              ? 0
              : 1 - rawCutoff
            : (rawCutoff ?? 0)
        const similarity = similarity_(
          alignedQueryFor(b),
          alignRepresentation(b, a),
          prefixWeight,
          similarityCutoff,
          pattern,
        )
        return kind === 'distance'
          ? normDistCutoff(1 - similarity, rawCutoff)
          : normSimCutoff(similarity, rawCutoff)
      }
    }

    return { prepareQuery, prepareChoice: prepareChoiceSequence }
  }
}

const jaroWinklerConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) => {
  parsePrefixWeight(options)
  return options
}

export const jaroWinklerSimilarity: MaybeSequenceMetricImplementation<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareJaroWinkler('similarity'),
    { configurationCanonicalizer: jaroWinklerConfigurationCanonicalizer },
  )
export const jaroWinklerDistance: MaybeSequenceMetricImplementation<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareJaroWinkler('distance'),
    { configurationCanonicalizer: jaroWinklerConfigurationCanonicalizer },
  )
