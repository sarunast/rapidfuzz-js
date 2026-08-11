import {
  distance as damerauDistance,
  normalizedDistance as damerauNormalizedDistance,
  normalizedSimilarity as damerauNormalizedSimilarity,
  similarity as damerauSimilarity,
} from '../../src/algorithms/damerauLevenshtein/index.js'
import {
  distance as hammingDistance,
  normalizedDistance as hammingNormalizedDistance,
  normalizedSimilarity as hammingNormalizedSimilarity,
  similarity as hammingSimilarity,
  type HammingDistanceConfiguration,
  type HammingSimilarityConfiguration,
} from '../../src/algorithms/hamming/index.js'
import {
  distance as indelDistance,
  normalizedDistance as indelNormalizedDistance,
  normalizedSimilarity as indelNormalizedSimilarity,
  similarity as indelSimilarity,
} from '../../src/algorithms/indel/index.js'
import {
  distance as jaroDistance,
  normalizedDistance as jaroNormalizedDistance,
  normalizedSimilarity as jaroNormalizedSimilarity,
  similarity as jaroSimilarity,
} from '../../src/algorithms/jaro/index.js'
import {
  distance as jaroWinklerDistance,
  normalizedDistance as jaroWinklerNormalizedDistance,
  normalizedSimilarity as jaroWinklerNormalizedSimilarity,
  similarity as jaroWinklerSimilarity,
  type JaroWinklerDistanceConfiguration,
  type JaroWinklerConfiguration,
} from '../../src/algorithms/jaroWinkler/index.js'
import {
  distance as lcsDistance,
  normalizedDistance as lcsNormalizedDistance,
  normalizedSimilarity as lcsNormalizedSimilarity,
  similarity as lcsSimilarity,
} from '../../src/algorithms/lcs/index.js'
import {
  distance as levenshteinDistance,
  normalizedDistance as levenshteinNormalizedDistance,
  normalizedSimilarity as levenshteinNormalizedSimilarity,
  similarity as levenshteinSimilarity,
  type LevenshteinDistanceConfiguration,
  type LevenshteinSimilarityConfiguration,
} from '../../src/algorithms/levenshtein/index.js'
import {
  distance as osaDistance,
  normalizedDistance as osaNormalizedDistance,
  normalizedSimilarity as osaNormalizedSimilarity,
  similarity as osaSimilarity,
} from '../../src/algorithms/osa/index.js'
import {
  distance as postfixDistance,
  normalizedDistance as postfixNormalizedDistance,
  normalizedSimilarity as postfixNormalizedSimilarity,
  similarity as postfixSimilarity,
} from '../../src/algorithms/postfix/index.js'
import {
  distance as prefixDistance,
  normalizedDistance as prefixNormalizedDistance,
  normalizedSimilarity as prefixNormalizedSimilarity,
  similarity as prefixSimilarity,
} from '../../src/algorithms/prefix/index.js'
import type { Metric } from '../../src/core/metric.js'
import { createScorer, type Scorer } from '../../src/core/scorer.js'
import type { Sequence } from '../../src/core/types.js'
import type { SimilarityConfiguration } from '../../src/core/types.js'

interface ExecutionOptions {
  readonly threshold?: number | undefined
}

function isScorer(value: unknown): value is Scorer {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof Reflect.get(value, 'score') === 'function' &&
    (Reflect.get(value, 'direction') === 'similarity' ||
      Reflect.get(value, 'direction') === 'distance')
  )
}

function compileForTest(metric: unknown, options: object | undefined): Scorer {
  const configuration: Record<string, unknown> = {}
  if (options !== undefined) {
    for (const [key, value] of Object.entries(options)) {
      if (key !== 'threshold') configuration[key] = value
    }
  }
  const scorer: unknown = Reflect.apply(createScorer, undefined, [metric, configuration])
  if (!isScorer(scorer)) throw new TypeError('createScorer returned an invalid scorer')
  return scorer
}

class MetricHarness<DistanceConfig extends object, SimilarityConfig extends object> {
  readonly #distance: Metric<'distance', DistanceConfig>
  readonly #similarity: Metric<'similarity', SimilarityConfig>
  readonly #normalizedDistance: Metric<'distance', DistanceConfig>
  readonly #normalizedSimilarity: Metric<'similarity', SimilarityConfig>

  constructor(
    distance: Metric<'distance', DistanceConfig>,
    similarity: Metric<'similarity', SimilarityConfig>,
    normalizedDistance: Metric<'distance', DistanceConfig>,
    normalizedSimilarity: Metric<'similarity', SimilarityConfig>,
  ) {
    this.#distance = distance
    this.#similarity = similarity
    this.#normalizedDistance = normalizedDistance
    this.#normalizedSimilarity = normalizedSimilarity
  }

  distance(
    a: Sequence,
    b: Sequence,
    options?: DistanceConfig & ExecutionOptions,
  ): number | undefined {
    const scorer = compileForTest(this.#distance, options)
    return options?.threshold === undefined
      ? scorer.score(a, b)
      : scorer.score(a, b, { threshold: options.threshold })
  }

  similarity(
    a: Sequence,
    b: Sequence,
    options?: SimilarityConfig & ExecutionOptions,
  ): number | undefined {
    const scorer = compileForTest(this.#similarity, options)
    return options?.threshold === undefined
      ? scorer.score(a, b)
      : scorer.score(a, b, { threshold: options.threshold })
  }

  normalizedDistance(
    a: Sequence,
    b: Sequence,
    options?: DistanceConfig & ExecutionOptions,
  ): number | undefined {
    const scorer = compileForTest(this.#normalizedDistance, options)
    return options?.threshold === undefined
      ? scorer.score(a, b)
      : scorer.score(a, b, { threshold: options.threshold })
  }

  normalizedSimilarity(
    a: Sequence,
    b: Sequence,
    options?: SimilarityConfig & ExecutionOptions,
  ): number | undefined {
    const scorer = compileForTest(this.#normalizedSimilarity, options)
    return options?.threshold === undefined
      ? scorer.score(a, b)
      : scorer.score(a, b, { threshold: options.threshold })
  }
}

type EmptyConfiguration = Record<never, never>

export const DamerauLevenshtein = new MetricHarness<
  EmptyConfiguration,
  SimilarityConfiguration
>(
  damerauDistance,
  damerauSimilarity,
  damerauNormalizedDistance,
  damerauNormalizedSimilarity,
)

export const Hamming = new MetricHarness<
  HammingDistanceConfiguration,
  HammingSimilarityConfiguration
>(
  hammingDistance,
  hammingSimilarity,
  hammingNormalizedDistance,
  hammingNormalizedSimilarity,
)

export const Indel = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  indelDistance,
  indelSimilarity,
  indelNormalizedDistance,
  indelNormalizedSimilarity,
)

export const LCSseq = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  lcsDistance,
  lcsSimilarity,
  lcsNormalizedDistance,
  lcsNormalizedSimilarity,
)

export const Levenshtein = new MetricHarness<
  LevenshteinDistanceConfiguration,
  LevenshteinSimilarityConfiguration
>(
  levenshteinDistance,
  levenshteinSimilarity,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
)

export const OSA = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  osaDistance,
  osaSimilarity,
  osaNormalizedDistance,
  osaNormalizedSimilarity,
)

export const Postfix = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  postfixDistance,
  postfixSimilarity,
  postfixNormalizedDistance,
  postfixNormalizedSimilarity,
)

export const Prefix = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  prefixDistance,
  prefixSimilarity,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
)
export const Jaro = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  jaroDistance,
  jaroSimilarity,
  jaroNormalizedDistance,
  jaroNormalizedSimilarity,
)
export const JaroWinkler = new MetricHarness<
  JaroWinklerDistanceConfiguration,
  JaroWinklerConfiguration
>(
  jaroWinklerDistance,
  jaroWinklerSimilarity,
  jaroWinklerNormalizedDistance,
  jaroWinklerNormalizedSimilarity,
)
