import {
  distance as damerauDistance,
  similarity as damerauSimilarity,
} from '../../src/algorithms/damerauLevenshtein/index.js'
import {
  distance as hammingDistance,
  similarity as hammingSimilarity,
  type HammingDistanceConfiguration,
  type HammingSimilarityConfiguration,
} from '../../src/algorithms/hamming/index.js'
import {
  distance as indelDistance,
  similarity as indelSimilarity,
} from '../../src/algorithms/indel/index.js'
import { similarity as jaroSimilarity } from '../../src/algorithms/jaro/index.js'
import {
  similarity as jaroWinklerSimilarity,
  type JaroWinklerConfiguration,
} from '../../src/algorithms/jaroWinkler/index.js'
import {
  distance as lcsDistance,
  similarity as lcsSimilarity,
} from '../../src/algorithms/lcs/index.js'
import {
  distance as levenshteinDistance,
  similarity as levenshteinSimilarity,
  type LevenshteinDistanceConfiguration,
  type LevenshteinSimilarityConfiguration,
} from '../../src/algorithms/levenshtein/index.js'
import {
  distance as osaDistance,
  similarity as osaSimilarity,
} from '../../src/algorithms/osa/index.js'
import {
  distance as postfixDistance,
  similarity as postfixSimilarity,
} from '../../src/algorithms/postfix/index.js'
import {
  distance as prefixDistance,
  similarity as prefixSimilarity,
} from '../../src/algorithms/prefix/index.js'
import type { Metric } from '../../src/core/metric.js'
import { createScorer } from '../../src/core/scorer.js'
import type { Sequence } from '../../src/core/types.js'
import type { SimilarityConfiguration } from '../../src/core/types.js'

interface ExecutionOptions {
  readonly threshold?: number | undefined
}

class MetricHarness<DistanceConfig extends object, SimilarityConfig extends object> {
  readonly #distance: Metric<'distance', DistanceConfig>
  readonly #similarity: Metric<'similarity', SimilarityConfig>

  constructor(
    distance: Metric<'distance', DistanceConfig>,
    similarity: Metric<'similarity', SimilarityConfig>,
  ) {
    this.#distance = distance
    this.#similarity = similarity
  }

  distance(
    a: Sequence,
    b: Sequence,
    options?: DistanceConfig & ExecutionOptions,
  ): number | undefined {
    const scorer = createScorer(this.#distance, options)
    return options?.threshold === undefined
      ? scorer.score(a, b)
      : scorer.score(a, b, { threshold: options.threshold })
  }

  similarity(
    a: Sequence,
    b: Sequence,
    options?: SimilarityConfig & ExecutionOptions,
  ): number | undefined {
    const scorer = createScorer(this.#similarity, options)
    return options?.threshold === undefined
      ? scorer.score(a, b)
      : scorer.score(a, b, { threshold: options.threshold })
  }
}

type EmptyConfiguration = Record<never, never>

export const DamerauLevenshtein = new MetricHarness<
  EmptyConfiguration,
  SimilarityConfiguration
>(damerauDistance, damerauSimilarity)

export const Hamming = new MetricHarness<
  HammingDistanceConfiguration,
  HammingSimilarityConfiguration
>(hammingDistance, hammingSimilarity)

export const Indel = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  indelDistance,
  indelSimilarity,
)

export const LCSseq = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  lcsDistance,
  lcsSimilarity,
)

export const Levenshtein = new MetricHarness<
  LevenshteinDistanceConfiguration,
  LevenshteinSimilarityConfiguration
>(levenshteinDistance, levenshteinSimilarity)

export const OSA = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  osaDistance,
  osaSimilarity,
)

export const Postfix = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  postfixDistance,
  postfixSimilarity,
)

export const Prefix = new MetricHarness<EmptyConfiguration, SimilarityConfiguration>(
  prefixDistance,
  prefixSimilarity,
)

class SimilarityHarness<Config extends object> {
  readonly #metric: Metric<'similarity', Config>

  constructor(metric: Metric<'similarity', Config>) {
    this.#metric = metric
  }

  similarity(
    a: Sequence,
    b: Sequence,
    options?: Config & ExecutionOptions,
  ): number | undefined {
    const scorer = createScorer(this.#metric, options)
    return options?.threshold === undefined
      ? scorer.score(a, b)
      : scorer.score(a, b, { threshold: options.threshold })
  }
}

export const Jaro = new SimilarityHarness(jaroSimilarity)
export const JaroWinkler = new SimilarityHarness<JaroWinklerConfiguration>(
  jaroWinklerSimilarity,
)
