import {
  damerauLevenshteinDistance,
  damerauLevenshteinNormalizedDistance,
  damerauLevenshteinNormalizedSimilarity,
  damerauLevenshteinSimilarity,
} from '../../src/algorithms/damerauLevenshtein/implementation.js'
import {
  hammingDistance,
  hammingNormalizedDistance,
  hammingNormalizedSimilarity,
  hammingSimilarity,
} from '../../src/algorithms/hamming/implementation.js'
import {
  indelDistance,
  indelNormalizedDistance,
  indelNormalizedSimilarity,
  indelSimilarity,
} from '../../src/algorithms/indel/implementation.js'
import {
  jaroDistance,
  jaroNormalizedDistance,
  jaroNormalizedSimilarity,
  jaroSimilarity,
} from '../../src/algorithms/jaro/implementation.js'
import {
  jaroWinklerDistance,
  jaroWinklerNormalizedDistance,
  jaroWinklerNormalizedSimilarity,
  jaroWinklerSimilarity,
} from '../../src/algorithms/jaroWinkler/implementation.js'
import {
  lcsSeqDistance,
  lcsSeqNormalizedDistance,
  lcsSeqNormalizedSimilarity,
  lcsSeqSimilarity,
} from '../../src/algorithms/lcs/implementation.js'
import {
  levenshteinCosts,
  levenshteinDistance,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
} from '../../src/algorithms/levenshtein/metric.js'
import {
  osaDistance,
  osaNormalizedDistance,
  osaNormalizedSimilarity,
  osaSimilarity,
} from '../../src/algorithms/osa/implementation.js'
import {
  postfixDistance,
  postfixNormalizedDistance,
  postfixNormalizedSimilarity,
  postfixSimilarity,
} from '../../src/algorithms/postfix/implementation.js'
import {
  prefixDistance,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
  prefixSimilarity,
} from '../../src/algorithms/prefix/implementation.js'
/**
 * Port of RapidFuzz's `tests/distance/common.py` — one `GenericScorer` per
 * metric, each declaring the `maximum` its four entry points must agree on.
 */
import { GenericScorer, maxLen, type TestOptions } from './common.js'

export const DamerauLevenshtein: GenericScorer = new GenericScorer(
  {
    distance: damerauLevenshteinDistance,
    similarity: damerauLevenshteinSimilarity,
    normalizedDistance: damerauLevenshteinNormalizedDistance,
    normalizedSimilarity: damerauLevenshteinNormalizedSimilarity,
  },
  (s1, s2) => ({ maximum: maxLen(s1, s2), symmetric: true }),
)

export const Hamming: GenericScorer = new GenericScorer(
  {
    distance: hammingDistance,
    similarity: hammingSimilarity,
    normalizedDistance: hammingNormalizedDistance,
    normalizedSimilarity: hammingNormalizedSimilarity,
  },
  (s1, s2) => ({ maximum: maxLen(s1, s2), symmetric: true }),
)

export const Indel: GenericScorer = new GenericScorer(
  {
    distance: indelDistance,
    similarity: indelSimilarity,
    normalizedDistance: indelNormalizedDistance,
    normalizedSimilarity: indelNormalizedSimilarity,
  },
  (s1, s2) => ({ maximum: s1.length + s2.length, symmetric: true }),
)

export const Jaro: GenericScorer = new GenericScorer(
  {
    distance: jaroDistance,
    similarity: jaroSimilarity,
    normalizedDistance: jaroNormalizedDistance,
    normalizedSimilarity: jaroNormalizedSimilarity,
  },
  () => ({ maximum: 1, symmetric: true }),
)

export const JaroWinkler: GenericScorer = new GenericScorer(
  {
    distance: jaroWinklerDistance,
    similarity: jaroWinklerSimilarity,
    normalizedDistance: jaroWinklerNormalizedDistance,
    normalizedSimilarity: jaroWinklerNormalizedSimilarity,
  },
  () => ({ maximum: 1, symmetric: true }),
)

export const LCSseq: GenericScorer = new GenericScorer(
  {
    distance: lcsSeqDistance,
    similarity: lcsSeqSimilarity,
    normalizedDistance: lcsSeqNormalizedDistance,
    normalizedSimilarity: lcsSeqNormalizedSimilarity,
  },
  (s1, s2) => ({ maximum: maxLen(s1, s2), symmetric: true }),
)

/**
 * Mirrors `get_scorer_flags_levenshtein`.
 *
 * Reads the option through the production normalizer rather than destructuring
 * it, so the two spellings of `weights` cannot diverge between what the scorer
 * scores and what this harness expects it to score.
 */
function levenshteinFlags(
  s1: { length: number },
  s2: { length: number },
  options: TestOptions,
) {
  const { insertion, deletion, substitution } = levenshteinCosts(options.weights)
  const indel = s1.length * deletion + s2.length * insertion

  const maximum =
    s1.length >= s2.length
      ? Math.min(indel, s2.length * substitution + (s1.length - s2.length) * deletion)
      : Math.min(indel, s1.length * substitution + (s2.length - s1.length) * insertion)

  return { maximum, symmetric: insertion === deletion }
}

export const Levenshtein: GenericScorer = new GenericScorer(
  {
    distance: levenshteinDistance,
    similarity: levenshteinSimilarity,
    normalizedDistance: levenshteinNormalizedDistance,
    normalizedSimilarity: levenshteinNormalizedSimilarity,
  },
  levenshteinFlags,
)

export const OSA: GenericScorer = new GenericScorer(
  {
    distance: osaDistance,
    similarity: osaSimilarity,
    normalizedDistance: osaNormalizedDistance,
    normalizedSimilarity: osaNormalizedSimilarity,
  },
  (s1, s2) => ({ maximum: maxLen(s1, s2), symmetric: true }),
)

export const Postfix: GenericScorer = new GenericScorer(
  {
    distance: postfixDistance,
    similarity: postfixSimilarity,
    normalizedDistance: postfixNormalizedDistance,
    normalizedSimilarity: postfixNormalizedSimilarity,
  },
  (s1, s2) => ({ maximum: maxLen(s1, s2), symmetric: true }),
)

export const Prefix: GenericScorer = new GenericScorer(
  {
    distance: prefixDistance,
    similarity: prefixSimilarity,
    normalizedDistance: prefixNormalizedDistance,
    normalizedSimilarity: prefixNormalizedSimilarity,
  },
  (s1, s2) => ({ maximum: maxLen(s1, s2), symmetric: true }),
)
