import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import {
  indelDistance,
  indelEditops,
  indelNormalizedDistance,
  indelNormalizedSimilarity,
  indelOpcodes,
  indelSimilarity,
} from './implementation.js'

/**
 * Edit distance counting insertions and deletions only — no substitution,
 * so changing a character costs two steps.
 *
 * ```ts
 * distance('lewenstein', 'levenshtein') // 3
 * ```
 *
 * It is the mirror of the longest common subsequence:
 * `distance(a, b) === a.length + b.length - 2 * lcs(a, b)`. The whole fuzz
 * family is built on its normalized form.
 */
export const distance: BuiltInMetric<'indel.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: indelDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * How much the two sequences share, in edit units: `maximum − distance`.
 *
 * **Not a 0–1 score** — {@link normalizedSimilarity} is the fraction.
 */
export const similarity: BuiltInMetric<'indel.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: indelSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * {@link distance} as a `0..1` fraction of the combined length — `0`
 * identical, `1` nothing in common.
 */
export const normalizedDistance: BuiltInMetric<'indel.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: indelNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
/**
 * {@link similarity} as a `0..1` fraction — `1` identical. This is the
 * measure the fuzz scorers scale to `0..100`.
 *
 * ```ts
 * normalizedSimilarity('abc', 'axc') // 0.666… — the b/x swap costs 2 of 6
 * ```
 */
export const normalizedSimilarity: BuiltInMetric<
  'indel.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: indelNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
export { indelEditops as editops, indelOpcodes as opcodes }
