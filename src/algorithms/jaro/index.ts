import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import { jaroDistance, jaroSimilarity } from './implementation.js'

/** `1 − similarity`, on the same `0..1` scale. */
export const distance: BuiltInMetric<'jaro.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroDistance,
    directImplementation: jaroDistance,
    direction: 'distance',
    bounds: [0, 1],
  })

/**
 * Jaro similarity, `0..1` — built for short strings, especially names, where a
 * couple of differences should not crater the score.
 *
 * Rather than counting edits it finds elements common to both sides that sit
 * close enough in position (within half the longer length), then penalises how
 * many of those matches are out of order.
 *
 * ```ts
 * similarity('martha', 'marhta') // 0.9444… — all letters match, one pair swapped
 * ```
 *
 * Most callers matching people or places want Jaro-Winkler, which adds a
 * shared-prefix bonus on top of this.
 */
export const similarity: BuiltInMetric<'jaro.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroSimilarity,
    directImplementation: jaroSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })

// Jaro is normalized by construction, so these are the same metrics under the
// names the other algorithms use. `typeof` carries the identity across instead
// of restating it, which is what keeps their prepared choices interchangeable.
/** Jaro is already `0..1`, so this is {@link distance} itself. */
export const normalizedDistance: typeof distance = distance
/** Jaro is already `0..1`, so this is {@link similarity} itself. */
export const normalizedSimilarity: typeof similarity = similarity
