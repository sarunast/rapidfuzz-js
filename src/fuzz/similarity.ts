import { prepareLcsPattern } from '../algorithms/lcs/implementation.js'
import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  prepareChoiceSequence,
  preparedChoiceSequence,
  scorerSequence,
  type PreparationFactory,
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { PreparedKernel } from '../core/scoring/compilation.js'
import type { Sequence } from '../core/types.js'
import { fuzzMetric } from './internal/metric.js'
import { ratioHeld, ratio_impl } from './internal/partialWindow.js'
import type { FuzzOptions } from './types.js'

/** Narrow prepared-query implementation for the basic fuzz similarity. */
export function prepareSimilarity(): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const held = scorerSequence(query)
    const pattern = prepareLcsPattern(held, 0, held.length)
    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const choice = preparedChoiceSequence(rawChoice)
      return ratioHeld(pattern, held.length, choice, rawCutoff ?? 0)
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: prepareChoiceSequence })
}

export const ratio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(ratio_impl, FUZZ_FLAGS, prepareSimilarity())

/**
 * The baseline of the family: normalized Indel similarity scaled to `0..100`.
 *
 * ```ts
 * similarity('this is a test', 'this is a test!') // 96.55…
 * ```
 *
 * It compares the two inputs *whole*, so anything structural counts against the
 * score — reordered words, extra words, one side containing the other:
 *
 * ```ts
 * similarity('smith john', 'john smith') // 50 — same words, wrong order
 * similarity('new york jets', 'the new york jets play tonight') // 60.46…
 * ```
 *
 * Those are the cases the rest of the family exists for. Reach past this one when
 * you can name which of them your data has.
 *
 * RapidFuzz calls it `ratio`.
 */
export const similarity: BuiltInMetric<'fuzz.similarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(ratio)
