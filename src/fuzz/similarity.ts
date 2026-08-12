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
import type { PreparedKernel } from '../core/protocol.js'
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

export const similarity: BuiltInMetric<'fuzz.similarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(ratio)
