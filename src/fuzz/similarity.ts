import { prepareLcsPattern } from '../algorithms/lcs/implementation.js'
import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  prepareChoiceSequence,
  preparedChoiceSequence,
  scorerSequence,
  type PreparationFactory,
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import type { PreparedKernel } from '../core/protocol.js'
import type { Sequence } from '../core/types.js'
import { ratioHeld, ratio_impl } from './internal/partialWindow.js'
import type { FuzzConfiguration, FuzzOptions } from './types.js'

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

const BOUNDS: readonly [number, number] = [0, 100]

export const ratio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(ratio_impl, FUZZ_FLAGS, prepareSimilarity())

export const similarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: ratio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
