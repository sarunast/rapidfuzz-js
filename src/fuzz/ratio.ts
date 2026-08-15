import { prepareLcsPattern } from '../algorithms/lcs/implementation.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../core/scoring/builtIn/implementation.js'
import type { BuiltInMetric } from '../core/scoring/builtIn/metric.js'
import {
  prepareChoiceSequence,
  preparedChoiceSequence,
  type PreparationFactory,
} from '../core/scoring/builtIn/preparation.js'
import type { PreparedKernel } from '../core/scoring/compilation.js'
import { scorerSequence } from '../core/sequence.js'
import type { Sequence } from '../core/types.js'
import { fuzzMetric } from './metric.js'
import { ratioHeld, ratio_impl } from './partialWindow.js'
import type { FuzzOptions } from './types.js'

/** Narrow prepared-query implementation for the basic fuzz similarity. */
export function prepareRatio(): PreparationFactory {
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

export const fuzzRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(ratio_impl, FUZZ_FLAGS, prepareRatio())

/**
 * The baseline of the family: normalized Indel similarity scaled to `0..100`.
 *
 * ```ts
 * ratio('this is a test', 'this is a test!') // 96.55…
 * ```
 *
 * It compares the two inputs *whole*, so anything structural counts against the
 * score — reordered words, extra words, one side containing the other:
 *
 * ```ts
 * ratio('smith john', 'john smith') // 50 — same words, wrong order
 * ratio('new york jets', 'the new york jets play tonight') // 60.46…
 * ```
 *
 * Those are the cases the rest of the family exists for. Reach past this one when
 * you can name which of them your data has.
 */
export const ratio: BuiltInMetric<'fuzz.ratio', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(fuzzRatio)
