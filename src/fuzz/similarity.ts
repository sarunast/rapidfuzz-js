import {
  lcsSeqLengthPrepared,
  lcsSeqLengthPreparedBounded,
  prepareLcsPattern,
} from '../algorithms/lcs/implementation.js'
import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  prepareScorerChoice,
  preparedScorerSequence,
  withChoicePreparer,
  type PrepareScorer,
  type PreparedScorerFactory,
  type PreparedScore,
  FUZZ_FLAGS,
  type NormalizedScorer,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import { ratio_impl } from './internal/partialWindow.js'
import type { FuzzConfiguration, FuzzOptions } from './types.js'

function scoreRatio(
  query: ArrayLike<unknown>,
  pattern: ReturnType<typeof prepareLcsPattern>,
  choice: ArrayLike<unknown>,
  cutoff: number,
): number {
  const maximum = query.length + choice.length
  if (maximum === 0) return cutoff <= 100 ? 100 : 0
  const ceiling =
    (1 - (maximum - 2 * Math.min(query.length, choice.length)) / maximum) * 100
  if (ceiling < cutoff) return 0
  const required = Math.max(0, Math.floor((cutoff * maximum) / 200))
  const lcs =
    cutoff >= 70 && maximum >= 128
      ? lcsSeqLengthPreparedBounded(pattern, choice, 0, choice.length, required)
      : lcsSeqLengthPrepared(pattern, choice, 0, choice.length)
  if (lcs < 0) return 0
  const score = (1 - (maximum - 2 * lcs) / maximum) * 100
  return score >= cutoff ? score : 0
}

/** Narrow prepared-query implementation for the basic fuzz similarity. */
export function prepareSimilarity(): PreparedScorerFactory {
  const prepare: PrepareScorer = (query) => {
    const held = preparedScorerSequence(prepareScorerChoice(query))
    const pattern = prepareLcsPattern(held, 0, held.length)
    const score: PreparedScore = (rawChoice, rawCutoff) => {
      const choice = preparedScorerSequence(rawChoice)
      return scoreRatio(held, pattern, choice, rawCutoff ?? 0)
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

const BOUNDS: readonly [number, number] = [0, 100]

export const ratio: NormalizedScorer<FuzzOptions> = /* @__PURE__ */ withPreparedFlags(
  ratio_impl,
  FUZZ_FLAGS,
  prepareSimilarity(),
)

export const similarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: ratio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
