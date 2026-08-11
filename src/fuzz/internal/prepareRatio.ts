import {
  lcsSeqLengthPrepared,
  lcsSeqLengthPreparedBounded,
  prepareLcsPattern,
} from '../../algorithms/lcs/implementation.js'
import {
  isNone,
  isSequence,
  prepareScorerChoice,
  preparedScorerSequence,
  scorerSequence,
  withChoicePreparer,
  type PrepareScorer,
  type PreparedScore,
} from '../../algorithms/shared/scorerSupport.js'

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
export function prepareRatio(): PrepareScorer {
  const prepare: PrepareScorer = (query) => {
    const held = preparedScorerSequence(prepareScorerChoice(query))
    if (held === null) throw new TypeError('fuzz scorers expect a sequence')
    const pattern = prepareLcsPattern(held, 0, held.length)
    const score: PreparedScore = (rawChoice, rawCutoff) => {
      if (isNone(rawChoice)) return 0
      let choice = preparedScorerSequence(rawChoice)
      if (choice === null) {
        if (!isSequence(rawChoice)) {
          throw new TypeError('fuzz scorers expect a string or an array-like sequence')
        }
        choice = scorerSequence(rawChoice)
      }
      return scoreRatio(held, pattern, choice, rawCutoff ?? 0)
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}
