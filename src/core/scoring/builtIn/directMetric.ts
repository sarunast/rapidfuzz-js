import { convPair } from '../../sequence.js'
import type { Sequence } from '../../types.js'
import { distanceCutoffFor, scoreFromDistance, type MetricScoreKind } from './cutoff.js'
import type { ScorerOptions } from './options.js'

type BoundedDistance = (
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  cutoff: number,
) => number

type SequenceMaximum = (s1: ArrayLike<unknown>, s2: ArrayLike<unknown>) => number

type DirectMetric = (s1: Sequence, s2: Sequence, options?: ScorerOptions) => number

export function directMetric(
  kind: MetricScoreKind,
  distance: BoundedDistance,
  maximum: SequenceMaximum,
  unbounded: number,
): DirectMetric {
  return (s1, s2, options = {}) => {
    const [a, b] = convPair(s1, s2)
    const max = maximum(a, b)
    const budget = distanceCutoffFor(kind, options.scoreCutoff, max, unbounded)
    return scoreFromDistance(kind, distance(a, b, budget), max, options.scoreCutoff)
  }
}
