export { createScorer } from './core/scorer.js'
export { isMatch, scoreIfMatch } from './core/match.js'
export { normalizeText } from './core/normalize.js'
export { bestMatch, createMatcher, search, searchIter } from './search/index.js'
export { scoreMatrix, scorePairs } from './batch/index.js'

export type { Metric } from './core/metric.js'
export type {
  CustomScorerConfiguration,
  Scorer,
  ThresholdOptions,
} from './core/scorer.js'
export type { Direction, MaybeSequence, Sequence } from './core/types.js'
export type {
  BestOptions,
  Items,
  Match,
  Matcher,
  MatcherOptions,
  MissingItemsPolicy,
  Normalizer,
  SearchOptions,
  SearchIterOptions,
} from './search/index.js'
export type {
  BatchOptions,
  ScoreArray,
  ScoreArrayKind,
  ScoreArrayOf,
  ScoreMatrix,
} from './batch/index.js'
