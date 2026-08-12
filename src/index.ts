export { createScorer } from './core/scorer.js'
export { isMatch, scoreIfMatch } from './core/match.js'
export { normalizeText } from './core/normalize.js'
export { bestMatch, createMatcher, search, searchIter } from './search/index.js'
export { scoreMatrix, scorePairs } from './batch/index.js'

export type { Metric } from './core/metric.js'
export type {
  CustomScorerConfiguration,
  PrepareChoiceOptions,
  PreparedChoiceOf,
  Scorer,
  ScorerOf,
  ThresholdOptions,
} from './core/scorer.js'
export type { PreparedChoice } from './core/prepared.js'
export type { Direction, MaybeSequence, Normalizer, Sequence } from './core/types.js'
export type {
  BestOptions,
  Match,
  Matcher,
  MatcherOptions,
  MissingItemsPolicy,
  PreparedMatcherOptions,
  SearchOptions,
} from './search/index.js'
export type {
  BatchOptions,
  ScoreArray,
  ScoreArrayKind,
  ScoreArrayOf,
  ScoreMatrix,
} from './batch/index.js'
