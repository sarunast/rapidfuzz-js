export { createScorer } from './core/scoring/scorer.js'
export { isMatch, scoreIfMatch } from './core/scoring/match.js'
export { normalizeText } from './core/normalize.js'
export {
  bestMatch,
  createIndexedMatcher,
  createMatcher,
  search,
  searchIter,
} from './search/index.js'
export { scoreMatrix } from './batch/scoreMatrix.js'
export { scorePairs } from './batch/scorePairs.js'

export type { Metric } from './core/scoring/metric.js'
export type {
  CustomScorerConfiguration,
  PrepareChoiceOptions,
  PreparedChoiceOf,
  Scorer,
  ScorerOf,
  ThresholdOptions,
} from './core/scoring/scorer.js'
export type { PreparedChoice } from './core/scoring/preparedChoice.js'
export type { Direction, MaybeSequence, Normalizer, Sequence } from './core/types.js'
export type {
  BestOptions,
  IndexedMatcherOptions,
  Match,
  Matcher,
  MatcherOptions,
  MissingItemsPolicy,
  PreparedMatcherOptions,
  SearchOptions,
} from './search/index.js'
export type { BatchOptions } from './batch/options.js'
export type { ScoreMatrix } from './batch/scoreMatrix.js'
export type { ScoreArray, ScoreArrayKind, ScoreArrayOf } from './batch/storage.js'
