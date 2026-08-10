// Pure re-export barrel. Safe for tree-shaking because the package declares
// "sideEffects": false and no module here does work at import time.
export { configure } from './configure.js'
export { isMatch, matchScore, type MatchOptions } from './match.js'
export type { ConfiguredScorer, ScorerConfig } from './configure.js'
export type {
  MaybeSequence,
  NormalizedScorer,
  Processor,
  Scorer,
  ScorerFlags,
  ScorerOptions,
  Sequence,
} from './_common.js'

export {
  damerauLevenshteinDistance,
  damerauLevenshteinNormalizedDistance,
  damerauLevenshteinNormalizedSimilarity,
  damerauLevenshteinSimilarity,
} from './distance/damerauLevenshtein.js'
export {
  hammingDistance,
  hammingNormalizedDistance,
  hammingNormalizedSimilarity,
  hammingSimilarity,
  type HammingOptions,
} from './distance/hamming.js'
export {
  indelDistance,
  indelNormalizedDistance,
  indelNormalizedSimilarity,
  indelSimilarity,
} from './distance/indel.js'
export {
  jaroDistance,
  jaroNormalizedDistance,
  jaroNormalizedSimilarity,
  jaroSimilarity,
} from './distance/jaro.js'
export {
  jaroWinklerDistance,
  jaroWinklerNormalizedDistance,
  jaroWinklerNormalizedSimilarity,
  jaroWinklerSimilarity,
  type JaroWinklerOptions,
} from './distance/jaroWinkler.js'
export {
  lcsSeqDistance,
  lcsSeqNormalizedDistance,
  lcsSeqNormalizedSimilarity,
  lcsSeqSimilarity,
} from './distance/lcsSeq.js'
export {
  levenshteinDistance,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
  type LevenshteinEditopsOptions,
  type LevenshteinOptions,
  type LevenshteinWeights,
} from './distance/levenshtein.js'
export {
  osaDistance,
  osaNormalizedDistance,
  osaNormalizedSimilarity,
  osaSimilarity,
} from './distance/osa.js'
export {
  postfixDistance,
  postfixNormalizedDistance,
  postfixNormalizedSimilarity,
  postfixSimilarity,
} from './distance/postfix.js'
export {
  prefixDistance,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
  prefixSimilarity,
} from './distance/prefix.js'
export {
  partialRatio,
  partialRatioAlignment,
  partialTokenRatio,
  partialTokenSetRatio,
  partialTokenSortRatio,
  qRatio,
  ratio,
  tokenRatio,
  tokenSetRatio,
  tokenSortRatio,
  wRatio,
  type FuzzInput,
  type FuzzOptions,
  type ScoreAlignment,
} from './fuzz.js'
export {
  extract,
  extractIter,
  extractOne,
  scoreMatrix,
  scorePairs,
  type Choices,
  type ExtractOptions,
  type ExtractResult,
  type SearchOptions,
  type SearchScorer,
  type ScoreOptions,
} from './search.js'
export type {
  ScoreArray,
  ScoreArrayKind,
  ScoreArrayOf,
  ScoreMatrix,
} from './_scoreArray.js'
export {
  Editops,
  Opcodes,
  type Editop,
  type EditopTag,
  type MatchingBlock,
  type Opcode,
  type OpcodeTag,
} from './distance/editops.js'
export { hammingEditops, hammingOpcodes } from './distance/hamming.js'
export { indelEditops, indelOpcodes } from './distance/indel.js'
export { lcsSeqEditops, lcsSeqOpcodes } from './distance/lcsSeq.js'
export { levenshteinEditops, levenshteinOpcodes } from './distance/levenshtein.js'
export { defaultProcess } from './utils.js'
