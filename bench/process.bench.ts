import {
  distance as levenshteinDistance,
  normalizedDistance as levenshteinNormalizedDistance,
  normalizedSimilarity as levenshteinNormalizedSimilarity,
  similarity as levenshteinSimilarity,
} from '../src/algorithms/levenshtein/index.js'
import {
  weightedSimilarity,
  similarity as fuzzSimilarity,
  tokenSortSimilarity,
} from '../src/fuzz/index.js'
import {
  bestMatch,
  createMatcher,
  createScorer,
  normalizeText,
  scoreMatrix,
  scorePairs,
  search,
  searchIter,
} from '../src/index.js'
import { sentences, words } from './tooling/corpus.js'
import { describe, measure } from './tooling/harness.js'

const choices = words(2_000, 12)
const symmetricChoices = choices.slice(0, 200)
const query = 'abcdefghijkl'
const pairQueries = choices.map(() => query)
const titles = sentences(2_000, 5)
const titleQueries = sentences(30, 5, 0x1122_3344)
const titleQuery = 'alpha bravo charlie delta echo'

const fuzzy = createScorer(fuzzSimilarity)
const adaptive = createScorer(weightedSimilarity)
const tokenSort = createScorer(tokenSortSimilarity)
const rawDistance = createScorer(levenshteinDistance)
const rawSimilarity = createScorer(levenshteinSimilarity)
const normalizedDistance = createScorer(levenshteinNormalizedDistance)
const normalized = createScorer(levenshteinNormalizedSimilarity)
const asymmetricNormalized = createScorer(levenshteinNormalizedSimilarity, {
  weights: { insertion: 1, deletion: 2, substitution: 1 },
})
const fuzzyMatcher = createMatcher(choices, { scorer: fuzzy })
const normalizedMatcher = createMatcher(choices, { scorer: normalized })
const titleMatcher = createMatcher(titles, {
  scorer: tokenSort,
  normalize: normalizeText,
})
// What the prepared-choice cases measure against their Matcher siblings: the
// same amortized preparation, held by the caller instead of by a snapshot.
const preparedChoices = choices.map((text) => ({ prepared: fuzzy.prepareChoice(text) }))
// Library-managed normalization, because the search below names a normalizer:
// a handle that normalized its own text records nothing, and the two sides are
// refused rather than scored against each other.
const preparedTitles = titles.map((text) => ({
  prepared: tokenSort.prepareChoice(text, { normalize: normalizeText }),
}))

describe('direct Metric and Scorer calls', () => {
  measure('2000 pairs, fuzzy metric', () => {
    for (const choice of choices) fuzzSimilarity(query, choice)
  })
  measure('2000 pairs, fuzzy scorer', () => {
    for (const choice of choices) fuzzy.score(query, choice)
  })
  measure('2000 pairs, fuzzy scorer threshold 80', () => {
    for (const choice of choices) fuzzy.score(query, choice, { threshold: 80 })
  })
  measure('2000 pairs, normalized metric', () => {
    for (const choice of choices) levenshteinNormalizedSimilarity(query, choice)
  })
  measure('2000 pairs, normalized scorer', () => {
    for (const choice of choices) normalized.score(query, choice)
  })
  measure('2000 pairs, normalized scorer threshold 0.8', () => {
    for (const choice of choices) normalized.score(query, choice, { threshold: 0.8 })
  })
  measure('2000 pairs, raw distance metric', () => {
    for (const choice of choices) levenshteinDistance(query, choice)
  })
  measure('2000 pairs, raw similarity metric', () => {
    for (const choice of choices) levenshteinSimilarity(query, choice)
  })
  measure('2000 pairs, normalized distance metric', () => {
    for (const choice of choices) levenshteinNormalizedDistance(query, choice)
  })
})

describe('bestMatch, one query', () => {
  measure('2000 choices, fuzzy', () => {
    bestMatch(query, choices, { scorer: fuzzy })
  })
  measure('2000 choices, fuzzy threshold 80', () => {
    bestMatch(query, choices, { scorer: fuzzy, threshold: 80 })
  })
  measure('2000 choices, normalized', () => {
    bestMatch(query, choices, { scorer: normalized })
  })
  measure('2000 titles, adaptive fuzzy + normalize', () => {
    bestMatch(titleQuery, titles, {
      scorer: adaptive,
      normalize: normalizeText,
    })
  })
})

describe('search, one query', () => {
  measure('2000 choices, fuzzy limit 1', () => {
    search(query, choices, { scorer: fuzzy, limit: 1 })
  })
  measure('2000 choices, fuzzy limit 5', () => {
    search(query, choices, { scorer: fuzzy, limit: 5 })
  })
  measure('2000 choices, normalized limit 5', () => {
    search(query, choices, { scorer: normalized, limit: 5 })
  })
  measure('2000 titles, token sort limit 5', () => {
    search(titleQuery, titles, { scorer: tokenSort, limit: 5 })
  })
  measure('2000 choices, fuzzy unlimited', () => {
    search(query, choices, { scorer: fuzzy, limit: null })
  })
})

describe('searchIter, one query', () => {
  measure('2000 choices, fuzzy first 5', () => {
    let count = 0
    for (const _match of searchIter(query, choices, { scorer: fuzzy })) {
      if (++count === 5) break
    }
  })
  measure('2000 choices, fuzzy full', () => {
    for (const _match of searchIter(query, choices, { scorer: fuzzy })) {
      // Consume the iterator without retaining its results.
    }
  })
})

describe('createMatcher construction', () => {
  measure('2000 choices, fuzzy', () => {
    createMatcher(choices, { scorer: fuzzy })
  })
  measure('2000 choices, normalized', () => {
    createMatcher(choices, { scorer: normalized })
  })
  measure('2000 titles, token sort + normalize', () => {
    createMatcher(titles, { scorer: tokenSort, normalize: normalizeText })
  })
})

describe('repeated Matcher queries', () => {
  measure('30 x 2000, fuzzy bestMatch one-shot', () => {
    for (const value of titleQueries) bestMatch(value, choices, { scorer: fuzzy })
  })
  measure('30 x 2000, fuzzy bestMatch prepared one-shot', () => {
    for (const value of titleQueries) {
      bestMatch(value, preparedChoices, {
        scorer: fuzzy,
        getPrepared: (row) => row.prepared,
      })
    }
  })
  measure('30 x 2000, fuzzy Matcher best', () => {
    for (const value of titleQueries) fuzzyMatcher.best(value)
  })
  measure('30 x 2000, normalized Matcher best', () => {
    for (const value of titleQueries) normalizedMatcher.best(value)
  })
  measure('30 x 2000, token sort search one-shot', () => {
    for (const value of titleQueries) {
      search(value, titles, { scorer: tokenSort, normalize: normalizeText, limit: 5 })
    }
  })
  measure('30 x 2000, token sort search prepared one-shot', () => {
    for (const value of titleQueries) {
      search(value, preparedTitles, {
        scorer: tokenSort,
        getPrepared: (row) => row.prepared,
        normalize: normalizeText,
        limit: 5,
      })
    }
  })
  measure('30 x 2000, token sort Matcher search', () => {
    for (const value of titleQueries) titleMatcher.search(value, { limit: 5 })
  })
  measure('30 x 2000, fuzzy Matcher search limit 1', () => {
    for (const value of titleQueries) fuzzyMatcher.search(value, { limit: 1 })
  })
  measure('30 x 2000, fuzzy Matcher searchIter', () => {
    for (const value of titleQueries) {
      for (const _match of fuzzyMatcher.searchIter(value, { threshold: 50 })) {
        // Consume source-order matches.
      }
    }
  })
})

describe('scorePairs with explicit Scorer', () => {
  measure('2000 pairs, fuzzy', () => {
    scorePairs(pairQueries, choices, { scorer: fuzzy })
  })
  measure('2000 pairs, normalized', () => {
    scorePairs(pairQueries, choices, { scorer: normalized })
  })
  measure('2000 pairs, fuzzy + normalize', () => {
    scorePairs(pairQueries, choices, { scorer: fuzzy, normalize: normalizeText })
  })
  measure('2000 pairs, normalized threshold', () => {
    scorePairs(pairQueries, choices, { scorer: normalized, threshold: 0.8 })
  })
  measure('2000 pairs, normalized multiplier', () => {
    scorePairs(pairQueries, choices, { scorer: normalized, scoreMultiplier: 100 })
  })
  measure('2000 pairs, raw distance', () => {
    scorePairs(pairQueries, choices, { scorer: rawDistance })
  })
  measure('2000 pairs, raw similarity', () => {
    scorePairs(pairQueries, choices, { scorer: rawSimilarity })
  })
  measure('2000 pairs, normalized distance', () => {
    scorePairs(pairQueries, choices, { scorer: normalizedDistance })
  })
})

describe('scoreMatrix with explicit Scorer', () => {
  measure('30 x 2000, fuzzy', () => {
    scoreMatrix(titleQueries, choices, { scorer: fuzzy })
  })
  measure('30 x 2000, normalized', () => {
    scoreMatrix(titleQueries, choices, { scorer: normalized })
  })
  measure('30 x 2000, normalized threshold', () => {
    scoreMatrix(titleQueries, choices, { scorer: normalized, threshold: 0.8 })
  })
  measure('30 x 2000, normalized multiplier', () => {
    scoreMatrix(titleQueries, choices, { scorer: normalized, scoreMultiplier: 100 })
  })
  measure('200 x 200, symmetric normalized', () => {
    scoreMatrix(symmetricChoices, symmetricChoices, { scorer: normalized })
  })
  measure('30 x 2000, asymmetric normalized', () => {
    scoreMatrix(titleQueries, choices, { scorer: asymmetricNormalized })
  })
})
