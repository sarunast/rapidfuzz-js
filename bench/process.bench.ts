import { describe } from 'vitest'

import { similarity as levenshteinSimilarity } from '../src/algorithms/levenshtein/index.js'
import {
  fuzzySimilarity,
  similarity as fuzzSimilarity,
  tokenSortSimilarity,
} from '../src/fuzz/index.js'
import {
  bestMatch,
  createMatcher,
  createScorer,
  normalizeText,
  scoreMatrix,
  search,
} from '../src/index.js'
import { sentences, words } from './_corpus.js'
import { measure } from './_harness.js'

const choices = words(2_000, 12)
const query = 'abcdefghijkl'
const titles = sentences(2_000, 5)
const titleQueries = sentences(30, 5, 0x1122_3344)
const titleQuery = 'alpha bravo charlie delta echo'

const fuzzy = createScorer(fuzzSimilarity)
const adaptive = createScorer(fuzzySimilarity)
const tokenSort = createScorer(tokenSortSimilarity)
const normalized = createScorer(levenshteinSimilarity)
const fuzzyMatcher = createMatcher(choices, { scorer: fuzzy })
const normalizedMatcher = createMatcher(choices, { scorer: normalized })
const titleMatcher = createMatcher(titles, {
  scorer: tokenSort,
  normalize: normalizeText,
})

describe('direct Metric and Scorer calls', () => {
  measure('2000 pairs, fuzzy metric', () => {
    for (const choice of choices) fuzzSimilarity(query, choice)
  })
  measure('2000 pairs, fuzzy scorer', () => {
    for (const choice of choices) fuzzy.score(query, choice)
  })
  measure('2000 pairs, normalized metric', () => {
    for (const choice of choices) levenshteinSimilarity(query, choice)
  })
  measure('2000 pairs, normalized scorer', () => {
    for (const choice of choices) normalized.score(query, choice)
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
  measure('2000 choices, fuzzy limit 5', () => {
    search(query, choices, { scorer: fuzzy, limit: 5 })
  })
  measure('2000 choices, normalized limit 5', () => {
    search(query, choices, { scorer: normalized, limit: 5 })
  })
  measure('2000 titles, token sort limit 5', () => {
    search(titleQuery, titles, { scorer: tokenSort, limit: 5 })
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
  measure('30 x 2000, fuzzy best', () => {
    for (const value of titleQueries) fuzzyMatcher.best(value)
  })
  measure('30 x 2000, normalized best', () => {
    for (const value of titleQueries) normalizedMatcher.best(value)
  })
  measure('30 x 2000, token sort search', () => {
    for (const value of titleQueries) titleMatcher.search(value, { limit: 5 })
  })
})

describe('scoreMatrix with explicit Scorer', () => {
  measure('30 x 2000, fuzzy', () => {
    scoreMatrix(titleQueries, choices, { scorer: fuzzy })
  })
  measure('30 x 2000, normalized', () => {
    scoreMatrix(titleQueries, choices, { scorer: normalized })
  })
})
