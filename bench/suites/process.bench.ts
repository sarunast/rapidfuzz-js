import {
  distance as damerauDistance,
  normalizedSimilarity as damerauNormalizedSimilarity,
} from '../../src/algorithms/damerauLevenshtein/index.js'
import { normalizedSimilarity as indelNormalizedSimilarity } from '../../src/algorithms/indel/index.js'
import { normalizedSimilarity as lcsNormalizedSimilarity } from '../../src/algorithms/lcs/index.js'
import {
  distance as levenshteinDistance,
  normalizedDistance as levenshteinNormalizedDistance,
  normalizedSimilarity as levenshteinNormalizedSimilarity,
  similarity as levenshteinSimilarity,
} from '../../src/algorithms/levenshtein/index.js'
import { normalizedSimilarity as osaNormalizedSimilarity } from '../../src/algorithms/osa/index.js'
import {
  weightedRatio,
  ratio as fuzzRatio,
  tokenSetRatio,
  tokenSortRatio,
} from '../../src/fuzz/index.js'
import {
  bestMatch,
  createMatcher,
  createScorer,
  normalizeText,
  scoreMatrix,
  scorePairs,
  search,
  searchIter,
} from '../../src/index.js'
import { sentences, similarPairs, words } from '../harness/corpus.js'
import { describe, measure } from '../harness/harness.js'

const choices = words(2_000, 12)
const symmetricChoices = choices.slice(0, 200)
const query = 'abcdefghijkl'
const pairQueries = choices.map(() => query)
const titles = sentences(2_000, 5)
const titleQueries = sentences(30, 5, 0x1122_3344)
const titleQuery = 'alpha bravo charlie delta echo'

const fuzzy = createScorer(fuzzRatio)
const adaptive = createScorer(weightedRatio)
const tokenSort = createScorer(tokenSortRatio)
const rawDistance = createScorer(levenshteinDistance)
const rawSimilarity = createScorer(levenshteinSimilarity)
const normalizedDistance = createScorer(levenshteinNormalizedDistance)
const normalized = createScorer(levenshteinNormalizedSimilarity)
const preparedIndel = createScorer(indelNormalizedSimilarity)
const preparedLcs = createScorer(lcsNormalizedSimilarity)
const preparedOsa = createScorer(osaNormalizedSimilarity)
const preparedDamerau = createScorer(damerauNormalizedSimilarity)
const preparedDamerauDistance = createScorer(damerauDistance)
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
    for (const choice of choices) fuzzRatio(query, choice)
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

// Every prepared case above holds a 12-character query, which is one machine
// word — so nothing here reaches the two- and three-word widths that most fuzz
// scoring actually runs at.
//
// The choices are the same length as the query on purpose. `ratioHeld` rejects
// on a length ratio before it scores anything, so a 64-character query against
// 96-character choices never reaches a kernel at all under a threshold; and its
// bounded kernel is gated on the two lengths summing to 128, which is why the
// two-word case draws 64 rather than 48.
describe('prepared queries, two and three words', () => {
  const twoWordChoices = words(2_000, 64, 0x51ed_2701)
  const threeWordChoices = words(2_000, 80, 0x51ed_2702)
  // Drawn separately rather than taken from the list. A query that is also a
  // choice scores a perfect 100 on the first candidate, `bestMatch` stops
  // there, and the case measures the early exit instead of 2000 kernel runs —
  // which it did, at a thousandth of the time the others take.
  const twoWordQuery = words(1, 64, 0x51ed_2703)[0] ?? ''
  const threeWordQuery = words(1, 80, 0x51ed_2704)[0] ?? ''

  measure('2000 64-char choices, fuzzy', () => {
    bestMatch(twoWordQuery, twoWordChoices, { scorer: fuzzy })
  })
  measure('2000 64-char choices, fuzzy threshold 80', () => {
    bestMatch(twoWordQuery, twoWordChoices, { scorer: fuzzy, threshold: 80 })
  })
  measure('2000 80-char choices, fuzzy', () => {
    bestMatch(threeWordQuery, threeWordChoices, { scorer: fuzzy })
  })
  measure('2000 80-char choices, fuzzy threshold 80', () => {
    bestMatch(threeWordQuery, threeWordChoices, { scorer: fuzzy, threshold: 80 })
  })
})

// The prepared Indel scorer, which nothing else here builds — every scorer above
// is fuzz, Levenshtein or token-sort. It is the only way into the one-word arm
// of the bounded prepared kernel: `ratioHeld` gates that kernel on the two
// lengths summing to 128, which puts the target out of reach of a pattern of 32
// elements or fewer, so no fuzz workload enters it at all. Counted rather than
// reasoned about — 2000 calls here against 0 from every fuzz case.
describe('prepared Indel, one-word queries', () => {
  measure('2000 choices, normalized threshold 0.8', () => {
    bestMatch(query, choices, { scorer: preparedIndel, threshold: 0.8 })
  })
  measure('2000 choices, normalized threshold 0.5', () => {
    bestMatch(query, choices, { scorer: preparedIndel, threshold: 0.5 })
  })
  measure('2000 titles, 30-char query, normalized threshold 0.8', () => {
    bestMatch(titleQuery, titles, { scorer: preparedIndel, threshold: 0.8 })
  })
  measure('2000 choices, search limit 5, threshold 0.7', () => {
    search(query, choices, { scorer: preparedIndel, threshold: 0.7, limit: 5 })
  })
})

// The prepared Ukkonen band, which nothing else in the suite reaches. It needs
// a held query wide enough that the band comes out narrower than the row, and
// a threshold tight enough to draw one — 128 elements is the floor and none of
// the prepared cases above is a quarter of that. Counted: 0 calls from every
// existing case in this file and from `partialRatio` under a cutoff, against
// 2000 from each of these.
//
// Both sides of the cutoff, because the band's bookkeeping is paid by every
// candidate while the abandonment is only collected from the ones that fail.
describe('prepared band, 256-char queries', () => {
  const bandChoices = words(2_000, 256, 0x51ed_2705)
  const bandQuery = words(1, 256, 0x51ed_2706)[0] ?? ''
  const bandSimilar = similarPairs(2_000, 256, 0.15, 0x51ed_2707)
  const bandSimilarChoices = bandSimilar.map(([, b]) => b)
  const bandSimilarQuery = bandSimilar[0]?.[0] ?? ''

  measure('2000 unrelated choices, fuzzy threshold 80', () => {
    bestMatch(bandQuery, bandChoices, { scorer: fuzzy, threshold: 80 })
  })
  measure('2000 similar choices, fuzzy threshold 90', () => {
    bestMatch(bandSimilarQuery, bandSimilarChoices, { scorer: fuzzy, threshold: 90 })
  })
  measure('2000 similar choices, prepared Indel threshold 0.95', () => {
    bestMatch(bandSimilarQuery, bandSimilarChoices, {
      scorer: preparedIndel,
      threshold: 0.95,
    })
  })
})

// The prepared side of the three distance families that had no coverage here.
// `prepared Indel` above is the only prepared non-Levenshtein case in the file,
// so a change to the shared distance-family policy could have moved LCS, OSA or
// Damerau with nothing to notice. Both a thresholded scan and a limited search,
// because the cutoff-to-distance conversion those families duplicate is only
// exercised when a threshold is present.
describe('prepared distance families, one-word queries', () => {
  measure('2000 choices, LCS normalized threshold 0.8', () => {
    bestMatch(query, choices, { scorer: preparedLcs, threshold: 0.8 })
  })
  measure('2000 choices, OSA normalized threshold 0.8', () => {
    bestMatch(query, choices, { scorer: preparedOsa, threshold: 0.8 })
  })
  measure('2000 choices, Damerau normalized threshold 0.8', () => {
    bestMatch(query, choices, { scorer: preparedDamerau, threshold: 0.8 })
  })
  measure('2000 choices, LCS search limit 5, threshold 0.7', () => {
    search(query, choices, { scorer: preparedLcs, threshold: 0.7, limit: 5 })
  })
  measure('2000 choices, OSA search limit 5, threshold 0.7', () => {
    search(query, choices, { scorer: preparedOsa, threshold: 0.7, limit: 5 })
  })
  // No threshold, so every candidate is scored and the conversion runs on the
  // unbounded path instead of the rejecting one.
  measure('2000 choices, Damerau raw distance, no threshold', () => {
    bestMatch(query, choices, { scorer: preparedDamerauDistance })
  })
})

// The one thing correctness tests cannot see about `OptimumProof`: whether it
// still fires. Token-set similarity scores exactly 100 when one non-empty token
// set contains the other, so `createMatcher` answers from a token index instead
// of scoring anything. A regression that made the proof always decline would
// pass every test in the suite and show up only here.
//
// The two cases are the two halves of that, and both are load-bearing:
//
//   settles  — the containing choice is placed last, so a scan pays for all
//              2000 candidates before reaching it. This is a constant-time
//              index lookup, and reads three orders of magnitude below its
//              sibling. If it ever climbs to meet the declining case, the
//              proof stopped firing.
//   declines — no choice holds both query tokens, so the proof gives up and the
//              scan runs. This is the case that says declining stays free; if
//              it moves, the proof started costing something on the corpora it
//              cannot help.
describe('token-set containment proof', () => {
  const containmentChoices = sentences(2_000, 5, 0x630f_7a11)
  // Sentinel tokens rather than words drawn from the corpus: the proof settles
  // on the *earliest* containing choice, so a query whose tokens co-occur early
  // by chance would measure a short scan instead of the shortcut.
  containmentChoices[containmentChoices.length - 1] =
    `zulu quebec ${containmentChoices[containmentChoices.length - 1] ?? ''}`
  const containedQuery = 'zulu quebec'
  const uncontainedQuery = 'sierra tango'
  const containment = createMatcher(containmentChoices, {
    scorer: createScorer(tokenSetRatio),
  })

  measure('2000 sentences, proof settles', () => {
    containment.best(containedQuery)
  })
  measure('2000 sentences, proof declines', () => {
    containment.best(uncontainedQuery)
  })
})
