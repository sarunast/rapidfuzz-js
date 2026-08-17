/**
 * Weighted exact-token Tversky — `elementWeights` at `gramSize: 1` — indexed
 * against the exhaustive `createMatcher` path it replaces.
 *
 * Its own file rather than a group in `ngramTokenIndex.bench.ts` for two
 * reasons: a `describe` body runs at module load, so the matchers built here
 * would change the environment every case in that file is measured in, and its
 * baseline is what says a production change was neutral. One file per corpus
 * shape, and the runner gives each file its own child process.
 *
 * Both arms of every pair see the same corpus and the same queries, and each
 * `measure` body inlines its own loop for the reason `bench/distance.bench.ts`
 * gives: V8 attaches an inline cache to a function literal, so a shared body
 * would measure whichever group ran first while its call site was monomorphic.
 */

import { similarity as tverskyMetric } from '../../src/algorithms/tversky/index.js'
import { createIndexedMatcher, createMatcher, createScorer } from '../../src/index.js'
import { words } from '../harness/corpus.js'
import { describe, measure } from '../harness/harness.js'

// The corpus `ngramTokenIndex.bench.ts` uses, for the same reason: role-prefixed
// vocabularies make a row's tokens unique as a multiset, so an exhaustive `best`
// arm has to scan rather than settle on an early duplicate. Three posting
// lengths — a common token names a sixth of the corpus, a secondary skill 400
// rows, a primary skill 25.
const SKILLS = words(400, 9, 0x30b1_0055)
const PRIMARY = SKILLS.map((word) => `primary:${word}`)
const SECONDARY = SKILLS.slice(0, 25).map((word) => `secondary:${word}`)
const COMMON = ['senior', 'engineer', 'remote', 'zurich', 'typescript', 'react']
const records = Array.from({ length: 10_000 }, (_, index) => [
  COMMON[index % COMMON.length],
  PRIMARY[index % PRIMARY.length],
  SECONDARY[Math.floor(index / 400)],
  PRIMARY[(index * 7 + 1) % PRIMARY.length],
  PRIMARY[(index * 13 + 5) % PRIMARY.length],
])
// `(index % 400, floor(index / 400))` is unique over 10,000 rows and the two sit
// in different vocabularies, so the queried prefix names exactly one row.
const recordQueries = Array.from({ length: 100 }, (_, index) =>
  records[(index * 97 + 4_099) % records.length].slice(0, 3),
)
// The tiers a caller would actually configure: distinctive skills above the
// baseline, shared vocabulary below it, a legal suffix near zero.
const TIERS = new Map<unknown, number>([
  ...PRIMARY.map((token): [unknown, number] => [token, 4]),
  ...SECONDARY.map((token): [unknown, number] => [token, 3]),
  ...COMMON.map((token): [unknown, number] => [token, 0.2]),
  ['ag', 0.05],
])
// What weight-group diversity costs, and nothing else: every corpus token weighs
// the same 2 against a default of 1, so the queries, the postings, their
// densities and the candidate set are the tiered arm's — only the group
// partition changes, from three groups a query to one. Slicing the queries
// instead would have measured posting selectivity.
const ONE_GROUP = new Map<unknown, number>(
  [...PRIMARY, ...SECONDARY, ...COMMON].map((token): [unknown, number] => [token, 2]),
)
// The same corpus with one token in every row, so its posting is dense.
const denseRecords = records.map((record) => [...record, 'ag'])
const denseQueries = recordQueries.map((query) => [...query, 'ag'])

const IGNORED_DENSE = new Map<unknown, number>([...TIERS, ['ag', 0]])
const ALL_ONE = new Map<unknown, number>(
  [...PRIMARY, ...SECONDARY, ...COMMON].map((token): [unknown, number] => [token, 1]),
)
// One positive weight group, a default inside it, and a single ignored token:
// the shape a per-score uniformity test would walk in full before it could
// answer. The pair exists so the answer's cost can be attributed to the table's
// size and nothing else — every corpus token, group assignment, query group and
// score below is identical between them, and only the entries no query or choice
// ever mentions differ.
const BASE_MIXED = new Map<unknown, number>([...ALL_ONE, ['ag', 0]])
const LARGE_MIXED = new Map<unknown, number>([
  ...BASE_MIXED,
  ...Array.from({ length: 20_000 }, (_, at): [unknown, number] => [`unused${at}`, 1]),
])

const tiered = createScorer(tverskyMetric, {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  elementWeights: TIERS,
})
const ignoredDense = createScorer(tverskyMetric, {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  elementWeights: IGNORED_DENSE,
})
const lowDense = createScorer(tverskyMetric, {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  elementWeights: TIERS,
})
const allOne = createScorer(tverskyMetric, {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  elementWeights: ALL_ONE,
})
const oneGroup = createScorer(tverskyMetric, {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  elementWeights: ONE_GROUP,
})
const baseTable = createScorer(tverskyMetric, {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  elementWeights: BASE_MIXED,
})
const largeTable = createScorer(tverskyMetric, {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  elementWeights: LARGE_MIXED,
})
const unweighted = createScorer(tverskyMetric, { gramSize: 1, alpha: 1, beta: 0.1 })

const tieredIndexed = createIndexedMatcher(records, { scorer: tiered })
const tieredMatcher = createMatcher(records, { scorer: tiered })
const oneGroupIndexed = createIndexedMatcher(records, { scorer: oneGroup })
const allOneIndexed = createIndexedMatcher(records, { scorer: allOne })
const unweightedIndexed = createIndexedMatcher(records, { scorer: unweighted })
const ignoredDenseIndexed = createIndexedMatcher(denseRecords, { scorer: ignoredDense })
const ignoredDenseMatcher = createMatcher(denseRecords, { scorer: ignoredDense })
const lowDenseIndexed = createIndexedMatcher(denseRecords, { scorer: lowDense })
const lowDenseMatcher = createMatcher(denseRecords, { scorer: lowDense })

describe('weighted token index build', () => {
  measure('indexed, 10000 weighted token records', () =>
    createIndexedMatcher(records, { scorer: tiered }),
  )
  measure('matcher, 10000 weighted token records', () =>
    createMatcher(records, { scorer: tiered }),
  )
  measure('indexed, 10000 unweighted token records', () =>
    createIndexedMatcher(records, { scorer: unweighted }),
  )
})

describe('weighted tiers, 10000 records', () => {
  measure('indexed, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      tieredIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      tieredMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, best, 100 queries', () => {
    for (const query of recordQueries) tieredIndexed.best(query)
  })
  measure('exhaustive, best, 100 queries', () => {
    for (const query of recordQueries) tieredMatcher.best(query)
  })
})

// A token in every row, weighing nothing: it must stay out of the traversal
// entirely, or one semantically worthless token turns every query into a
// whole-corpus scan.
describe('dense ignored token, 10000 records', () => {
  measure('indexed, 100 queries, threshold 0.5', () => {
    for (const query of denseQueries)
      ignoredDenseIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, 100 queries, threshold 0.5', () => {
    for (const query of denseQueries)
      ignoredDenseMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
})

// The same token weighing a little, which is the real dense weighted cost: every
// candidate is credited and every absence corrected.
describe('dense low-weight token, 10000 records', () => {
  measure('indexed, 100 queries, threshold 0.5', () => {
    for (const query of denseQueries)
      lowDenseIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, 100 queries, threshold 0.5', () => {
    for (const query of denseQueries)
      lowDenseMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
})

// What the weighted machinery costs against the same work unweighted. The
// all-weights-1 arm is the semantically empty one — one constant factor cancels
// from the ratio — so the scorer drops the weighting at compile time and this
// case says whether that detection holds. `one weight group` is the genuinely
// weighted arm over the same postings, which is what the representation costs.
describe('weighted machinery over unweighted work', () => {
  measure('indexed, all weights 1, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      allOneIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, one weight group, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      oneGroupIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, unweighted, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      unweightedIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('scorer.score, weighted, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += tiered.score(recordQueries[index % 100], records[index])
    }
    return total
  })
  measure('scorer.score, base weight table, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += baseTable.score(recordQueries[index % 100], records[index])
    }
    return total
  })
  measure('scorer.score, base table + 20000 unused weights, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += largeTable.score(recordQueries[index % 100], records[index])
    }
    return total
  })
  measure('scorer.score, all weights 1, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += allOne.score(recordQueries[index % 100], records[index])
    }
    return total
  })
  measure('scorer.score, unweighted, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += unweighted.score(recordQueries[index % 100], records[index])
    }
    return total
  })
})
