/**
 * `createIndexedMatcher` over **arbitrary elements** — token rows rather than
 * text — against the exhaustive `createMatcher` path it replaces.
 *
 * Its own file rather than a group in `ngramIndex.bench.ts`, because a
 * `describe` body runs at module load: eight 10,000-record matchers built
 * before the first measurement would change the environment every case in that
 * file is measured in, and its baseline is what says a production change was
 * neutral. One file per corpus shape, and the runner gives each file its own
 * child process.
 *
 * Both arms of every pair see the same corpus and the same queries, and each
 * `measure` body inlines its own loop for the reason `bench/distance.bench.ts`
 * gives: V8 attaches an inline cache to a function literal, so a shared body
 * would measure whichever group ran first while its call site was monomorphic.
 */

import { similarity as diceMetric } from '../../src/algorithms/dice/index.js'
import { similarity as tverskyMetric } from '../../src/algorithms/tversky/index.js'
import { createIndexedMatcher, createMatcher, createScorer } from '../../src/index.js'
import { words } from '../harness/corpus.js'
import { describe, measure } from '../harness/harness.js'

// Role-separated vocabularies, so a row's tokens are unique as a *multiset* and
// not merely as a sequence. Unigram overlap is order-independent: with one
// shared vocabulary, a row holding `(skillA, skillB)` and an earlier row holding
// `(skillB, skillA)` score each other 1, and the exhaustive `best` arm settles
// on the earlier one instead of scanning. Prefixing by role makes the swap
// impossible while keeping three posting-list lengths — a common token names a
// sixth of the corpus, a secondary skill 400 rows, a primary skill 25.
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
// `(index % 400, floor(index / 400))` is unique over 10,000 rows and the two
// sit in different vocabularies, so the queried prefix names exactly one row.
// Scattered rather than taken off the front, so the position `best` has to
// reach averages half the corpus rather than the first few rows.
const recordQueries = Array.from({ length: 100 }, (_, index) =>
  records[(index * 97 + 4_099) % records.length].slice(0, 3),
)

// The direct text corpus `ngramIndex.bench.ts` builds from, so the transition
// pair below is a ratio within this process rather than across two files.
const medium = words(10_000, 24)
// 10,000 choices keyed directly and then one this scheme cannot spell, which is
// the transition at its most expensive: every posting key already recorded is
// decoded and re-encoded.
const lateTransition = [...medium, ['react', 'typescript', 'node']]

const GRAM_SIZE = 3
const dice = createScorer(diceMetric, { gramSize: GRAM_SIZE })

// Both depths: unigrams are exact token overlap, shingles are ordered pairs, and
// the two key the same postings from different ordinal grams.
const tokenDice = createScorer(diceMetric, { gramSize: 1 })
const tokenDiceShingles = createScorer(diceMetric, { gramSize: 2 })
const tokenContainment = createScorer(tverskyMetric, { gramSize: 1, alpha: 1, beta: 0 })
const tokenShingleContainment = createScorer(tverskyMetric, {
  gramSize: 2,
  alpha: 1,
  beta: 0,
})

const recordIndexed = createIndexedMatcher(records, { scorer: tokenContainment })
const recordShingleIndexed = createIndexedMatcher(records, {
  scorer: tokenShingleContainment,
})
const recordDiceIndexed = createIndexedMatcher(records, { scorer: tokenDice })
const recordDiceShingleIndexed = createIndexedMatcher(records, {
  scorer: tokenDiceShingles,
})

const recordMatcher = createMatcher(records, { scorer: tokenContainment })
const recordShingleMatcher = createMatcher(records, { scorer: tokenShingleContainment })
const recordDiceMatcher = createMatcher(records, { scorer: tokenDice })
const recordDiceShingleMatcher = createMatcher(records, { scorer: tokenDiceShingles })

describe('token index build', () => {
  // The first two are the transition pair: same corpus, same scorer, one extra
  // choice that cannot be keyed directly.
  measure('indexed, 10000 direct choices', () =>
    createIndexedMatcher(medium, { scorer: dice }),
  )
  measure('indexed, 10000 choices, late ordinal transition', () =>
    createIndexedMatcher(lateTransition, { scorer: dice }),
  )
  measure('indexed, 10000 token records', () =>
    createIndexedMatcher(records, { scorer: tokenContainment }),
  )
  measure('matcher, 10000 token records', () =>
    createMatcher(records, { scorer: tokenContainment }),
  )
})

describe('token index, 10000 records', () => {
  measure('indexed, containment, 100 queries, threshold 0.8', () => {
    for (const query of recordQueries)
      recordIndexed.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('exhaustive, containment, 100 queries, threshold 0.8', () => {
    for (const query of recordQueries)
      recordMatcher.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('indexed, containment, best, 100 queries', () => {
    for (const query of recordQueries) recordIndexed.best(query)
  })
  measure('exhaustive, containment, best, 100 queries', () => {
    for (const query of recordQueries) recordMatcher.best(query)
  })
  measure('indexed, dice, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      recordDiceIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, dice, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      recordDiceMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
})

// Shingles are far more selective than single tokens here — the pair a query
// carries is unique to one row by construction, and the pair before it recurs
// only every 1,200 — so this group is the index at its best rather than at its
// most typical. The unigram group above, where a common token names a sixth of
// the corpus, is the mixed shape.
describe('token shingle index, 10000 records', () => {
  measure('indexed, containment, 100 queries, threshold 0.8', () => {
    for (const query of recordQueries)
      recordShingleIndexed.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('exhaustive, containment, 100 queries, threshold 0.8', () => {
    for (const query of recordQueries)
      recordShingleMatcher.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('indexed, containment, best, 100 queries', () => {
    for (const query of recordQueries) recordShingleIndexed.best(query)
  })
  measure('exhaustive, containment, best, 100 queries', () => {
    for (const query of recordQueries) recordShingleMatcher.best(query)
  })
  measure('indexed, dice, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      recordDiceShingleIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, dice, 100 queries, threshold 0.5', () => {
    for (const query of recordQueries)
      recordDiceShingleMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
})
