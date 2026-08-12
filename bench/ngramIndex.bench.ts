/**
 * `createIndexedMatcher` against the exhaustive `createMatcher` path it
 * replaces.
 *
 * This measured the bench-only prototype until the representation shipped;
 * it now measures the real thing, through the public API, with the query
 * conversion inside the timed body because a real query pays for it.
 *
 * Both arms of every pair see the same corpus and the same queries.
 *
 * **Mixed id widths, deliberately.** The 10,000-choice indexes store posting
 * ids in a `Uint16Array` and the 100,000-choice one in a `Uint32Array`, and
 * both are alive in this process at once — which the Stage B harness never
 * did, because it was comparing the two widths and had to keep each load site
 * monomorphic. Production has no such luxury: an application can hold indexes
 * of both sizes, so this file is where that costs something if it ever does.
 */

import { similarity as cosineMetric } from '../src/algorithms/cosine/index.js'
import { similarity as diceMetric } from '../src/algorithms/dice/index.js'
import {
  createIndexedMatcher,
  createMatcher,
  createScorer,
  search,
} from '../src/index.js'
import { sentences, words } from './tooling/corpus.js'
import { describe, measure } from './tooling/harness.js'

const GRAM_SIZE = 3

const dice = createScorer(diceMetric, { gramSize: GRAM_SIZE })
const cosine = createScorer(cosineMetric, { gramSize: GRAM_SIZE })

// Two shapes rather than one: 24 random letters give a near-flat gram
// distribution, while sentences of short words repeat theirs, which is what
// lengthens posting lists.
const small = words(1_000, 24)
const medium = words(10_000, 24)
const large = words(100_000, 24, 0x0ba7_d101)
const phrases = sentences(10_000, 4)

const smallQueries = words(100, 24, 0x1357_9bdf)
const mediumQueries = medium.slice(0, 100)
const phraseQueries = phrases.slice(0, 100)
const oneQuery = large[50_000]

const smallIndexed = createIndexedMatcher(small, { scorer: dice })
const mediumIndexed = createIndexedMatcher(medium, { scorer: dice })
const largeIndexed = createIndexedMatcher(large, { scorer: dice })
const phraseIndexed = createIndexedMatcher(phrases, { scorer: dice })
const mediumCosineIndexed = createIndexedMatcher(medium, { scorer: cosine })

const smallMatcher = createMatcher(small, { scorer: dice })
const mediumMatcher = createMatcher(medium, { scorer: dice })
const largeMatcher = createMatcher(large, { scorer: dice })
const phraseMatcher = createMatcher(phrases, { scorer: dice })
const mediumCosineMatcher = createMatcher(medium, { scorer: cosine })

// The control's own corpus, identical to the case it mirrors in
// `bench/ngram.bench.ts`.
const controlChoices = words(1_000, 12)
const controlQueries = words(100, 12, 0x1357_9bdf)
const controlPrepared = controlChoices.map((text) => ({
  prepared: createScorer(diceMetric).prepareChoice(text),
}))

// Every case inlines its own loop rather than calling one shared helper, for the
// reason `bench/distance.bench.ts` gives: V8 attaches an inline cache to a
// function literal, and a shared body would measure whichever group ran first
// while its call site was still monomorphic.

describe('ngram index build', () => {
  measure('indexed, 1000 choices', () => createIndexedMatcher(small, { scorer: dice }))
  measure('matcher, 1000 choices', () => createMatcher(small, { scorer: dice }))
  measure('indexed, 10000 choices', () => createIndexedMatcher(medium, { scorer: dice }))
  measure('matcher, 10000 choices', () => createMatcher(medium, { scorer: dice }))
})

describe('dice search, 1000 choices', () => {
  measure('indexed, 100 queries, threshold 0.5', () => {
    for (const query of smallQueries)
      smallIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, 100 queries, threshold 0.5', () => {
    for (const query of smallQueries)
      smallMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, 100 queries, threshold 0.8', () => {
    for (const query of smallQueries)
      smallIndexed.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('exhaustive, 100 queries, threshold 0.8', () => {
    for (const query of smallQueries)
      smallMatcher.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('indexed, best, 100 queries', () => {
    for (const query of smallQueries) smallIndexed.best(query)
  })
  measure('exhaustive, best, 100 queries', () => {
    for (const query of smallQueries) smallMatcher.best(query)
  })
})

describe('dice search, 10000 choices', () => {
  measure('indexed, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries)
      mediumIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries)
      mediumMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, 100 misses, threshold 0.5', () => {
    for (const query of smallQueries)
      mediumIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, 100 misses, threshold 0.5', () => {
    for (const query of smallQueries)
      mediumMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, best, 100 hits', () => {
    for (const query of mediumQueries) mediumIndexed.best(query)
  })
  measure('exhaustive, best, 100 hits', () => {
    for (const query of mediumQueries) mediumMatcher.best(query)
  })
  // `searchIter` is the member an index changes the shape of: it settles the
  // whole qualifying set before yielding where the exhaustive path scores as it
  // goes, so this pair is a genuine comparison rather than a formality.
  measure('indexed, searchIter, 100 hits', () => {
    for (const query of mediumQueries) {
      for (const match of mediumIndexed.searchIter(query, { threshold: 0.5 })) {
        if (match.score < 0) throw new Error('unreachable')
      }
    }
  })
  measure('exhaustive, searchIter, 100 hits', () => {
    for (const query of mediumQueries) {
      for (const match of mediumMatcher.searchIter(query, { threshold: 0.5 })) {
        if (match.score < 0) throw new Error('unreachable')
      }
    }
  })
})

describe('cosine search, 10000 choices', () => {
  measure('indexed, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries)
      mediumCosineIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries)
      mediumCosineMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
})

// Sentences repeat their words, so posting lists are longer here than in the
// random-letter corpora — the shape that decides whether posting traffic or
// candidate count dominates.
describe('dice search, 10000 sentences', () => {
  measure('indexed, 100 hits, threshold 0.5', () => {
    for (const query of phraseQueries)
      phraseIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, 100 hits, threshold 0.5', () => {
    for (const query of phraseQueries)
      phraseMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, 100 hits, threshold 0.8', () => {
    for (const query of phraseQueries)
      phraseIndexed.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('exhaustive, 100 hits, threshold 0.8', () => {
    for (const query of phraseQueries)
      phraseMatcher.search(query, { limit: 5, threshold: 0.8 })
  })
})

// One query, not a hundred: the exhaustive arm is tens of milliseconds per
// query at this size, and a hundred of them would make one sample longer than
// the whole rest of the file. This is also the only group whose index stores
// 32-bit posting ids, against the 16-bit ones every group above shares.
describe('dice search, 100000 choices', () => {
  measure('indexed, 1 query, threshold 0.5', () =>
    largeIndexed.search(oneQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('exhaustive, 1 query, threshold 0.5', () =>
    largeMatcher.search(oneQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('indexed, 1 query, threshold 0.8', () =>
    largeIndexed.search(oneQuery, { limit: 5, threshold: 0.8 }),
  )
  measure('exhaustive, 1 query, threshold 0.8', () =>
    largeMatcher.search(oneQuery, { limit: 5, threshold: 0.8 }),
  )
})

// The contamination control: the same case as `bench/ngram.bench.ts > dice
// search > 100 queries, 1000 prepared choices`, on the same corpus. An index
// changes what this process allocates, so the shared `Map.get` sites here can go
// polymorphic in a way they never do in that file. If this case reads materially
// slower than its twin, the time comparisons above are contaminated.
describe('control, matches bench/ngram.bench.ts', () => {
  const controlScorer = createScorer(diceMetric)
  measure('100 queries, 1000 prepared choices', () => {
    for (const each of controlQueries) {
      search(each, controlPrepared, {
        scorer: controlScorer,
        limit: null,
        getPrepared: (row) => row.prepared,
      })
    }
  })
})
