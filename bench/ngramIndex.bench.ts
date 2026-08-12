/**
 * Indexed n-gram search against the exhaustive Matcher path it would replace.
 *
 * The prototype lives in `bench/tooling/ngramIndex.ts` and is bench-only: it is
 * an experiment in whether one corpus-wide inverted representation can replace
 * the N prepared tries a Dice/Cosine Matcher retains. Correctness, structural
 * counters and retained memory are `bench/tooling/ngram-index-scale.ts`; this
 * file is the part that needs the harness's adaptive sampling.
 *
 * Both arms of every pair see the same corpus and the same queries, and the
 * indexed arm builds its query profile inside the measured body, because a real
 * query would.
 */

import { similarity as cosineMetric } from '../src/algorithms/cosine/index.js'
import { similarity as diceMetric } from '../src/algorithms/dice/index.js'
import { buildProfile } from '../src/algorithms/shared/ngram.js'
import { createMatcher, createScorer, search } from '../src/index.js'
import { sentences, words } from './tooling/corpus.js'
import { describe, measure } from './tooling/harness.js'
import { NGramIndex } from './tooling/ngramIndex.js'

const GRAM_SIZE = 3

const dice = createScorer(diceMetric, { gramSize: GRAM_SIZE })
const cosine = createScorer(cosineMetric, { gramSize: GRAM_SIZE })

function indexOf(choices: readonly string[]): NGramIndex {
  const index = new NGramIndex(GRAM_SIZE, choices.length)
  for (let id = 0; id < choices.length; id++) {
    index.add(id, buildProfile(choices[id], GRAM_SIZE))
  }
  index.compact()
  return index
}

// Two shapes rather than one: 24 random letters give a near-flat gram
// distribution, while sentences of short words repeat theirs, which is what
// lengthens posting lists. The scale script carries the degenerate two-letter
// corpus, where this representation is known to lose.
const small = words(1_000, 24)
const medium = words(10_000, 24)
const large = words(100_000, 24, 0x0ba7_d101)
const phrases = sentences(10_000, 4)

const smallQueries = words(100, 24, 0x1357_9bdf)
const mediumQueries = medium.slice(0, 100)
const phraseQueries = phrases.slice(0, 100)
const oneQuery = large[50_000]

const smallIndex = indexOf(small)
const mediumIndex = indexOf(medium)
const largeIndex = indexOf(large)
const phraseIndex = indexOf(phrases)

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
  // The indexed arm pays for every profile it builds and then drops, because
  // that is what `createMatcher` pays to retain them.
  measure('index, 1000 choices', () => indexOf(small))
  measure('matcher, 1000 choices', () => createMatcher(small, { scorer: dice }))
  measure('index, 10000 choices', () => indexOf(medium))
  measure('matcher, 10000 choices', () => createMatcher(medium, { scorer: dice }))
})

describe('dice search, 1000 choices', () => {
  measure('indexed, 100 queries, threshold 0.5', () => {
    for (const query of smallQueries) {
      smallIndex.diceSearch(buildProfile(query, GRAM_SIZE), 0.5, 5)
    }
  })
  measure('exhaustive, 100 queries, threshold 0.5', () => {
    for (const query of smallQueries)
      smallMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, 100 queries, threshold 0.8', () => {
    for (const query of smallQueries) {
      smallIndex.diceSearch(buildProfile(query, GRAM_SIZE), 0.8, 5)
    }
  })
  measure('exhaustive, 100 queries, threshold 0.8', () => {
    for (const query of smallQueries)
      smallMatcher.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('indexed, best, 100 queries', () => {
    for (const query of smallQueries) {
      smallIndex.diceBest(buildProfile(query, GRAM_SIZE), null)
    }
  })
  measure('exhaustive, best, 100 queries', () => {
    for (const query of smallQueries) smallMatcher.best(query)
  })
})

describe('dice search, 10000 choices', () => {
  measure('indexed, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries) {
      mediumIndex.diceSearch(buildProfile(query, GRAM_SIZE), 0.5, 5)
    }
  })
  measure('exhaustive, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries) {
      mediumMatcher.search(query, { limit: 5, threshold: 0.5 })
    }
  })
  measure('indexed, 100 misses, threshold 0.5', () => {
    for (const query of smallQueries) {
      mediumIndex.diceSearch(buildProfile(query, GRAM_SIZE), 0.5, 5)
    }
  })
  measure('exhaustive, 100 misses, threshold 0.5', () => {
    for (const query of smallQueries) {
      mediumMatcher.search(query, { limit: 5, threshold: 0.5 })
    }
  })
  measure('indexed, best, 100 hits', () => {
    for (const query of mediumQueries) {
      mediumIndex.diceBest(buildProfile(query, GRAM_SIZE), null)
    }
  })
  measure('exhaustive, best, 100 hits', () => {
    for (const query of mediumQueries) mediumMatcher.best(query)
  })
})

describe('cosine search, 10000 choices', () => {
  measure('indexed, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries) {
      mediumIndex.cosineSearch(buildProfile(query, GRAM_SIZE), 0.5, 5)
    }
  })
  measure('exhaustive, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries) {
      mediumCosineMatcher.search(query, { limit: 5, threshold: 0.5 })
    }
  })
})

// Sentences repeat their words, so posting lists are longer here than in the
// random-letter corpora — the shape that decides whether posting traffic or
// candidate count dominates.
describe('dice search, 10000 sentences', () => {
  measure('indexed, 100 hits, threshold 0.5', () => {
    for (const query of phraseQueries) {
      phraseIndex.diceSearch(buildProfile(query, GRAM_SIZE), 0.5, 5)
    }
  })
  measure('exhaustive, 100 hits, threshold 0.5', () => {
    for (const query of phraseQueries) {
      phraseMatcher.search(query, { limit: 5, threshold: 0.5 })
    }
  })
})

// One query, not a hundred: the exhaustive arm is tens of milliseconds per
// query at this size, and a hundred of them would make one sample longer than
// the whole rest of the file.
describe('dice search, 100000 choices', () => {
  measure('indexed, 1 query, threshold 0.5', () =>
    largeIndex.diceSearch(buildProfile(oneQuery, GRAM_SIZE), 0.5, 5),
  )
  measure('exhaustive, 1 query, threshold 0.5', () =>
    largeMatcher.search(oneQuery, { limit: 5, threshold: 0.5 }),
  )
})

// The contamination control: the same case as `bench/ngram.bench.ts > dice
// search > 100 queries, 1000 prepared choices`, on the same corpus. An index
// changes what this process allocates, so the shared `Map.get` sites here can go
// polymorphic in a way they never do in that file. If this case reads materially
// slower than its twin, the time comparisons above are contaminated and only the
// scale script's numbers can be trusted.
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
