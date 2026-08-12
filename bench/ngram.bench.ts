import {
  cosineDistance,
  cosineSimilarity,
} from '../src/algorithms/cosine/implementation.js'
import { similarity as cosineMetric } from '../src/algorithms/cosine/index.js'
import { diceDistance, diceSimilarity } from '../src/algorithms/dice/implementation.js'
import { similarity as diceMetric } from '../src/algorithms/dice/index.js'
import { createMatcher, createScorer, search } from '../src/index.js'
import { pairs, similarPairs, words } from './tooling/corpus.js'
import { describe, measure } from './tooling/harness.js'

// Three length classes, as elsewhere in the suite: a profile's cost is one trie
// insertion per element, so short inputs are dominated by conversion and
// allocation while long ones are dominated by the Map traffic.
const short = similarPairs(200, 8)
const medium = similarPairs(200, 32)
const long = similarPairs(100, 128)
const veryLong = similarPairs(20, 512)
const huge = similarPairs(5, 4096)
const dissimilar = pairs(words(200, 32))
// The shape the shared profile is deliberately slower on: one gram repeated
// thousands of times is one trie node whose count is walked up, where a
// serialized-key map would have hit a single hash slot. Kept as its own case so
// that trade — much less prepared memory for this — is a number rather than a
// claim, and so a future trie change has to say what it does to it.
const repetitive = Array.from({ length: 5 }, (_value, index) => {
  const letter = String.fromCharCode(0x61 + index)
  return [letter.repeat(4096), `${letter.repeat(4095)}b`]
})
// Lengths alone put these out of reach of a high cutoff, which is the case the
// gram-count bound exists to answer without building either trie.
const lengthSkewed = words(100, 512, 0x0ba7_d101).map((value, index) =>
  index % 2 === 0 ? [value, value.slice(0, 32)] : [value.slice(0, 32), value],
)

const choices = words(1_000, 12)
const query = 'abcdefghijkl'
const queries = words(100, 12, 0x1357_9bdf)

const scorer = createScorer(diceMetric)
const cosine = createScorer(cosineMetric)
const trigrams = createScorer(diceMetric, { gramSize: 3 })
const matcher = createMatcher(choices, { scorer })
// What the prepared-choice cases measure against the raw ones: the profile of
// every candidate built once, held by the caller.
const preparedChoices = choices.map((text) => ({ prepared: scorer.prepareChoice(text) }))
const preparedCosineChoices = choices.map((text) => ({
  prepared: cosine.prepareChoice(text),
}))
// Trigrams are the other depth a caller picks by hand, and the only other one
// the query kernel flattens; everything deeper walks both tries per candidate.
const preparedTrigramChoices = choices.map((text) => ({
  prepared: trigrams.prepareChoice(text),
}))

// Every case below inlines its own loop rather than calling one shared helper,
// for the reason `bench/distance.bench.ts` gives: V8 attaches an inline cache
// to a function literal, and a shared body would measure whichever group ran
// first while its call site was still monomorphic.

describe('diceSimilarity', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) diceSimilarity(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) diceSimilarity(a, b)
  })
  measure('32 chars, unrelated', () => {
    for (const [a, b] of dissimilar) diceSimilarity(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) diceSimilarity(a, b)
  })
  measure('512 chars, similar', () => {
    for (const [a, b] of veryLong) diceSimilarity(a, b)
  })
  measure('4096 chars, similar', () => {
    for (const [a, b] of huge) diceSimilarity(a, b)
  })
  measure('4096 chars, one repeated gram', () => {
    for (const [a, b] of repetitive) diceSimilarity(a, b)
  })
  measure('trigrams, 128 chars, similar', () => {
    for (const [a, b] of long) trigrams.score(a, b)
  })
})

describe('diceDistance', () => {
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) diceDistance(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) diceDistance(a, b)
  })
})

describe('cosineSimilarity', () => {
  measure('8 chars, similar', () => {
    for (const [a, b] of short) cosineSimilarity(a, b)
  })
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) cosineSimilarity(a, b)
  })
  measure('32 chars, unrelated', () => {
    for (const [a, b] of dissimilar) cosineSimilarity(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) cosineSimilarity(a, b)
  })
  measure('512 chars, similar', () => {
    for (const [a, b] of veryLong) cosineSimilarity(a, b)
  })
  measure('4096 chars, similar', () => {
    for (const [a, b] of huge) cosineSimilarity(a, b)
  })
  measure('4096 chars, one repeated gram', () => {
    for (const [a, b] of repetitive) cosineSimilarity(a, b)
  })
})

describe('cosineDistance', () => {
  measure('32 chars, similar', () => {
    for (const [a, b] of medium) cosineDistance(a, b)
  })
  measure('128 chars, similar', () => {
    for (const [a, b] of long) cosineDistance(a, b)
  })
})

describe('dice gram-count bound', () => {
  measure('512 vs 32 chars, cutoff 0.8', () => {
    for (const [a, b] of lengthSkewed) scorer.score(a, b, { threshold: 0.8 })
  })
  measure('512 vs 32 chars, no cutoff', () => {
    for (const [a, b] of lengthSkewed) scorer.score(a, b)
  })
  measure('128 chars similar, cutoff 0.8', () => {
    for (const [a, b] of long) scorer.score(a, b, { threshold: 0.8 })
  })
  // Cosine has no bound, so this is what one costs: the same shape with both
  // profiles built every time.
  measure('cosine, 512 vs 32 chars, cutoff 0.8', () => {
    for (const [a, b] of lengthSkewed) cosine.score(a, b, { threshold: 0.8 })
  })
})

describe('dice search', () => {
  measure('1 query, 1000 choices', () => {
    search(query, choices, { scorer, limit: null })
  })
  measure('1 query, 1000 choices, matcher', () => {
    matcher.search(query, { limit: null })
  })
  measure('1 query, 1000 prepared choices', () => {
    search(query, preparedChoices, {
      scorer,
      limit: null,
      getPrepared: (row) => row.prepared,
    })
  })
  measure('100 queries, 1000 choices', () => {
    for (const each of queries) search(each, choices, { scorer, limit: null })
  })
  measure('100 queries, 1000 prepared choices', () => {
    for (const each of queries) {
      search(each, preparedChoices, {
        scorer,
        limit: null,
        getPrepared: (row) => row.prepared,
      })
    }
  })
  measure('trigrams, 100 queries, 1000 prepared choices', () => {
    for (const each of queries) {
      search(each, preparedTrigramChoices, {
        scorer: trigrams,
        limit: null,
        getPrepared: (row) => row.prepared,
      })
    }
  })
})

describe('cosine search', () => {
  measure('1 query, 1000 choices', () => {
    search(query, choices, { scorer: cosine, limit: null })
  })
  measure('100 queries, 1000 choices', () => {
    for (const each of queries) search(each, choices, { scorer: cosine, limit: null })
  })
  measure('100 queries, 1000 prepared choices', () => {
    for (const each of queries) {
      search(each, preparedCosineChoices, {
        scorer: cosine,
        limit: null,
        getPrepared: (row) => row.prepared,
      })
    }
  })
})
