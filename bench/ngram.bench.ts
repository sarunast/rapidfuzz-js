import {
  cosineDistance,
  cosineSimilarity,
} from '../src/algorithms/cosine/implementation.js'
import { similarity as cosineMetric } from '../src/algorithms/cosine/index.js'
import { diceDistance, diceSimilarity } from '../src/algorithms/dice/implementation.js'
import { similarity as diceMetric } from '../src/algorithms/dice/index.js'
import { bestMatch, createMatcher, createScorer, search } from '../src/index.js'
import { editedPairs, pairs, similarPairs, words } from './tooling/corpus.js'
import { describe, measure } from './tooling/harness.js'

// Three length classes, as elsewhere in the suite: a profile's cost is one trie
// insertion per element, so short inputs are dominated by conversion and
// allocation while long ones are dominated by the Map traffic.
// Below the point where a packed profile's fixed cost — two typed arrays and a
// sort to hold a handful of grams — is paid back. Measured against the trie
// build it replaced: 1.07x slower at four characters, 1.29x faster at six, so
// the crossover sits between them and these two cases are what holds it there.
const tiny = similarPairs(200, 4)
const small = similarPairs(200, 6)
const short = similarPairs(200, 8)
const medium = similarPairs(200, 32)
const long = similarPairs(100, 128)
const veryLong = similarPairs(20, 512)
// The other end of the same trade, and where the packed representation used to
// lose: sorting every gram is `O(n log n)` where inserting into a trie is
// linear, and a one-shot comparison has no later query to amortise it over.
// Against the trie build it was level at 512 characters, 1.1x slower at 1024,
// 1.5x at 4096 and 1.6x at 8192 — which is what the transient direct counter in
// `dice/implementation.ts` was written to answer, and these are the lengths it
// answers them at. A prepared choice still sorts, once, and wins on every query
// after it, which is why the stored representation is not gated on length.
const large = similarPairs(10, 1024)
const huge = similarPairs(5, 4096)
const enormous = similarPairs(3, 8192)
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
// A short side against a long one, sharing the short side's grams: the shape a
// query-inside-a-document comparison has, and the one where the packed walk
// searches the long side rather than walking it.
const skewedSmallLarge = words(50, 1024, 0x0a1b_2c3d).map((text): [string, string] => [
  text.slice(0, 32),
  text,
])
const skewedSmallHuge = words(20, 4096, 0x0a1b_2c3e).map((text): [string, string] => [
  text.slice(0, 32),
  text,
])
const skewedMediumHuge = words(20, 4096, 0x0a1b_2c3f).map((text): [string, string] => [
  text.slice(0, 512),
  text,
])

// An astral code point has no packed rung at trigram depth — three of them
// exceed a safe integer — so one anywhere in a sequence sends the whole profile
// back to the trie, after a packing scan that is then discarded. Both sides
// carry one so the comparison stays trie against trie; `long` is the same shape
// packed, and the pair is what says what the fallback costs. Dice alone: which
// representation a profile takes is the builder's decision, and Cosine reads
// whatever it was given.
const astralTail = long.map(([a, b]): [string, string] => [
  `${a.slice(0, -2)}😀`,
  `${b.slice(0, -2)}😀`,
])
const astralThroughout = long.map(([a, b]): [string, string] => [
  a.replaceAll(/(.{9})./gu, '$1😀'),
  b.replaceAll(/(.{9})./gu, '$1😀'),
])
// The same refusal past the transient counter's threshold, where it stops being
// free: the counter packs both sides before the profiles do, so a refusing
// element is now met twice. Where it sits decides the cost — a leading one
// abandons the attempt immediately, a trailing one is reached only after nearly
// everything has been scanned, at 6.9-9.2%. `trigrams, 1024 chars, similar`
// above is this shape packed, and the gap to it is what these two guard.
const astralLeadingLarge = large.map(([a, b]): [string, string] => [
  `😀${a.slice(2)}`,
  `😀${b.slice(2)}`,
])
const astralTrailingLarge = large.map(([a, b]): [string, string] => [
  `${a.slice(0, -2)}😀`,
  `${b.slice(0, -2)}😀`,
])
const astralThroughoutLarge = large.map(([a, b]): [string, string] => [
  a.replaceAll(/(.{9})./gu, '$1😀'),
  b.replaceAll(/(.{9})./gu, '$1😀'),
])
// Either side of the gram count at which Dice routes a direct comparison to the
// transient counter — 511 grams against 512, so one pair takes the profiles and
// the next takes the counter and nothing else differs.
//
// `editedPairs`, not `similarPairs`: the counter is selected on a gram count, so
// a case labelled 513 characters has to *be* 513 characters. `similarPairs`
// inserts and deletes, and a pair generated at the boundary could land either
// side of it — which would make these two cases measure the same path and say
// nothing.
const belowGate = editedPairs(20, 512, 8)
const atGate = editedPairs(20, 513, 8)
const trigramsBelowGate = editedPairs(20, 513, 8, 0x51c0_de01)
const trigramsAtGate = editedPairs(20, 514, 8, 0x51c0_de01)
// Lengths alone put these out of reach of a high cutoff, which is the case the
// gram-count bound exists to answer without building either trie.
const lengthSkewed = words(100, 512, 0x0ba7_d101).map((value, index) =>
  index % 2 === 0 ? [value, value.slice(0, 32)] : [value.slice(0, 32), value],
)

const choices = words(1_000, 12)
const query = 'abcdefghijkl'
const queries = words(100, 12, 0x1357_9bdf)
// Same length as the query and unrelated to it, so the gram-count bound is 1
// for every one of them and only the score itself can reject a candidate. This
// is the shape a content-aware bound would have to earn its branch on.
const sameLength = words(2_000, 12, 0x0f1e_2d3c)
// The other side of the same question: a short query against long choices,
// where the bound rejects on gram counts alone but a raw search has already
// built the candidate's trie to find them.
const longChoices = words(1_000, 512, 0x0ba7_d101)
const shortQuery = 'abcdefghijklmnopqrstuvwxyzabcdef'

const scorer = createScorer(diceMetric)
const cosine = createScorer(cosineMetric)
const trigrams = createScorer(diceMetric, { gramSize: 3 })
const cosineTrigrams = createScorer(cosineMetric, { gramSize: 3 })
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
// The two trigram kernels duplicate their hot loops on purpose, so each needs
// its own case: one can regress without touching the other.
const preparedCosineTrigramChoices = choices.map((text) => ({
  prepared: cosineTrigrams.prepareChoice(text),
}))

// Every case below inlines its own loop rather than calling one shared helper,
// for the reason `bench/distance.bench.ts` gives: V8 attaches an inline cache
// to a function literal, and a shared body would measure whichever group ran
// first while its call site was still monomorphic.

describe('diceSimilarity', () => {
  measure('4 chars, similar', () => {
    for (const [a, b] of tiny) diceSimilarity(a, b)
  })
  measure('6 chars, similar', () => {
    for (const [a, b] of small) diceSimilarity(a, b)
  })
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
  measure('1024 chars, similar', () => {
    for (const [a, b] of large) diceSimilarity(a, b)
  })
  measure('4096 chars, similar', () => {
    for (const [a, b] of huge) diceSimilarity(a, b)
  })
  measure('8192 chars, similar', () => {
    for (const [a, b] of enormous) diceSimilarity(a, b)
  })
  measure('4096 chars, one repeated gram', () => {
    for (const [a, b] of repetitive) diceSimilarity(a, b)
  })
  // The same trade at the other routed depth. Both depths cross the counter's
  // gate here and both lose on text with almost no distinct grams, so a change
  // that improves one and ruins the other has to show it.
  measure('trigrams, 4096 chars, one repeated gram', () => {
    for (const [a, b] of repetitive) trigrams.score(a, b)
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
  measure('trigrams, 128 chars, similar', () => {
    for (const [a, b] of long) cosineTrigrams.score(a, b)
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

describe('dice direct, long and skewed', () => {
  // The symmetric cases above establish only that sorting loses to a trie
  // somewhere past 512 characters at bigrams. Whether that is one boundary or
  // several is what these ask: a short side keeps the sort small however long
  // the other one is, and trigrams sort the same number of grams into three
  // times the key width. A routing rule would need this whole matrix to move
  // together before it could be written as one length.
  measure('32 vs 1024 chars', () => {
    for (const [a, b] of skewedSmallLarge) diceSimilarity(a, b)
  })
  measure('32 vs 4096 chars', () => {
    for (const [a, b] of skewedSmallHuge) diceSimilarity(a, b)
  })
  measure('512 vs 4096 chars', () => {
    for (const [a, b] of skewedMediumHuge) diceSimilarity(a, b)
  })
  measure('trigrams, 1024 chars, similar', () => {
    for (const [a, b] of large) trigrams.score(a, b)
  })
  measure('trigrams, 4096 chars, similar', () => {
    for (const [a, b] of huge) trigrams.score(a, b)
  })
})

describe('dice packing fallback', () => {
  // The control: the same 128 characters, packed. The gap between it and the
  // two below is the trie's — building one and walking two of them — and not
  // the discarded packing scan, which measured 2%: against the build that never
  // attempted packing at all, an astral corpus is 1.02x slower wherever the
  // refusing element sits, while ASCII construction is 1.51x faster. What these
  // rows guard is that the fallback stays as cheap as the old path was, so a
  // future scan-first idea has to show it costs nothing here.
  measure('trigrams, 128 chars, ascii', () => {
    for (const [a, b] of long) trigrams.score(a, b)
  })
  measure('trigrams, 128 chars, astral at the end', () => {
    for (const [a, b] of astralTail) trigrams.score(a, b)
  })
  measure('trigrams, 128 chars, astral throughout', () => {
    for (const [a, b] of astralThroughout) trigrams.score(a, b)
  })
  measure('trigrams, 1024 chars, astral at the start', () => {
    for (const [a, b] of astralLeadingLarge) trigrams.score(a, b)
  })
  measure('trigrams, 1024 chars, astral at the end', () => {
    for (const [a, b] of astralTrailingLarge) trigrams.score(a, b)
  })
  measure('trigrams, 1024 chars, astral throughout', () => {
    for (const [a, b] of astralThroughoutLarge) trigrams.score(a, b)
  })
})

describe('dice counter routing boundary', () => {
  // 511 grams against 512: below the gate the profiles answer, at it the
  // transient counter does. The gap between the two rows in a pair is what the
  // routing decision is worth at the point it is taken, and the pair is what
  // stops the constant being moved without a number.
  measure('511 grams, profiles', () => {
    for (const [a, b] of belowGate) diceSimilarity(a, b)
  })
  measure('512 grams, counter', () => {
    for (const [a, b] of atGate) diceSimilarity(a, b)
  })
  measure('trigrams, 511 grams, profiles', () => {
    for (const [a, b] of trigramsBelowGate) trigrams.score(a, b)
  })
  measure('trigrams, 512 grams, counter', () => {
    for (const [a, b] of trigramsAtGate) trigrams.score(a, b)
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
  // A rising cutoff is the only thing rejecting anything here: every candidate
  // is the query's length, so the gram-count bound is 1 throughout.
  measure('bestMatch, 2000 same-length choices', () => {
    bestMatch(query, sameLength, { scorer })
  })
  measure('limit 5, 2000 same-length choices, threshold 0.5', () => {
    search(query, sameLength, { scorer, limit: 5, threshold: 0.5 })
  })
  // What the gram-count bound cannot save on a raw search: it rejects on the
  // counts alone, but `prepareChoice` has built the candidate's trie before the
  // kernel ever sees it.
  measure('32-char query, 1000 raw 512-char choices, threshold 0.8', () => {
    search(shortQuery, longChoices, { scorer, limit: null, threshold: 0.8 })
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
  measure('trigrams, 100 queries, 1000 prepared choices', () => {
    for (const each of queries) {
      search(each, preparedCosineTrigramChoices, {
        scorer: cosineTrigrams,
        limit: null,
        getPrepared: (row) => row.prepared,
      })
    }
  })
})
