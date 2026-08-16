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

import { similarity as cosineMetric } from '../../src/algorithms/cosine/index.js'
import { similarity as diceMetric } from '../../src/algorithms/dice/index.js'
import { similarity as tverskyMetric } from '../../src/algorithms/tversky/index.js'
import {
  createIndexedMatcher,
  createMatcher,
  createScorer,
  search,
} from '../../src/index.js'
import { sentences, words } from '../harness/corpus.js'
import { describe, measure } from '../harness/harness.js'

const GRAM_SIZE = 3

const dice = createScorer(diceMetric, { gramSize: GRAM_SIZE })
const cosine = createScorer(cosineMetric, { gramSize: GRAM_SIZE })
// Three Tversky shapes: the default weights (routed to the Dice index, so
// their case is a routing control against the Dice pair above it), the
// practical asymmetric configuration, and pure query containment.
const tverskyDefault = createScorer(tverskyMetric, {
  gramSize: GRAM_SIZE,
  alpha: 0.5,
  beta: 0.5,
})
const tverskyAsymmetric = createScorer(tverskyMetric, {
  gramSize: GRAM_SIZE,
  alpha: 1,
  beta: 0.1,
})
const tverskyContainment = createScorer(tverskyMetric, {
  gramSize: GRAM_SIZE,
  alpha: 1,
  beta: 0,
})

// Two shapes rather than one: 24 random letters give a near-flat gram
// distribution, while sentences of short words repeat theirs, which is what
// lengthens posting lists.
const small = words(1_000, 24)
const medium = words(10_000, 24)
const large = words(100_000, 24, 0x0ba7_d101)
const phrases = sentences(10_000, 4)

// Every choice sharing most of its grams with every other, which is the shape
// the index is worst at and the docs lead with. Built here rather than in
// `harness/corpus.ts` because that file is hashed into every baseline entry in
// the suite, and this corpus is one file's business.
// Three tail lengths, not one: against a corpus of a single length a short
// query is refused by Dice's length bound alone, and the exhaustive arm would
// be measured rejecting choices rather than scoring them.
const PATH_TAILS = ['dist/index.js', 'lib/a.js', 'src/nested/deep/module/index.ts']
const paths = words(10_000, 8, 0x51ed_2701).map(
  (name, index) => `node_modules/${name}/${PATH_TAILS[index % PATH_TAILS.length]}`,
)

const smallQueries = words(100, 24, 0x1357_9bdf)
const mediumQueries = medium.slice(0, 100)
const phraseQueries = phrases.slice(0, 100)
const oneQuery = large[50_000]
// The three a shared-prefix corpus answers differently: grams nearly every
// choice holds, a whole path, and a fragment only one choice holds.
const commonPrefixQuery = 'node_modules/'
const wholePathQuery = paths[8_391]
const rareFragmentQuery = paths[8_391].slice(13, 21)

// The containment shape: a short fragment inside a longer document, where
// `alpha: 1, beta: 0` scores 1 whatever the document adds around it. Built
// here rather than in `harness/corpus.ts` for the reason `paths` is.
const containedFragments = words(10_000, 12, 0x7e57_c0de)
const containers = containedFragments.map(
  (fragment, index) => `prefix${index % 7} ${fragment} tail${index % 5}`,
)
const containedQueries = containedFragments.slice(0, 100)

const smallIndexed = createIndexedMatcher(small, { scorer: dice })
const mediumIndexed = createIndexedMatcher(medium, { scorer: dice })
const largeIndexed = createIndexedMatcher(large, { scorer: dice })
const phraseIndexed = createIndexedMatcher(phrases, { scorer: dice })
const mediumCosineIndexed = createIndexedMatcher(medium, { scorer: cosine })
const pathIndexed = createIndexedMatcher(paths, { scorer: dice })
const mediumTverskyDefaultIndexed = createIndexedMatcher(medium, {
  scorer: tverskyDefault,
})
const mediumTverskyIndexed = createIndexedMatcher(medium, { scorer: tverskyAsymmetric })
const largeTverskyIndexed = createIndexedMatcher(large, { scorer: tverskyAsymmetric })
const pathTverskyIndexed = createIndexedMatcher(paths, { scorer: tverskyAsymmetric })
const containerIndexed = createIndexedMatcher(containers, { scorer: tverskyContainment })

const smallMatcher = createMatcher(small, { scorer: dice })
const mediumMatcher = createMatcher(medium, { scorer: dice })
const largeMatcher = createMatcher(large, { scorer: dice })
const phraseMatcher = createMatcher(phrases, { scorer: dice })
const mediumCosineMatcher = createMatcher(medium, { scorer: cosine })
const pathMatcher = createMatcher(paths, { scorer: dice })
const mediumTverskyMatcher = createMatcher(medium, { scorer: tverskyAsymmetric })
const largeTverskyMatcher = createMatcher(large, { scorer: tverskyAsymmetric })
const pathTverskyMatcher = createMatcher(paths, { scorer: tverskyAsymmetric })
const containerMatcher = createMatcher(containers, { scorer: tverskyContainment })

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

describe('tversky search, 10000 choices', () => {
  // The routing control: default weights are served by the Dice index, so this
  // case is expected to read level with `dice search, 10000 choices > indexed,
  // 100 hits, threshold 0.5`.
  measure('indexed, default weights, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries)
      mediumTverskyDefaultIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, asymmetric, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries)
      mediumTverskyIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, asymmetric, 100 hits, threshold 0.5', () => {
    for (const query of mediumQueries)
      mediumTverskyMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, asymmetric, 100 misses, threshold 0.5', () => {
    for (const query of smallQueries)
      mediumTverskyIndexed.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('exhaustive, asymmetric, 100 misses, threshold 0.5', () => {
    for (const query of smallQueries)
      mediumTverskyMatcher.search(query, { limit: 5, threshold: 0.5 })
  })
  measure('indexed, asymmetric, 100 hits, threshold 0.8', () => {
    for (const query of mediumQueries)
      mediumTverskyIndexed.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('exhaustive, asymmetric, 100 hits, threshold 0.8', () => {
    for (const query of mediumQueries)
      mediumTverskyMatcher.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('indexed, asymmetric, best, 100 hits', () => {
    for (const query of mediumQueries) mediumTverskyIndexed.best(query)
  })
  measure('exhaustive, asymmetric, best, 100 hits', () => {
    for (const query of mediumQueries) mediumTverskyMatcher.best(query)
  })
})

// The actual Tversky use case: a short query against documents that contain
// it whole, where `alpha: 1, beta: 0` pays no penalty for what the document
// adds around the fragment.
describe('tversky containment, 10000 containers', () => {
  measure('indexed, 100 contained queries, threshold 0.8', () => {
    for (const query of containedQueries)
      containerIndexed.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('exhaustive, 100 contained queries, threshold 0.8', () => {
    for (const query of containedQueries)
      containerMatcher.search(query, { limit: 5, threshold: 0.8 })
  })
  measure('indexed, best, 100 contained queries', () => {
    for (const query of containedQueries) containerIndexed.best(query)
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

// `limit: null` is a different selection strategy, not a larger `limit`: with no
// room bound the top-k insertion walk degrades to a quadratic one, so an
// unlimited call collects and sorts instead. Both thresholds, because the
// threshold is what decides whether the sorted set is a handful or the corpus.
describe('dice search, unlimited results', () => {
  measure('indexed, 100 queries, 1000 choices, threshold 0.5', () => {
    for (const query of smallQueries)
      smallIndexed.search(query, { limit: null, threshold: 0.5 })
  })
  measure('exhaustive, 100 queries, 1000 choices, threshold 0.5', () => {
    for (const query of smallQueries)
      smallMatcher.search(query, { limit: null, threshold: 0.5 })
  })
  measure('indexed, 20 queries, 10000 choices, threshold 0.5', () => {
    for (let at = 0; at < 20; at++)
      mediumIndexed.search(mediumQueries[at], { limit: null, threshold: 0.5 })
  })
  measure('exhaustive, 20 queries, 10000 choices, threshold 0.5', () => {
    for (let at = 0; at < 20; at++)
      mediumMatcher.search(mediumQueries[at], { limit: null, threshold: 0.5 })
  })
  // No threshold, so every choice qualifies and the result *is* the corpus —
  // the one shape where sorting costs more than inserting, and the reason this
  // pair is here rather than only the selective one above.
  measure('indexed, 5 queries, 10000 choices, no threshold', () => {
    for (let at = 0; at < 5; at++)
      mediumIndexed.search(mediumQueries[at], { limit: null })
  })
  measure('exhaustive, 5 queries, 10000 choices, no threshold', () => {
    for (let at = 0; at < 5; at++)
      mediumMatcher.search(mediumQueries[at], { limit: null })
  })
})

// The adverse shape, kept where a future optimisation has to walk past it: file
// paths under one directory share nearly all their grams, so a query's posting
// lists name most of the corpus and the index has little left to skip. One
// query per case, because the exhaustive arm scores 10,000 choices of ~40
// characters.
describe('dice search, 10000 shared-prefix paths', () => {
  measure('indexed, common prefix query', () =>
    pathIndexed.search(commonPrefixQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('exhaustive, common prefix query', () =>
    pathMatcher.search(commonPrefixQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('indexed, whole path query', () =>
    pathIndexed.search(wholePathQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('exhaustive, whole path query', () =>
    pathMatcher.search(wholePathQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('indexed, rare fragment query', () =>
    pathIndexed.search(rareFragmentQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('exhaustive, rare fragment query', () =>
    pathMatcher.search(rareFragmentQuery, { limit: 5, threshold: 0.5 }),
  )
  // The general Tversky arithmetic on the same adverse shape: whether the
  // extra scoring work matters where the posting lists prune almost nothing.
  measure('tversky indexed, common prefix query', () =>
    pathTverskyIndexed.search(commonPrefixQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('tversky exhaustive, common prefix query', () =>
    pathTverskyMatcher.search(commonPrefixQuery, { limit: 5, threshold: 0.5 }),
  )
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
  measure('tversky indexed, 1 query, threshold 0.5', () =>
    largeTverskyIndexed.search(oneQuery, { limit: 5, threshold: 0.5 }),
  )
  measure('tversky exhaustive, 1 query, threshold 0.5', () =>
    largeTverskyMatcher.search(oneQuery, { limit: 5, threshold: 0.5 }),
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
