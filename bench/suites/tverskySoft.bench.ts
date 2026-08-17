/**
 * What `elementSimilarity` costs, and what its size limit is holding back.
 *
 * Three questions, and they want different fixtures. A realistic pair — company
 * names where one token carries a typo — leaves one element against one, and
 * measures the feature as anyone will actually meet it. A scan asks the same
 * question of a corpus, where a query is prepared once and met with thousands of
 * candidates it shares nothing with, so it is the only fixture that can see what
 * the query side costs per candidate. The adverse pairs sit exactly on the limit
 * of 32 distinct unmatched elements a side with *every* candidate edge surviving,
 * which is the shape the limit exists to refuse the next size up of; they are
 * what keeps the numbers quoted beside `MAX_SOFT_ELEMENTS` honest.
 *
 * Two costs sit behind a soft pair and a fixture has to separate them. Edge
 * building is `n × m` element scores and answers to the distinct counts alone;
 * the matching is a shortest-path search over `n + m` nodes repeated once per
 * augmenting path, and how many of those there are is *not* a function of the
 * distinct counts. So the same `32 × 32` elements are measured three ways: at one
 * occurrence each, at thirty-two each, and with the counts skewed so that where
 * one side is rare the other is common. Scaling every count uniformly is free and
 * skewing them is not, which is the whole reason `MAX_SOFT_AUGMENTATIONS` exists
 * next to a limit on the element counts.
 *
 * The `1 × 32` case is the capped form of the shape the old product limit missed.
 * Give it 1024 distinct counterparts instead of 32 — a `1 × 1024` that limit
 * allowed at exactly its 1024 comparisons — and the pair took three orders of
 * magnitude longer. `MAX_SOFT_ELEMENTS` refuses that outright, which is why it
 * cannot be a case in this file.
 *
 * Its own file, for `tverskyEvidence.bench.ts`'s reasons: a `describe` body runs
 * at module load, so scorers built here would change the environment another
 * file's cases are measured in, and each `measure` body inlines its own loop
 * because V8 attaches an inline cache to a function literal.
 */

import { normalizedSimilarity as indelMetric } from '../../src/algorithms/indel/index.js'
import { similarity as tverskyMetric } from '../../src/algorithms/tversky/index.js'
import { createMatcher, createScorer } from '../../src/index.js'
import { words } from '../harness/corpus.js'
import { describe, measure } from '../harness/harness.js'

const THRESHOLD = 0.8
const element = createScorer(indelMetric)
const soft = createScorer(tverskyMetric, {
  gramSize: 1,
  elementSimilarity: { scorer: element, threshold: THRESHOLD },
})
const exact = createScorer(tverskyMetric, { gramSize: 1 })

const COMPANY = words(500, 9, 0x50f7_0001)
const SUFFIX = ['ag', 'gmbh', 'sa', 'ltd']

/** One doubled character, which is what a real typo in a company name looks like. */
function mistyped(word: string): string {
  return `${word.slice(0, 5)}${word[4]}${word.slice(5)}`
}

const realistic = Array.from({ length: 10_000 }, (_, index) => {
  const head = COMPANY[index % COMPANY.length]
  const tail = COMPANY[(index * 7 + 3) % COMPANY.length]
  const suffix = SUFFIX[index % SUFFIX.length]
  return {
    first: [head, tail, suffix],
    second: [mistyped(head), tail, suffix],
  }
})

// A scan is the third shape, and the only one where a query side is prepared
// once and met with many candidates. Each query is a one-token typo of one
// corpus record: that record reserves two tokens exactly and leaves a single
// fuzzy pair, and the other 1999 candidates share nothing, so exact matching
// reserves nothing and all nine cells are compared and rejected. Both are worth
// having — the near candidate is the least a preparation can be amortized over,
// and the disjoint ones are what a scan is actually made of. The pair fixtures
// above cannot see any of it: they score one pair per call, so a query is
// prepared for a single candidate.
const RECORDS = 2000
const SCAN_TOKENS = words(3 * RECORDS, 9, 0x31c4_0001)
const scanned = Array.from({ length: RECORDS }, (_, index) =>
  SCAN_TOKENS.slice(index * 3, index * 3 + 3),
)
const scanQueries = Array.from({ length: 5 }, (_, index) => {
  const record = scanned[index * 397]
  return [mistyped(record[0]), record[1], record[2]]
})
// The weighted engine derives more per query than the plain one — group ids,
// totals and a sorted distinct view — so a scan is the only fixture that can
// see whether it derives them once or once per candidate. Two groups rather
// than one: a uniform table prices nothing and compiles away to this file's
// unweighted scorer.
const SCAN_WEIGHTS = new Map<string, number>(scanned.map((record) => [record[0], 5]))
const softWeighted = createScorer(tverskyMetric, {
  gramSize: 1,
  elementSimilarity: { scorer: element, threshold: THRESHOLD },
  elementWeights: SCAN_WEIGHTS,
  defaultElementWeight: 1,
})
const softMatcher = createMatcher(scanned, { scorer: soft })
const exactMatcher = createMatcher(scanned, { scorer: exact })
const weightedMatcher = createMatcher(scanned, { scorer: softWeighted })

// Distinct tokens sharing a long body, so every one of the `n × m` candidate
// edges clears the threshold and the matching sees a full matrix. Distinct
// *random* tokens would measure the edge build alone — no edge would survive —
// and a corpus of three repeated tokens collapses in the element table before
// the solver is reached at all.
function block(count: number, offset: number): string[] {
  return Array.from(
    { length: count },
    (_, at) => `shared-company-body-and-more-${String(offset + at).padStart(4, '0')}`,
  )
}

/** How often the element at a position occurs. */
type Occurrences = (at: number) => number

/** The same distinct elements, each occurring as often as `times` says. */
function repeated(source: readonly string[], times: Occurrences): string[] {
  const occurrences: string[] = []
  source.forEach((token, at) => {
    for (let count = times(at); count > 0; count--) occurrences.push(token)
  })
  return occurrences
}

const once: Occurrences = () => 1

function adverse(
  rows: number,
  columns: number,
  pairs: number,
  rowTimes: Occurrences = once,
  columnTimes: Occurrences = once,
) {
  return Array.from({ length: pairs }, (_, index) => ({
    first: repeated(block(rows, index * rows), rowTimes),
    second: repeated(block(columns, 5000 + index * columns), columnTimes),
  }))
}

const square = adverse(32, 32, 10)
const multiplied = adverse(
  32,
  32,
  10,
  () => 32,
  () => 32,
)
// One element carrying most of a side's occurrences against a side that spreads
// them evenly — the same 32 x 32 elements and the same edges as `square`, so any
// difference is the augmenting paths the skew buys.
const skewed = adverse(
  32,
  32,
  10,
  (at) => (at === 0 ? 4096 : 1),
  () => 3,
)
const oneAgainstMany = adverse(1, 32, 10, () => 1024)

// The fixture is the whole experiment, so prove it rather than trust it. Density
// is a claim about the *edges*, not about the score: an unbalanced shape scores
// low however many edges survive, because the longer side keeps what nothing can
// pair with. Ask the element scorer directly instead.
for (const [name, shape] of [
  ['32 x 32', square],
  ['32 x 32 repeated', multiplied],
  ['32 x 32 skewed', skewed],
  ['1 x 32 repeated', oneAgainstMany],
] as const) {
  const probe = shape[0]
  if (exact.score(probe.first, probe.second) !== 0) {
    throw new Error(
      `${name} shares tokens exactly — the leftovers are not the whole matrix`,
    )
  }
  for (const one of new Set(probe.first)) {
    for (const other of new Set(probe.second)) {
      if (element.score(one, other) < THRESHOLD) {
        throw new Error(
          `${name} drops the edge ${one}/${other} — this is not full density`,
        )
      }
    }
  }
}

describe('a soft pair at a realistic size', () => {
  measure('one typo in three tokens, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += soft.score(realistic[index].first, realistic[index].second)
    }
    return total
  })
  measure('the same pairs without elementSimilarity, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += exact.score(realistic[index].first, realistic[index].second)
    }
    return total
  })
})

describe('a soft scan over a corpus', () => {
  measure('2000 choices, 5 queries', () => {
    let total = 0
    for (let index = 0; index < 5; index++) {
      const best = softMatcher.best(scanQueries[index])
      total += best === undefined ? 0 : best.score
    }
    return total
  })
  measure('the same scan with element weights, 5 queries', () => {
    let total = 0
    for (let index = 0; index < 5; index++) {
      const best = weightedMatcher.best(scanQueries[index])
      total += best === undefined ? 0 : best.score
    }
    return total
  })
  measure('the same scan without elementSimilarity, 5 queries', () => {
    let total = 0
    for (let index = 0; index < 5; index++) {
      const best = exactMatcher.best(scanQueries[index])
      total += best === undefined ? 0 : best.score
    }
    return total
  })
})

describe('a soft pair on the size limit', () => {
  measure('32 x 32 leftovers, every edge surviving, 10 pairs', () => {
    let total = 0
    for (let index = 0; index < 10; index++) {
      total += soft.score(square[index].first, square[index].second)
    }
    return total
  })
  measure('the same 32 x 32 with 32 occurrences each, 10 pairs', () => {
    let total = 0
    for (let index = 0; index < 10; index++) {
      total += soft.score(multiplied[index].first, multiplied[index].second)
    }
    return total
  })
  measure('the same 32 x 32 with the occurrences skewed, 10 pairs', () => {
    let total = 0
    for (let index = 0; index < 10; index++) {
      total += soft.score(skewed[index].first, skewed[index].second)
    }
    return total
  })
  measure('1 x 32 leftovers, the one occurring 1024 times, 10 pairs', () => {
    let total = 0
    for (let index = 0; index < 10; index++) {
      total += soft.score(oneAgainstMany[index].first, oneAgainstMany[index].second)
    }
    return total
  })
  measure('32 x 32 tokens without elementSimilarity, 10 pairs', () => {
    let total = 0
    for (let index = 0; index < 10; index++) {
      total += exact.score(square[index].first, square[index].second)
    }
    return total
  })
})
