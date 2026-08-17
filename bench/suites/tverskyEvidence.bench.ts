/**
 * What `explain` costs, against the `score` it explains.
 *
 * The point is not a speed gate — evidence is cold by design, allocates per
 * call, and runs over the handful of results a search already chose. It is the
 * shape of the cost: explanation has to scale with the **pair**, not with the
 * corpus and not with the size of the compiled weight table.
 *
 * Its own file rather than a group in `weightedTokenIndex.bench.ts`, for that
 * file's own reason: a `describe` body runs at module load, so scorers built
 * here would change the environment its cases are measured in. Each `measure`
 * body inlines its own loop, because V8 attaches an inline cache to a function
 * literal and a shared body would measure whichever group ran first.
 */

import { similarity as tverskyMetric } from '../../src/algorithms/tversky/index.js'
import { createScorer } from '../../src/index.js'
import { words } from '../harness/corpus.js'
import { describe, measure } from '../harness/harness.js'

const SKILLS = words(400, 9, 0x30b1_0055)
const PRIMARY = SKILLS.map((word) => `primary:${word}`)
const COMMON = ['senior', 'engineer', 'remote', 'zurich', 'typescript', 'react']

const pairs = Array.from({ length: 10_000 }, (_, index) => ({
  first: [
    COMMON[index % COMMON.length],
    PRIMARY[index % PRIMARY.length],
    PRIMARY[(index * 7 + 1) % PRIMARY.length],
  ],
  second: [
    COMMON[(index + 1) % COMMON.length],
    PRIMARY[index % PRIMARY.length],
    PRIMARY[(index * 13 + 5) % PRIMARY.length],
  ],
}))

// Genuinely weighted, not merely present: one amount everywhere is
// uniform-positive, which the scorer detects and compiles away — a table filled
// with `1` under a default of `1` would measure the unweighted engine and prove
// the opposite of what the table-size pair is for. A cheap tier and an ignored
// token keep `uniformPositive` false while the primary skills ride the default.
const BASE_TABLE = new Map<unknown, number>([
  ...COMMON.map((token): [unknown, number] => [token, 0.2]),
  ['ag', 0],
])
// The same table plus 20,000 entries no pair below ever mentions, at the weight
// the compared skills already carry. Seven entries against 20,007, with an
// identical set of weight groups: every compared element, its weight, its group
// and its score are the same on both sides, so the difference is the cost of
// looking into a larger map rather than of walking it. Naming a new amount here
// would add a group and move the others' ordinals, which is a second variable.
const LARGE_TABLE = new Map<unknown, number>([
  ...BASE_TABLE,
  ...Array.from({ length: 20_000 }, (_, at): [unknown, number] => [`unused${at}`, 4]),
])

const configuration = {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  defaultElementWeight: 4,
} as const
const weighted = createScorer(tverskyMetric, {
  ...configuration,
  elementWeights: BASE_TABLE,
})
const largeTable = createScorer(tverskyMetric, {
  ...configuration,
  elementWeights: LARGE_TABLE,
})
const plain = createScorer(tverskyMetric, { gramSize: 1, alpha: 1, beta: 0.1 })

// The construction above is the whole experiment, so prove it rather than trust
// it: a uniform-positive table compiles away, and the two tables must agree to
// the last bit or they are not measuring the same comparison.
const probe = [pairs[0].first, pairs[0].second] as const
if (weighted.score(...probe) === plain.score(...probe)) {
  throw new Error('the weight table priced nothing — it compiled away')
}
if (weighted.score(...probe) !== largeTable.score(...probe)) {
  throw new Error('the two weight tables do not score the same pair alike')
}

describe('explaining a pair', () => {
  measure('scorer.score, weighted, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += weighted.score(pairs[index].first, pairs[index].second)
    }
    return total
  })
  measure('scorer.explain, weighted, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += weighted.explain(pairs[index].first, pairs[index].second).matches.length
    }
    return total
  })
  measure('scorer.explain, unweighted, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += plain.explain(pairs[index].first, pairs[index].second).matches.length
    }
    return total
  })
})

// The case this file exists for: the same compared elements against a weight
// table nearly three thousand times larger. A per-element map lookup answers in
// much the same time; anything that walked the table would not.
describe('explaining against a larger weight table', () => {
  measure('scorer.explain, base weight table, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += weighted.explain(pairs[index].first, pairs[index].second).matches.length
    }
    return total
  })
  measure('scorer.explain, base table + 20000 unused weights, 10000 pairs', () => {
    let total = 0
    for (let index = 0; index < 10_000; index++) {
      total += largeTable.explain(pairs[index].first, pairs[index].second).matches.length
    }
    return total
  })
})
