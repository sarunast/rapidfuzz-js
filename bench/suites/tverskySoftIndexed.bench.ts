/** Exhaustive-versus-indexed outer Soft Tversky search at deployment sizes. */
import { normalizedSimilarity as indelMetric } from '../../src/algorithms/indel/index.js'
import { similarity as tverskyMetric } from '../../src/algorithms/tversky/index.js'
import { createIndexedMatcher, createMatcher, createScorer } from '../../src/index.js'
import { words } from '../harness/corpus.js'
import { describe, measure } from '../harness/harness.js'

const inner = createScorer(indelMetric)
const scorer = createScorer(tverskyMetric, {
  gramSize: 1,
  elementSimilarity: { scorer: inner, threshold: 0.8 },
})
const lowThresholdScorer = createScorer(tverskyMetric, {
  gramSize: 1,
  elementSimilarity: { scorer: inner, threshold: 0.5 },
})
const vocabulary = words(150_000, 10, 0x51f7_4001)
const all = Array.from({ length: 50_000 }, (_, id) => [
  vocabulary[id * 3],
  vocabulary[id * 3 + 1],
  id % 3 === 0 ? 'ag' : id % 3 === 1 ? 'gmbh' : vocabulary[id * 3 + 2],
])
const typo = (token: string): string => `${token.slice(0, 5)}x${token.slice(5)}`

for (const size of [2_000, 50_000]) {
  const choices = all.slice(0, size)
  const exhaustive = createMatcher(choices, { scorer })
  const indexed = createIndexedMatcher(choices, { scorer })
  const lowExhaustive = createMatcher(choices, { scorer: lowThresholdScorer })
  const lowIndexed = createIndexedMatcher(choices, { scorer: lowThresholdScorer })
  const weightedScorer = createScorer(tverskyMetric, {
    gramSize: 1,
    elementSimilarity: { scorer: inner, threshold: 0.8 },
    elementWeights: new Map(choices.map((choice) => [choice[0], 5])),
    defaultElementWeight: 1,
  })
  const weightedExhaustive = createMatcher(choices, { scorer: weightedScorer })
  const weightedIndexed = createIndexedMatcher(choices, { scorer: weightedScorer })
  // 398 % 3 === 2, so its third token is a unique word rather than a shared
  // suffix; 399 % 3 === 0, so its third token is `ag`. The pair is what
  // separates "two exact tokens still locate it" from "only the vocabulary
  // index can".
  const isolated = choices[398]
  const suffixed = choices[399]
  const queries = [
    ['exact', isolated],
    ['typo without common token', [typo(isolated[0]), isolated[1], isolated[2]]],
    // The two remaining tokens are unseen, so the exact postings cannot reach
    // the intended choice at all and the typo has to travel the inner q-gram
    // vocabulary index. This is the path the feature exists for.
    ['typo, fuzzy reachable only', [typo(isolated[0]), 'unseen-second', 'unseen-third']],
    // The same, except the one exact token left is a suffix a third of the
    // corpus shares — so correctness drags every `ag` choice into the union.
    ['typo behind a common token', [typo(suffixed[0]), 'unseen-second', 'ag']],
    ['unseen', ['never-seen-token', 'nor-this-one', 'unknown-suffix']],
    ['common exact token', ['never-seen-token', 'nor-this-one', 'gmbh']],
    ['no neighbor', ['xxxxxxxxxx', 'yyyyyyyyyy', 'zzzzzzzzzz']],
  ] as const
  describe(`indexed Soft Tversky over ${size} three-token choices`, () => {
    for (const [name, matcher] of [
      ['exhaustive', exhaustive],
      ['indexed', indexed],
    ] as const) {
      for (const [queryName, query] of queries) {
        measure(`${name} best / ${queryName}`, () => matcher.best(query)?.score ?? 0)
        measure(
          `${name} top-5 / ${queryName}`,
          () => matcher.search(query, { limit: 5 }).length,
        )
        measure(
          `${name} thresholded / ${queryName}`,
          () => matcher.search(query, { limit: null, threshold: 0.7 }).length,
        )
      }
    }
    const typoQuery = queries[1][1]
    measure('exhaustive best / low inner threshold', () => lowExhaustive.best(typoQuery))
    measure('indexed best / low inner threshold', () => lowIndexed.best(typoQuery))
    measure('exhaustive best / weighted typo', () => weightedExhaustive.best(typoQuery))
    measure('indexed best / weighted typo', () => weightedIndexed.best(typoQuery))
  })
}
