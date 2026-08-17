// Fixtures and the exhaustive oracle every inverted-index test checks
// against. Shared rather than copied: an index is only ever correct
// relative to the exhaustive scorer it reproduces, so all the files
// have to be comparing against the same one.
//
// A spec carries metric configuration only; gram size is a representation
// dimension the test loops own. Representation suites take one spec per
// index implementation, weight semantics stay in the Tversky parity runs.

import { similarity as cosineSimilarity } from '../src/algorithms/cosine/index.js'
import { similarity as diceSimilarity } from '../src/algorithms/dice/index.js'
import { createCosineIndexBuilder } from '../src/algorithms/ngram/inverted/cosine.js'
import { createDiceIndexBuilder } from '../src/algorithms/ngram/inverted/dice.js'
import { createTverskyIndexBuilder } from '../src/algorithms/ngram/inverted/tversky.js'
import { similarity as tverskySimilarity } from '../src/algorithms/tversky/index.js'
import type { ChoiceIndex } from '../src/core/scoring/choiceIndex.js'
import { createScorer } from '../src/core/scoring/scorer.js'
import type { Sequence } from '../src/core/types.js'
import { createMatcher } from '../src/index.js'

export type MetricSpec =
  | { readonly metric: 'dice' }
  | { readonly metric: 'cosine' }
  | { readonly metric: 'tversky'; readonly alpha: number; readonly beta: number }

function matcherOf(spec: MetricSpec, gramSize: number, choices: readonly Sequence[]) {
  switch (spec.metric) {
    case 'dice':
      return createMatcher(choices, {
        scorer: createScorer(diceSimilarity, { gramSize }),
      })
    case 'cosine':
      return createMatcher(choices, {
        scorer: createScorer(cosineSimilarity, { gramSize }),
      })
    case 'tversky':
      return createMatcher(choices, {
        scorer: createScorer(tverskySimilarity, {
          gramSize,
          alpha: spec.alpha,
          beta: spec.beta,
        }),
      })
  }
}

export function indexOf(
  spec: MetricSpec,
  gramSize: number,
  choices: readonly Sequence[],
): ChoiceIndex {
  const builder =
    spec.metric === 'dice'
      ? createDiceIndexBuilder(gramSize)
      : spec.metric === 'cosine'
        ? createCosineIndexBuilder(gramSize)
        : createTverskyIndexBuilder(gramSize, spec.alpha, spec.beta)
  for (const choice of choices) builder.add(choice)
  return builder.seal()
}

/** What the exhaustive Matcher answers, as `(id, score)` pairs. */
export function exhaustive(
  spec: MetricSpec,
  gramSize: number,
  choices: readonly Sequence[],
  query: Sequence,
  threshold: number | null,
  limit: number | null,
): { id: number; score: number }[] {
  const matcher = matcherOf(spec, gramSize, choices)
  const call = threshold === null ? { limit } : { limit, threshold }
  return matcher
    .search(query, call)
    .map((match) => ({ id: Number(match.key), score: match.score }))
}

export function pairs(selected: {
  ids: Uint32Array
  scores: Float64Array
  length: number
}): { id: number; score: number }[] {
  const out: { id: number; score: number }[] = []
  for (let at = 0; at < selected.length; at++) {
    out.push({ id: selected.ids[at], score: selected.scores[at] })
  }
  return out
}

/** `searchIter`'s answer: every qualifying match, in collection order. */
export function exhaustiveScan(
  spec: MetricSpec,
  gramSize: number,
  choices: readonly Sequence[],
  query: Sequence,
  threshold: number | null,
): { id: number; score: number }[] {
  const matcher = matcherOf(spec, gramSize, choices)
  const call = threshold === null ? undefined : { threshold }
  return [...matcher.searchIter(query, call)].map((match) => ({
    id: Number(match.key),
    score: match.score,
  }))
}

/** One spec per index implementation, for representation-level suites. */
export const REPRESENTATION_SPECS = [
  { metric: 'dice' },
  { metric: 'cosine' },
  { metric: 'tversky', alpha: 1, beta: 0.1 },
] as const satisfies readonly MetricSpec[]

/** The weight shapes Tversky parity has to hold at. */
export const TVERSKY_SPECS = [
  { metric: 'tversky', alpha: 0.5, beta: 0.5 },
  { metric: 'tversky', alpha: 1, beta: 1 },
  { metric: 'tversky', alpha: 1, beta: 0 },
  { metric: 'tversky', alpha: 1, beta: 0.1 },
  { metric: 'tversky', alpha: 0.2, beta: 0.7 },
  { metric: 'tversky', alpha: 2, beta: 10 },
] as const satisfies readonly MetricSpec[]

export const THRESHOLDS: readonly (number | null)[] = [null, 0, 0.1, 0.5, 0.8, 1]
export const LIMITS: readonly (number | null)[] = [1, 3, null]

export const CORPORA: readonly (readonly string[])[] = [
  [],
  [''],
  ['a'],
  ['ab', 'ab', 'ba'],
  ['abc', 'abcd', 'ab', 'a', ''],
  ['banana', 'bananas', 'ananab', 'band', 'b'],
  ['😀abc', 'abc😀', '😀abc', 'a\ud800b', '\ud800\ud800'],
  ['aaaa', 'aaaaa', 'aaaaaa', 'aaab'],
  ['zzz', 'yyy', 'xxx'],
  // Every choice shares `no`/`od`, which is what makes a list dense.
  ['node', 'nodes', 'noded', 'nodex', 'nodey', 'nodez', 'qq'],
]

export const QUERIES: readonly string[] = [
  '',
  'a',
  'ab',
  'abc',
  'banana',
  '😀abc',
  'aaaa',
  'qqq',
  'node',
]
