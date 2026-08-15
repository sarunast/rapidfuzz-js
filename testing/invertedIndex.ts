// Fixtures and the exhaustive oracle every inverted-index test checks
// against. Shared rather than copied: an index is only ever correct
// relative to the exhaustive scorer it reproduces, so all four files
// have to be comparing against the same one.

import { similarity as cosineSimilarity } from '../src/algorithms/cosine/index.js'
import { similarity as diceSimilarity } from '../src/algorithms/dice/index.js'
import { createCosineIndexBuilder } from '../src/algorithms/ngram/inverted/cosine.js'
import { createDiceIndexBuilder } from '../src/algorithms/ngram/inverted/dice.js'
import type { ChoiceIndex } from '../src/core/scoring/choiceIndex.js'
import { createScorer } from '../src/core/scoring/scorer.js'
import { createMatcher } from '../src/index.js'

export type Metric = 'dice' | 'cosine'

export function indexOf(
  metric: Metric,
  gramSize: number,
  choices: readonly string[],
): ChoiceIndex {
  const builder =
    metric === 'dice'
      ? createDiceIndexBuilder(gramSize)
      : createCosineIndexBuilder(gramSize)
  for (const choice of choices) builder.add(choice)
  return builder.seal()
}

/** What the exhaustive Matcher answers, as `(id, score)` pairs. */
export function exhaustive(
  metric: Metric,
  gramSize: number,
  choices: readonly string[],
  query: string,
  threshold: number | null,
  limit: number | null,
): { id: number; score: number }[] {
  const scorer = createScorer(metric === 'dice' ? diceSimilarity : cosineSimilarity, {
    gramSize,
  })
  const matcher = createMatcher(choices, { scorer })
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
  metric: Metric,
  gramSize: number,
  choices: readonly string[],
  query: string,
  threshold: number | null,
): { id: number; score: number }[] {
  const scorer = createScorer(metric === 'dice' ? diceSimilarity : cosineSimilarity, {
    gramSize,
  })
  const matcher = createMatcher(choices, { scorer })
  const call = threshold === null ? undefined : { threshold }
  return [...matcher.searchIter(query, call)].map((match) => ({
    id: Number(match.key),
    score: match.score,
  }))
}

export const METRICS: readonly Metric[] = ['dice', 'cosine']
export const THRESHOLDS: readonly (number | null)[] = [null, 0, 0.5, 0.8, 1]
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
