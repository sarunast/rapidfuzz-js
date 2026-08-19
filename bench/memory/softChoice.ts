/**
 * What a soft Tversky matcher retains per corpus item.
 *
 * Soft matching is the one scorer here whose prepared choice is a *derived*
 * representation rather than a packed profile, so what it holds is a design
 * decision taken per field: the occurrence walk, the distinct-element counts,
 * the canonical elements a weighted score prices, and — not held, and the open
 * question — the weighted profile itself. A scan benchmark cannot price any of
 * that. It reports the time a held representation saves and says nothing about
 * the megabytes it costs over a corpus the throughput suite never builds.
 *
 * ```sh
 * pnpm bench:memory:soft
 * pnpm bench:memory:soft --choices=20000 --tokens=12
 * pnpm bench:memory:soft --vocabulary=20000
 * ```
 *
 * **`--vocabulary` is what prices the index**, not `--choices`. The indexed
 * arms retain one inner q-gram entry per *distinct* token, so a corpus whose
 * every occurrence is distinct and one drawing three tokens from a shared
 * vocabulary cost the same matcher very different amounts. The default draws
 * every token distinct, which is the worst case for the vocabulary side.
 *
 * **One configuration per child, always.** A heap delta taken after another
 * matcher already exists in the process measures the collector's mood as much
 * as the matcher: the arms differ by tens of bytes a choice against a live set
 * of tens of megabytes, and a second matcher in the same process moved a
 * reading further than the difference being hunted.
 *
 * **Exact Tversky is measured beside it as the reference**, because the number
 * that decides anything is relative. A weighted soft choice costing more than
 * an exact weighted one is a fact about this engine; costing 400 B is a fact
 * about V8's object header.
 *
 * Three-token records are the default because that is what the feature is for —
 * company names where one token carries a typo — and the shape where a fixed
 * per-object cost hurts most.
 */
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import { normalizedSimilarity as indelSimilarity } from '../../dist/algorithms/indel/index.js'
import { similarity as tverskySimilarity } from '../../dist/algorithms/tversky/index.js'
import { createIndexedMatcher, createMatcher, createScorer } from '../../dist/index.js'
import { words } from '../harness/corpus.ts'
import {
  assertCliOptions,
  collectGarbage,
  numberOption,
  parseCli,
  runIsolated,
  sampleMemory,
  writeStructured,
} from './harness.ts'

const DEFAULT_CHOICES = 50_000
const DEFAULT_TOKENS = 3
/** Zero draws a distinct token per occurrence rather than sharing a pool. */
const DEFAULT_VOCABULARY = 0
const WORD_LENGTH = 9
const SEED = 0x31c4_0001
const ELEMENT_THRESHOLD = 0.8

/** How a scorer is configured, which is the whole experiment. */
type Arm =
  | 'soft'
  | 'softIndexed'
  | 'softWeighted'
  | 'softWeightedIndexed'
  | 'exact'
  | 'exactWeighted'

const ARMS: readonly Arm[] = [
  'soft',
  'softIndexed',
  'softWeighted',
  'softWeightedIndexed',
  'exact',
  'exactWeighted',
]

function isArm(value: string): value is Arm {
  return ARMS.some((arm) => arm === value)
}

interface ArmResult {
  readonly arm: Arm
  readonly choices: number
  readonly tokensPerChoice: number
  readonly distinctTokens: number
  readonly retainedBytes: number
  readonly retainedBytesPerChoice: number
  readonly retainedBytesPerDistinctToken: number
  readonly constructionMs: number
}

function corpusOf(
  choices: number,
  tokensPerChoice: number,
  vocabulary: number,
): string[][] {
  if (vocabulary === 0) {
    const tokens = words(choices * tokensPerChoice, WORD_LENGTH, SEED)
    return Array.from({ length: choices }, (_unused, at) =>
      tokens.slice(at * tokensPerChoice, at * tokensPerChoice + tokensPerChoice),
    )
  }
  // Cycling the pool in order is what "deterministic uniform reuse" means, and
  // it stays uniform at every pool size — a fixed stride only spreads the draws
  // when it happens to be coprime with the pool.
  const pool = words(vocabulary, WORD_LENGTH, SEED)
  let at = 0
  return Array.from({ length: choices }, () =>
    Array.from({ length: tokensPerChoice }, () => pool[at++ % vocabulary]),
  )
}

/**
 * One element weighted above the rest of its record, which is what a caller
 * reaches for element weights to say. A uniform table prices nothing and
 * compiles away to the unweighted engine.
 */
function weightsOf(corpus: readonly string[][]): Map<string, number> {
  return new Map(corpus.map((record) => [record[0], 5]))
}

function scorerOf(arm: Arm, corpus: readonly string[][]) {
  const element = { scorer: createScorer(indelSimilarity), threshold: ELEMENT_THRESHOLD }
  if (arm === 'soft' || arm === 'softIndexed') {
    return createScorer(tverskySimilarity, { gramSize: 1, elementSimilarity: element })
  }
  if (arm === 'softWeighted' || arm === 'softWeightedIndexed') {
    return createScorer(tverskySimilarity, {
      gramSize: 1,
      elementSimilarity: element,
      elementWeights: weightsOf(corpus),
      defaultElementWeight: 1,
    })
  }
  if (arm === 'exact') return createScorer(tverskySimilarity, { gramSize: 1 })
  return createScorer(tverskySimilarity, {
    gramSize: 1,
    elementWeights: weightsOf(corpus),
    defaultElementWeight: 1,
  })
}

function measure(
  arm: Arm,
  choices: number,
  tokensPerChoice: number,
  vocabulary: number,
): ArmResult {
  const corpus = corpusOf(choices, tokensPerChoice, vocabulary)
  const scorer = scorerOf(arm, corpus)
  collectGarbage()
  const before = sampleMemory().retained
  const started = performance.now()
  const matcher = arm.endsWith('Indexed')
    ? createIndexedMatcher(corpus, { scorer })
    : createMatcher(corpus, { scorer })
  const constructionMs = performance.now() - started
  collectGarbage()
  const retained = sampleMemory().retained - before
  if (matcher.size !== choices) throw new Error('the matcher dropped a choice')
  // A corpus with one distinct token per occurrence and one with a small shared
  // vocabulary cost the index very differently, so the per-choice figure alone
  // does not say what the vocabulary side is worth.
  const distinctTokens = new Set(corpus.flat()).size
  return {
    arm,
    choices,
    tokensPerChoice,
    distinctTokens,
    retainedBytes: retained,
    retainedBytesPerChoice: retained / choices,
    retainedBytesPerDistinctToken: retained / distinctTokens,
    constructionMs,
  }
}

const here = fileURLToPath(import.meta.url)
const options = parseCli(process.argv.slice(2))
assertCliOptions(options, ['child', 'arm', 'choices', 'tokens', 'vocabulary'])
const choices = numberOption(options, 'choices', DEFAULT_CHOICES)
const tokens = numberOption(options, 'tokens', DEFAULT_TOKENS)
const vocabulary = numberOption(options, 'vocabulary', DEFAULT_VOCABULARY)

if (options.has('child')) {
  const requested = options.get('arm')
  if (typeof requested !== 'string' || !isArm(requested)) {
    throw new TypeError(`--arm must be one of ${ARMS.join(', ')}`)
  }
  writeStructured(measure(requested, choices, tokens, vocabulary))
} else {
  writeStructured(
    ARMS.map((arm) =>
      runIsolated<ArmResult>(here, [
        `--arm=${arm}`,
        `--choices=${choices}`,
        `--tokens=${tokens}`,
        // Omitted rather than passed as zero: `numberOption` takes only
        // positive integers, and absent is what "a distinct token per
        // occurrence" means.
        ...(vocabulary === 0 ? [] : [`--vocabulary=${vocabulary}`]),
      ]),
    ),
  )
}
