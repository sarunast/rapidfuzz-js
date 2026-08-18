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
 * ```
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
import { createMatcher, createScorer } from '../../dist/index.js'
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
const WORD_LENGTH = 9
const SEED = 0x31c4_0001
const ELEMENT_THRESHOLD = 0.8

/** How a scorer is configured, which is the whole experiment. */
type Arm = 'soft' | 'softWeighted' | 'exact' | 'exactWeighted'

const ARMS: readonly Arm[] = ['soft', 'softWeighted', 'exact', 'exactWeighted']

function isArm(value: string): value is Arm {
  return ARMS.some((arm) => arm === value)
}

interface ArmResult {
  readonly arm: Arm
  readonly choices: number
  readonly tokensPerChoice: number
  readonly retainedBytes: number
  readonly retainedBytesPerChoice: number
}

function corpusOf(choices: number, tokensPerChoice: number): string[][] {
  const tokens = words(choices * tokensPerChoice, WORD_LENGTH, SEED)
  return Array.from({ length: choices }, (_unused, at) =>
    tokens.slice(at * tokensPerChoice, at * tokensPerChoice + tokensPerChoice),
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
  if (arm === 'soft') {
    return createScorer(tverskySimilarity, { gramSize: 1, elementSimilarity: element })
  }
  if (arm === 'softWeighted') {
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

function measure(arm: Arm, choices: number, tokensPerChoice: number): ArmResult {
  const corpus = corpusOf(choices, tokensPerChoice)
  const scorer = scorerOf(arm, corpus)
  collectGarbage()
  const before = sampleMemory().retained
  const matcher = createMatcher(corpus, { scorer })
  collectGarbage()
  const retained = sampleMemory().retained - before
  if (matcher.size !== choices) throw new Error('the matcher dropped a choice')
  return {
    arm,
    choices,
    tokensPerChoice,
    retainedBytes: retained,
    retainedBytesPerChoice: retained / choices,
  }
}

const here = fileURLToPath(import.meta.url)
const options = parseCli(process.argv.slice(2))
assertCliOptions(options, ['child', 'arm', 'choices', 'tokens'])
const choices = numberOption(options, 'choices', DEFAULT_CHOICES)
const tokens = numberOption(options, 'tokens', DEFAULT_TOKENS)

if (options.has('child')) {
  const requested = options.get('arm')
  if (typeof requested !== 'string' || !isArm(requested)) {
    throw new TypeError(`--arm must be one of ${ARMS.join(', ')}`)
  }
  writeStructured(measure(requested, choices, tokens))
} else {
  writeStructured(
    ARMS.map((arm) =>
      runIsolated<ArmResult>(here, [
        `--arm=${arm}`,
        `--choices=${choices}`,
        `--tokens=${tokens}`,
      ]),
    ),
  )
}
