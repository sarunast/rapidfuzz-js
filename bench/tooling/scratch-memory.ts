/**
 * What the shared scratch keeps after a comparison that is over its cap.
 *
 * The throughput suite cannot see this: a buffer that stays reachable costs no
 * time at all, and the case that allocated it has already reported its number.
 * What it costs is process memory, and the input that costs the most is one the
 * timings never reach — a sequence of distinct elements gives every one of them
 * its own block of `words`, so the mask pool grows with `length * words` where
 * text grows with the alphabet.
 *
 * Two readings, because only the second is a leak: what the comparison holds
 * while it runs, which nothing can avoid, and what survives one unrelated
 * comparison afterwards.
 *
 * The second reading has a floor of tens of kilobytes either way — a heap
 * measured after three collections moves that much between two runs that did
 * nothing at all, and one control run came back at -877 kB. It is a scale
 * check, not a byte count: the buffers that legitimately survive come to about
 * 11 kB, against a peak of 67 MB. Read the median across rounds, and read a
 * regression as an order of magnitude rather than a number.
 */
import process from 'node:process'

import { distance as levenshteinDistance } from '../../dist/algorithms/levenshtein/index.js'

const count = Number(process.argv[2] ?? 20_000)
if (!Number.isSafeInteger(count) || count <= 0) {
  throw new RangeError('element count must be a positive safe integer')
}
if (globalThis.gc === undefined) {
  throw new Error('run this benchmark with --expose-gc')
}

function collect(): void {
  globalThis.gc?.()
  globalThis.gc?.()
  globalThis.gc?.()
}

function retainedBytes(): number {
  const usage = process.memoryUsage()
  return usage.heapUsed + usage.arrayBuffers
}

// Distinct objects rather than text: an element is whatever the caller's
// sequence holds, and nothing about the API narrows it to a character.
function unique(tag: string): ReadonlyArray<unknown> {
  return Array.from({ length: count }, (_, index) => ({ tag, index }))
}

const ROUNDS = 5

/** One round: the peak the comparison reaches, and what outlives it. */
function round(): { held: number; survived: number } {
  collect()
  const before = retainedBytes()

  // Scoped so the sequences themselves are unreachable by the first reading,
  // which leaves the scratch as the only thing between it and the baseline.
  const distance = ((): number => levenshteinDistance(unique('a'), unique('b')))()
  if (distance !== count) throw new Error('the comparison did not run to completion')
  collect()
  const held = retainedBytes() - before

  levenshteinDistance('kitten', 'sitting')
  collect()
  return { held, survived: retainedBytes() - before }
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[(sorted.length - 1) >> 1] ?? 0
}

const rounds = Array.from({ length: ROUNDS }, round)
const held = rounds.map((entry) => entry.held)
const survived = rounds.map((entry) => entry.survived)

process.stdout.write(
  `${JSON.stringify(
    {
      elements: count,
      rounds: ROUNDS,
      heldByTheComparison: median(held),
      heldAfterUnrelatedWork: median(survived),
      heldAfterUnrelatedWorkRange: [Math.min(...survived), Math.max(...survived)],
      heldPerElement: median(held) / count,
    },
    null,
    2,
  )}\n`,
)
