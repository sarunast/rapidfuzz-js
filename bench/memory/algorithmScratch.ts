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
 * Both readings are taken after the call has returned and after three
 * collections, so neither is the peak the comparison reached while it ran —
 * nothing here measures that, and a reading that needs it has to come from an
 * allocation hook rather than a heap delta. What they separate is when the
 * scratch is let go: immediately, which is what the ownership split promises,
 * or only once some later call happens to displace it.
 *
 * Each round measures against a heap the round before it left, so the median
 * across them is the growth a further comparison adds rather than the total: it
 * reads near zero once the capped pool and the symbol table are there, and the
 * first round is the one that pays for them. Growth is the reading that matters
 * — a bound only holds if a thousand more comparisons do not move it.
 *
 * Both have a floor of tens of kilobytes either way — a heap measured after
 * three collections moves that much between two runs that did nothing at all,
 * and one control run came back at -877 kB. It is a scale check, not a byte
 * count, against the 67 MB such a comparison allocates. Read a regression as an
 * order of magnitude rather than a number.
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

/** One round: what the comparison leaves behind, and what then outlives it. */
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
      heldAfterTheCallReturned: median(held),
      heldAfterUnrelatedWork: median(survived),
      heldAfterUnrelatedWorkRange: [Math.min(...survived), Math.max(...survived)],
      heldPerElement: median(held) / count,
    },
    null,
    2,
  )}\n`,
)
