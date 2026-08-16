/**
 * What an indexed matcher's query scratch retains once a broad query has run,
 * and what releasing it would cost.
 *
 * `QueryState.reserve` was grow-only: a `limit: null` search over N choices
 * reserves N ids and N scores — 12 bytes a choice — and a later `best()` reserved
 * one without ever giving them back. The construction figure the README quotes
 * cannot see this, because `bench/comparison/indexedSearch/memory.mjs` reads the
 * heap before any query runs.
 *
 * ```sh
 * pnpm bench:memory:index
 * pnpm bench:memory:index --max=1000000
 * ```
 *
 * **Deltas are computed inside the child, never across children.** At a million
 * choices the index is hundreds of megabytes while the scratch under test is
 * twelve, so a median of absolute heap readings would lose the effect in the
 * noise of the large structure. Each child subtracts against its own
 * post-construction baseline and reports the differences; the parent medians
 * those. Each result is dropped by returning from the function that made it, so
 * it is structurally unreachable before the collection that follows.
 *
 * **The two halves drive different surfaces, deliberately.** Memory goes through
 * the public `createIndexedMatcher`, because what a real matcher retains is the
 * question. Timing goes through the sealed `ChoiceIndex` underneath it, because
 * every public entry allocates a result copy of the same 12 bytes a choice that
 * is under measurement — `search` ranks into a fresh pair and builds a `Match`
 * each, and `searchIter` copies the borrowed arrays before it yields anything.
 * At a million choices that is 12 MB of unrelated garbage per timed call, enough
 * for a collection to land in one arm and not the other. `scan(query, null)`
 * reaches the same corpus-sized collect path and returns the scratch itself.
 *
 * **The churn arms alternate by round.** The second arm in a process runs
 * against a warmer call site, and the effect being hunted may be a few percent,
 * so the order flips each round and each child reports its own paired ratio.
 * Medianing the ratios keeps the in-process pairing, which is where most of the
 * child-to-child noise cancels; a ratio of medians would throw it away.
 *
 * **The ladder prices the release itself.** A cap can shrink to what the narrow
 * query needs or to the cap; the two differ only over a run of *growing* narrow
 * queries, where shrinking to the need reallocates at every rung and shrinking
 * to the cap allocates once. Each rung is timed as it grows and again once the
 * scratch already fits it, so the difference is that rung's allocation — four of
 * them, summed, against one allocation of `RETAINED_RESULT_SLOTS`.
 *
 * Before a cap lands the churn ratio should read about 1.00, because nothing
 * reallocates yet. That run is the control the "after" is read against.
 */
import { execFileSync } from 'node:child_process'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import { words } from '../harness/corpus.ts'

const GRAM_SIZE = 2
const WORD_LENGTH = 24
const SEED = 0x0ba7_d101

/** Children per (size, half). Odd, so a median is a measurement. */
const ROUNDS = 5

/** Timed samples per churn arm, per child. Odd for the same reason. */
const SAMPLES = 9

/** Untimed queries before the samples, so the call site is warm for both arms. */
const WARMUP = 20

/** Narrow limits a broad query could be followed by, in growing order. */
const LADDER = [1, 5, 10_000, 50_000]

/**
 * Ladder samples per rung. Fewer than the churn arms take, because the top
 * rungs rank tens of thousands of results and cost seconds at a million
 * choices — and what is being read off them is an allocation either side of a
 * hundred milliseconds of ranking, which does not need nine repeats to settle.
 */
const LADDER_SAMPLES = 3

/**
 * A million-choice indexed matcher and the `Match` array an unlimited search
 * over it materialises both exceed a default old space.
 */
const CHILD_HEAP_MB = 8192

function collect(): void {
  globalThis.gc?.()
  globalThis.gc?.()
  globalThis.gc?.()
}

function retainedBytes(): number {
  const usage = process.memoryUsage()
  return usage.heapUsed + usage.arrayBuffers
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[(sorted.length - 1) >> 1] ?? 0
}

async function indexedMatcherOf(count: number) {
  const { similarity } = await import('../../dist/algorithms/dice/index.js')
  const { createIndexedMatcher, createScorer } = await import('../../dist/index.js')

  const scorer = createScorer(similarity, { gramSize: GRAM_SIZE })
  const choices = words(count, WORD_LENGTH, SEED)
  return { matcher: createIndexedMatcher(choices, { scorer }), query: choices[0] }
}

/** The borrowed result pair an index hands back, valid until its next call. */
interface SelectedChoices {
  readonly ids: Uint32Array
  readonly scores: Float64Array
  readonly length: number
}

/** The two members of `ChoiceIndex` this probe drives. */
interface SealedChoiceIndex {
  scan(query: string, threshold: number | null): SelectedChoices
  select(query: string, threshold: number | null, limit: number | null): SelectedChoices
}

/** What `createDiceIndexBuilder` returns, narrowed to what this probe calls. */
interface ChoiceIndexBuilder {
  add(choice: string): void
  seal(): SealedChoiceIndex
}

/**
 * The sealed index a matcher would hold, built directly.
 *
 * `dist/` publishes declarations for its entry points only, so the specifier is
 * a resolved URL rather than a literal: TypeScript types a dynamic import it
 * cannot follow as `any`, which the annotation then pins to the shape above.
 */
async function sealedIndexOf(count: number) {
  const dice = new URL('../../dist/algorithms/ngram/inverted/dice.js', import.meta.url)
    .href
  const loaded: { createDiceIndexBuilder: (gramSize: number) => ChoiceIndexBuilder } =
    await import(dice)

  const choices = words(count, WORD_LENGTH, SEED)
  const builder = loaded.createDiceIndexBuilder(GRAM_SIZE)
  for (const choice of choices) builder.add(choice)
  return { index: builder.seal(), query: choices[0] }
}

// ---------------------------------------------------------------- the child

interface MemoryReading {
  readonly afterBroad: number
  readonly afterBest: number
  readonly afterSecondBroad: number
}

/**
 * What survives a broad query, a narrow one, and a second broad one, each read
 * against this process's own post-construction baseline.
 */
async function measureMemory(count: number): Promise<MemoryReading> {
  const { matcher, query } = await indexedMatcherOf(count)

  // Returning is the drop: the results are unreachable by the time the caller
  // collects, without a scope whose only job is to say so.
  function broadAndDrop(): void {
    const found = matcher.search(query, { limit: null })
    if (found.length !== count) {
      throw new Error(`a broad query reached ${found.length} of ${count} choices`)
    }
  }

  function bestAndDrop(): void {
    if (matcher.best(query) === undefined) throw new Error('the query matched nothing')
  }

  collect()
  const constructed = retainedBytes()

  broadAndDrop()
  collect()
  const afterBroad = retainedBytes() - constructed

  bestAndDrop()
  collect()
  const afterBest = retainedBytes() - constructed

  broadAndDrop()
  collect()
  const afterSecondBroad = retainedBytes() - constructed

  if (matcher.size !== count) throw new Error('the matcher did not hold every choice')
  return { afterBroad, afterBest, afterSecondBroad }
}

interface RungReading {
  readonly limit: number
  readonly growMs: number
  readonly fitsMs: number
}

interface ChurnReading {
  readonly reuseMs: number
  readonly reallocateMs: number
  readonly ratio: number
  readonly ladder: readonly RungReading[]
}

/**
 * What a broad query costs with the oversized scratch still held, against what
 * it costs when a narrow query has been through since.
 */
async function measureChurn(count: number, round: number): Promise<ChurnReading> {
  const { index, query } = await sealedIndexOf(count)

  /** A scan with no threshold: every choice qualifies, so it reserves one slot each. */
  function broad(): void {
    const found = index.scan(query, null)
    if (found.length !== count) {
      throw new Error(`a broad scan returned ${found.length} of ${count} choices`)
    }
  }

  function narrow(limit: number): void {
    const room = limit < count ? limit : count
    const found = index.select(query, null, limit)
    if (found.length !== room) {
      throw new Error(`a limit of ${limit} returned ${found.length} of ${room} choices`)
    }
  }

  function timeBroad(): number {
    const started = process.hrtime.bigint()
    const found = index.scan(query, null)
    const elapsed = Number(process.hrtime.bigint() - started) / 1e6
    if (found.length !== count) {
      throw new Error(`a broad scan returned ${found.length} of ${count} choices`)
    }
    return elapsed
  }

  function timeNarrow(limit: number): number {
    const room = limit < count ? limit : count
    const started = process.hrtime.bigint()
    const found = index.select(query, null, limit)
    const elapsed = Number(process.hrtime.bigint() - started) / 1e6
    if (found.length !== room) {
      throw new Error(`a limit of ${limit} returned ${found.length} of ${room} choices`)
    }
    return elapsed
  }

  /** The scratch is already corpus-sized, so the timed query reserves nothing. */
  function sampleReuse(): number {
    broad()
    return timeBroad()
  }

  /** A narrow query in between is what the cap shrinks the scratch on. */
  function sampleReallocate(): number {
    broad()
    narrow(1)
    return timeBroad()
  }

  broad()
  for (let warm = 0; warm < WARMUP; warm++) {
    timeBroad()
    timeNarrow(1)
  }

  const reuse: number[] = []
  const reallocate: number[] = []
  const reuseFirst = round % 2 === 0
  for (let sample = 0; sample < SAMPLES; sample++) {
    if (reuseFirst) {
      reuse.push(sampleReuse())
      reallocate.push(sampleReallocate())
    } else {
      reallocate.push(sampleReallocate())
      reuse.push(sampleReuse())
    }
  }

  const grow = LADDER.map((): number[] => [])
  const fits = LADDER.map((): number[] => [])
  for (let sample = 0; sample < LADDER_SAMPLES; sample++) {
    broad()
    for (let rung = 0; rung < LADDER.length; rung++)
      grow[rung].push(timeNarrow(LADDER[rung]))
    for (let rung = 0; rung < LADDER.length; rung++) {
      narrow(LADDER[rung])
      fits[rung].push(timeNarrow(LADDER[rung]))
    }
  }

  const reuseMs = median(reuse)
  const reallocateMs = median(reallocate)
  return {
    reuseMs,
    reallocateMs,
    ratio: reuseMs === 0 ? 0 : reallocateMs / reuseMs,
    ladder: LADDER.map((limit, rung) => ({
      limit,
      growMs: median(grow[rung]),
      fitsMs: median(fits[rung]),
    })),
  }
}

function argumentValue(prefix: string): string | undefined {
  const found = process.argv.find((value) => value.startsWith(prefix))
  return found === undefined ? undefined : found.slice(prefix.length)
}

if (process.argv.includes('--child')) {
  if (globalThis.gc === undefined) throw new Error('run the child with --expose-gc')
  const count = Number(argumentValue('--size='))
  const round = Number(argumentValue('--round='))
  const half = argumentValue('--half=')
  const reading =
    half === 'memory' ? await measureMemory(count) : await measureChurn(count, round)
  process.stdout.write(`${JSON.stringify(reading)}\n`)
} else {
  // -------------------------------------------------------------- the parent

  const sizes = [10_000, 100_000]
  const max = Number(argumentValue('--max=') ?? 0)
  if (max >= 250_000) sizes.push(250_000)
  if (max >= 1_000_000) sizes.push(1_000_000)

  const here = fileURLToPath(import.meta.url)

  function runChild(half: string, count: number, round: number): unknown {
    const output = execFileSync(
      process.execPath,
      [
        '--expose-gc',
        `--max-old-space-size=${CHILD_HEAP_MB}`,
        here,
        '--child',
        `--half=${half}`,
        `--size=${count}`,
        `--round=${round}`,
      ],
      { encoding: 'utf8', maxBuffer: 1 << 20 },
    )
    return JSON.parse(output)
  }

  function isMemoryReading(value: unknown): value is MemoryReading {
    return typeof value === 'object' && value !== null && 'afterBroad' in value
  }

  function isChurnReading(value: unknown): value is ChurnReading {
    return typeof value === 'object' && value !== null && 'ratio' in value
  }

  function perChoice(value: number, count: number): string {
    return `${(value / count).toFixed(1)} B`.padStart(12)
  }

  console.log(
    `\n  Query scratch an indexed matcher retains, median of ${ROUNDS} child processes.\n` +
      "  Each figure is that child's own delta against its post-construction heap,\n" +
      '  so the index itself is on the wrong side of the baseline by construction.',
  )

  for (const count of sizes) {
    const memory: MemoryReading[] = []
    const churn: ChurnReading[] = []
    for (let round = 0; round < ROUNDS; round++) {
      const memoryReading = runChild('memory', count, round)
      if (isMemoryReading(memoryReading)) memory.push(memoryReading)
      const churnReading = runChild('churn', count, round)
      if (isChurnReading(churnReading)) churn.push(churnReading)
    }

    console.log(`\n  ${count.toLocaleString()} choices`)
    console.log(`  ${'held after'.padEnd(34)}${'per choice'.padStart(12)}`)
    console.log(
      `  ${'a broad search'.padEnd(34)}` +
        perChoice(median(memory.map((entry) => entry.afterBroad)), count),
    )
    console.log(
      `  ${'a following best()'.padEnd(34)}` +
        perChoice(median(memory.map((entry) => entry.afterBest)), count),
    )
    console.log(
      `  ${'a second broad search'.padEnd(34)}` +
        perChoice(median(memory.map((entry) => entry.afterSecondBroad)), count),
    )

    const ratio = median(churn.map((entry) => entry.ratio))
    console.log(
      '\n  a broad scan after a narrow one, against back-to-back broad scans: ' +
        `${ratio.toFixed(3)}x\n` +
        `  ${median(churn.map((entry) => entry.reuseMs)).toFixed(3)} ms reusing, ` +
        `${median(churn.map((entry) => entry.reallocateMs)).toFixed(3)} ms after the narrow query`,
    )

    console.log(
      `\n  ${'a narrow query at limit'.padEnd(26)}${'growing'.padStart(11)}${'already fits'.padStart(14)}`,
    )
    for (let rung = 0; rung < LADDER.length; rung++) {
      const growing = median(churn.map((entry) => entry.ladder[rung].growMs))
      const fitting = median(churn.map((entry) => entry.ladder[rung].fitsMs))
      console.log(
        `  ${LADDER[rung].toLocaleString().padEnd(26)}` +
          `${growing.toFixed(3).padStart(8)} ms${fitting.toFixed(3).padStart(11)} ms`,
      )
    }
  }

  console.log(
    `\n  Children ran with a ${CHILD_HEAP_MB} MB old space: a million-choice matcher and\n` +
      '  the Match array an unlimited search materialises do not fit in a default one.\n',
  )
}
