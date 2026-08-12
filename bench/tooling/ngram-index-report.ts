/**
 * Parity, structural counters and retained memory for the inverted n-gram index
 * prototype, at corpus sizes the bench harness cannot hold.
 *
 * Run through `ngram-index-scale.ts`, which bundles this file — it reaches into
 * `src/`, so node cannot execute it directly.
 *
 * Parity runs first and throws on the first mismatch: a latency number for a
 * wrong result is worth nothing. Rows are written as JSON lines as they are
 * produced, so an interrupted run still leaves usable output.
 */

import process from 'node:process'

import { similarity as cosineMetric } from '../../src/algorithms/cosine/index.js'
import { similarity as diceMetric } from '../../src/algorithms/dice/index.js'
import { buildProfile, NGramProfile } from '../../src/algorithms/shared/ngram.js'
import { createMatcher, createScorer } from '../../src/index.js'
import { NGramIndex, type IndexCounters, type Scored } from './ngramIndex.js'

type Metric = 'dice' | 'cosine'

const METRICS: readonly Metric[] = ['dice', 'cosine']

/**
 * Only what this script calls, so a `Matcher`'s generic parameters stay out of
 * every local annotation. Structural assignment, never a cast.
 */
interface ExhaustiveMatcher {
  readonly size: number
  search(
    query: string,
    options: { limit: number | null; threshold?: number },
  ): readonly { readonly key: number; readonly score: number }[]
  best(
    query: string,
    options?: { threshold?: number },
  ): { readonly key: number; readonly score: number } | undefined
}

function matcherFor(
  metric: Metric,
  gramSize: number,
  choices: readonly string[],
): ExhaustiveMatcher {
  const items = [...choices]
  return metric === 'dice'
    ? createMatcher(items, { scorer: createScorer(diceMetric, { gramSize }) })
    : createMatcher(items, { scorer: createScorer(cosineMetric, { gramSize }) })
}

// ---------------------------------------------------------------- corpora

/** xorshift32, as `bench/tooling/corpus.ts` uses it: no `Math.random` anywhere. */
function rng(seed: number): () => number {
  let state = seed >>> 0
  if (state === 0) throw new RangeError('xorshift32 seed must be non-zero')
  return () => {
    state ^= state << 13
    state >>>= 0
    state ^= state >>> 17
    state ^= state << 5
    state >>>= 0
    return state / 0x1_0000_0000
  }
}

const LOWER: readonly string[] = [...'abcdefghijklmnopqrstuvwxyz']

/**
 * `alphabet-<k>` is uniform text over `k` letters, and `k` is the dial that
 * moves the predictor: with `k` letters there are `k ** gramSize` possible
 * grams, so a bigger alphabet spreads the corpus over more, shorter posting
 * lists. The sweep exists because the first round measured only 2 and 26, which
 * left the whole middle of the range — where the index stops paying — unmeasured.
 */
type CorpusClass = `alphabet-${number}` | 'zipf-words'

const CORPUS_CLASSES: readonly CorpusClass[] = ['alphabet-2', 'alphabet-26', 'zipf-words']

const SWEEP_CLASSES: readonly CorpusClass[] = [
  'alphabet-3',
  'alphabet-4',
  'alphabet-5',
  'alphabet-6',
  'alphabet-8',
  'alphabet-10',
  'alphabet-12',
  'alphabet-16',
  'alphabet-20',
  'alphabet-26',
  'zipf-words',
]

type QueryClass =
  | 'exact hit'
  | '1 typo'
  | '2 typos'
  | 'unrelated'
  | 'short'
  | 'common substring'
  | 'rare substring'

const QUERY_CLASSES: readonly QueryClass[] = [
  'exact hit',
  '1 typo',
  '2 typos',
  'unrelated',
  'short',
  'common substring',
  'rare substring',
]

interface Corpus {
  readonly choices: string[]
  readonly queries: ReadonlyMap<QueryClass, string>
  /**
   * A uniform-random corpus has a flat gram distribution, so its "common" and
   * "rare" substring queries are the same class by construction. Only
   * `zipf-words` separates them, which is the whole reason it is here: posting
   * skew is what an inverted index lives or dies by, and uniform text has none.
   */
  readonly separatesFrequency: boolean
}

/**
 * Joined rather than built with `+=`, and that is a measurement decision. A
 * 24-character string concatenated one character at a time is a chain of V8
 * cons strings, and the first `convSequence` over it flattens the chain — so the
 * corpus *shrinks* while the index is being built, and the retained-bytes delta
 * came out negative for exactly the two corpora built that way. `join` allocates
 * a flat string up front, so the baseline is already what it will be.
 */
function word(next: () => number, length: number, alphabet: readonly string[]): string {
  const characters: string[] = new Array<string>(length)
  for (let index = 0; index < length; index++) {
    characters[index] = alphabet[Math.floor(next() * alphabet.length)]
  }
  return characters.join('')
}

function substitute(
  next: () => number,
  source: string,
  count: number,
  alphabet: readonly string[],
): string {
  const characters = [...source]
  for (let edit = 0; edit < count && characters.length > 0; edit++) {
    const at = Math.floor(next() * characters.length)
    let replacement = alphabet[Math.floor(next() * alphabet.length)]
    while (replacement === characters[at]) {
      replacement = alphabet[Math.floor(next() * alphabet.length)]
    }
    characters[at] = replacement
  }
  return characters.join('')
}

function zipfCorpus(count: number, gramSize: number): Corpus {
  const next = rng(0x0d15_ea5e)
  const vocabulary: string[] = []
  for (let index = 0; index < 400; index++) {
    vocabulary.push(word(next, 3 + Math.floor(next() * 6), LOWER))
  }
  const weights: number[] = vocabulary.map((_value, rank) => 1 / (rank + 1))
  let total = 0
  for (const weight of weights) total += weight
  const pick = (): string => {
    let target = next() * total
    for (let rank = 0; rank < weights.length; rank++) {
      target -= weights[rank]
      if (target <= 0) return vocabulary[rank]
    }
    return vocabulary[weights.length - 1]
  }
  const choices: string[] = []
  for (let index = 0; index < count; index++) {
    const parts: string[] = []
    for (let part = 0; part < 4; part++) parts.push(pick())
    choices.push(parts.join(' '))
  }
  const hit = choices[Math.floor(count / 2)] ?? vocabulary[0]
  const frequent = vocabulary[0]
  const rare = vocabulary[vocabulary.length - 1]
  return {
    choices,
    queries: new Map<QueryClass, string>([
      ['exact hit', hit],
      ['1 typo', substitute(next, hit, 1, LOWER)],
      ['2 typos', substitute(next, hit, 2, LOWER)],
      ['unrelated', `${word(next, 6, LOWER)} ${word(next, 7, LOWER)}`],
      ['short', hit.slice(0, gramSize + 2)],
      ['common substring', `${frequent} ${frequent}`],
      ['rare substring', `${rare} ${rare}`],
    ]),
    separatesFrequency: true,
  }
}

function uniformCorpus(
  count: number,
  gramSize: number,
  alphabet: readonly string[],
  seed: number,
): Corpus {
  const next = rng(seed)
  const choices: string[] = []
  for (let index = 0; index < count; index++) choices.push(word(next, 24, alphabet))
  const hit = choices[Math.floor(count / 2)] ?? word(next, 24, alphabet)
  const other = choices[0] ?? hit
  return {
    choices,
    queries: new Map<QueryClass, string>([
      ['exact hit', hit],
      ['1 typo', substitute(next, hit, 1, alphabet)],
      ['2 typos', substitute(next, hit, 2, alphabet)],
      ['unrelated', word(next, 24, alphabet)],
      ['short', hit.slice(0, gramSize + 2)],
      ['common substring', hit.slice(0, 12)],
      ['rare substring', other.slice(0, 12)],
    ]),
    separatesFrequency: false,
  }
}

function corpusOf(kind: CorpusClass, count: number, gramSize: number): Corpus {
  if (kind === 'zipf-words') return zipfCorpus(count, gramSize)
  const letters = Number(kind.slice('alphabet-'.length))
  if (!Number.isSafeInteger(letters) || letters < 2 || letters > LOWER.length) {
    throw new RangeError(`alphabet size ${kind} is outside 2..${LOWER.length}`)
  }
  return uniformCorpus(count, gramSize, LOWER.slice(0, letters), 0x9e37_79b9)
}

// ---------------------------------------------------------------- index build

interface BuiltIndex {
  readonly index: NGramIndex
  readonly buildMs: number
}

/**
 * The fair build: every choice's profile is constructed here and dropped again,
 * because building one per choice is exactly what a Matcher pays during
 * construction. Handing prepared profiles to `add` would answer a different
 * question and answer it flatteringly.
 */
let packedKeys = true
let startRadix: number | null = null
let directBuild = false

function buildIndex(choices: readonly string[], gramSize: number): BuiltIndex {
  const started = process.hrtime.bigint()
  const index = new NGramIndex(gramSize, choices.length, packedKeys, startRadix)
  if (directBuild) {
    for (let id = 0; id < choices.length; id++) index.addSequence(id, choices[id])
  } else {
    for (let id = 0; id < choices.length; id++) {
      index.add(id, buildProfile(choices[id], gramSize))
    }
  }
  index.compact()
  return { index, buildMs: Number(process.hrtime.bigint() - started) / 1e6 }
}

function indexedBest(
  index: NGramIndex,
  metric: Metric,
  query: NGramProfile,
  threshold: number | null,
): Scored | undefined {
  return metric === 'dice'
    ? index.diceBest(query, threshold)
    : index.cosineBest(query, threshold)
}

function indexedSearch(
  index: NGramIndex,
  metric: Metric,
  query: NGramProfile,
  threshold: number | null,
  limit: number | null,
): Scored[] {
  return metric === 'dice'
    ? index.diceSearch(query, threshold, limit)
    : index.cosineSearch(query, threshold, limit)
}

// ---------------------------------------------------------------- parity

interface ParityCase {
  readonly metric: Metric
  readonly gramSize: number
  readonly choices: readonly string[]
  readonly query: string
  readonly threshold: number | null
  readonly limit: number | null
}

function label(each: ParityCase, call: 'search' | 'best'): string {
  return `${call} ${JSON.stringify(each)}`
}

function checkCase(each: ParityCase): void {
  const { metric, gramSize, choices, query, threshold, limit } = each
  const matcher = matcherFor(metric, gramSize, choices)
  const index = buildIndex(choices, gramSize).index
  const profile = buildProfile(query, gramSize)

  const expectedSearch = matcher.search(
    query,
    threshold === null ? { limit } : { limit, threshold },
  )
  const actualSearch = indexedSearch(index, metric, profile, threshold, limit)
  const sameSearch =
    expectedSearch.length === actualSearch.length &&
    expectedSearch.every(
      (entry, at) =>
        entry.key === actualSearch[at].id && entry.score === actualSearch[at].score,
    )
  if (!sameSearch) {
    throw new Error(
      `${label(each, 'search')}\n  exhaustive: ${JSON.stringify(
        expectedSearch.map((entry) => [entry.key, entry.score]),
      )}\n  indexed:    ${JSON.stringify(
        actualSearch.map((entry) => [entry.id, entry.score]),
      )}`,
    )
  }

  const expectedBest = matcher.best(query, threshold === null ? undefined : { threshold })
  const actualBest = indexedBest(index, metric, profile, threshold)
  const sameBest =
    expectedBest === undefined
      ? actualBest === undefined
      : actualBest !== undefined &&
        actualBest.id === expectedBest.key &&
        actualBest.score === expectedBest.score
  if (!sameBest) {
    throw new Error(
      `${label(each, 'best')}\n  exhaustive: ${JSON.stringify(
        expectedBest === undefined ? null : [expectedBest.key, expectedBest.score],
      )}\n  indexed:    ${JSON.stringify(
        actualBest === undefined ? null : [actualBest.id, actualBest.score],
      )}`,
    )
  }

  // The prefix-filtered path is held to the same standard as the full one: it
  // skips posting lists, so the only thing that says it skipped the right ones
  // is that the result is unchanged.
  if (metric === 'dice') {
    const filtered = index.dicePrefixSearch(profile, threshold, limit)
    const sameFiltered =
      expectedSearch.length === filtered.length &&
      expectedSearch.every(
        (entry, at) =>
          entry.key === filtered[at].id && entry.score === filtered[at].score,
      )
    if (!sameFiltered) {
      throw new Error(
        `${label(each, 'search')} [prefix]\n  exhaustive: ${JSON.stringify(
          expectedSearch.map((entry) => [entry.key, entry.score]),
        )}\n  filtered:   ${JSON.stringify(filtered.map((entry) => [entry.id, entry.score]))}`,
      )
    }
    const filteredBest = index.dicePrefixBest(profile, threshold)
    const sameFilteredBest =
      expectedBest === undefined
        ? filteredBest === undefined
        : filteredBest !== undefined &&
          filteredBest.id === expectedBest.key &&
          filteredBest.score === expectedBest.score
    if (!sameFilteredBest) {
      throw new Error(
        `${label(each, 'best')} [prefix]\n  exhaustive: ${JSON.stringify(
          expectedBest === undefined ? null : [expectedBest.key, expectedBest.score],
        )}\n  filtered:   ${JSON.stringify(
          filteredBest === undefined ? null : [filteredBest.id, filteredBest.score],
        )}`,
      )
    }
  }

  // The invariant every zero-fill rule rests on: with a query that has grams, a
  // positive score and a posting-list hit are the same event.
  if (profile.gramCount > 0) {
    const all = indexedSearch(index, metric, profile, null, null)
    const positives = all.filter((entry) => entry.score > 0).length
    if (positives !== index.counters.candidatesTouched) {
      throw new Error(
        `${label(each, 'search')}\n  ${positives} positive scores against ` +
          `${index.counters.candidatesTouched} touched candidates`,
      )
    }
    if (all.length !== choices.length) {
      throw new Error(
        `${label(each, 'search')}\n  unlimited search returned ${all.length} of ` +
          `${choices.length} choices`,
      )
    }
  }
}

const THRESHOLDS: readonly (number | null)[] = [null, 0, 0.5, 0.8, 1]
const LIMITS: readonly (number | null)[] = [0, 1, 3, null]

const FIXED_CORPORA: readonly (readonly string[])[] = [
  [],
  [''],
  ['a'],
  ['ab', 'ab', 'ba'],
  ['abc', 'abcd', 'ab', 'a', ''],
  ['banana', 'bananas', 'ananab', 'band', 'b'],
  ['😀abc', 'abc😀', '😀abc', 'a\ud800b', '\ud800\ud800'],
  ['aaaa', 'aaaaa', 'aaaaaa', 'aaab'],
  ['zzz', 'yyy', 'xxx'],
]

const FIXED_QUERIES: readonly string[] = [
  '',
  'a',
  'ab',
  'abc',
  'banana',
  '😀abc',
  'aaaa',
  'qqq',
]

function fixedParity(): number {
  let cases = 0
  for (const metric of METRICS) {
    for (const gramSize of [2, 3]) {
      for (const choices of FIXED_CORPORA) {
        for (const query of FIXED_QUERIES) {
          for (const threshold of THRESHOLDS) {
            for (const limit of LIMITS) {
              checkCase({ metric, gramSize, choices, query, threshold, limit })
              cases++
            }
          }
        }
      }
    }
  }
  return cases
}

async function parity(runs: number): Promise<void> {
  const cases = fixedParity()
  process.stdout.write(`${JSON.stringify({ kind: 'parity', mode: 'fixed', cases })}\n`)

  const fc = await import('fast-check')
  const letters = fc.constantFrom('a', 'b', 'c', '😀', '\ud800', ' ')
  const text = fc.array(letters, { maxLength: 12 }).map((parts) => parts.join(''))
  fc.assert(
    fc.property(
      fc.array(text, { maxLength: 12 }),
      text,
      fc.constantFrom(...THRESHOLDS),
      fc.constantFrom(...LIMITS),
      fc.constantFrom(2, 3),
      fc.constantFrom(...METRICS),
      (choices, query, threshold, limit, gramSize, metric) => {
        checkCase({ metric, gramSize, choices, query, threshold, limit })
        return true
      },
    ),
    { numRuns: runs, seed: 0x5eed },
  )
  process.stdout.write(`${JSON.stringify({ kind: 'parity', mode: 'random', runs })}\n`)
}

// ------------------------------------------------------- retained-profile check

/**
 * A structural companion to the heap numbers: the memory claim holds only if the
 * index kept no profile, and no correctness test would notice one hiding in it.
 * Walks every posting list, so it runs on small corpora only.
 */
function retainsProfile(value: unknown, depth: number): boolean {
  if (depth > 6) return false
  if (value instanceof NGramProfile) return true
  if (typeof value !== 'object' || value === null) return false
  if (ArrayBuffer.isView(value)) return false
  if (value instanceof Map) {
    for (const entry of value.values()) if (retainsProfile(entry, depth + 1)) return true
    return false
  }
  if (Array.isArray(value)) return value.some((entry) => retainsProfile(entry, depth + 1))
  return Object.values(value).some((entry) => retainsProfile(entry, depth + 1))
}

// ---------------------------------------------------------------- measurement

function collect(): void {
  const gc = globalThis.gc
  if (gc === undefined) throw new Error('run this script with --expose-gc')
  gc()
  gc()
  gc()
}

function retainedBytes(): number {
  const usage = process.memoryUsage()
  return usage.heapUsed + usage.arrayBuffers
}

function medianOf(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2
}

/** Where every timed body's result goes, so V8 cannot delete the work. */
const sink: { value: unknown } = { value: undefined }

/**
 * Warmed before it is timed, and the warmup is the same size as the measurement.
 * Without it the 10k and 100k exhaustive arms reported the same milliseconds for
 * ten times the work — the first call of three was carrying the median. These are
 * still indicative numbers; `bench/ngramIndex.bench.ts` is where the adaptive
 * sampling lives.
 */
function timeMedian(runs: number, body: () => unknown): number {
  const warmups = Math.max(3, runs)
  for (let run = 0; run < warmups; run++) sink.value = body()
  const samples: number[] = []
  for (let run = 0; run < runs; run++) {
    const started = process.hrtime.bigint()
    sink.value = body()
    samples.push(Number(process.hrtime.bigint() - started) / 1e6)
  }
  return medianOf(samples)
}

interface CounterRow {
  readonly kind: 'counters'
  readonly n: number
  readonly corpus: CorpusClass
  readonly gramSize: number
  readonly metric: Metric
  readonly queryClass: QueryClass
  readonly separatesFrequency: boolean
  readonly distinctQueryGrams: number
  readonly postingEntriesTouched: number
  readonly postingsPerChoice: number
  readonly postingsPerChoicePerGram: number
  readonly candidatesTouched: number
  readonly candidatesTouchedRatio: number
  readonly candidatesQualified: number
  readonly indexedMs: number
  readonly exhaustiveMs: number | null
  /** The prefix-filtered Dice path, where it applies; null for Cosine. */
  readonly filteredMs: number | null
  readonly filteredPostings: number | null
  readonly filteredVerifyProbes: number | null
  readonly filteredVerified: number | null
  readonly indexBuildMs: number
  readonly matcherBuildMs: number | null
  readonly gramVariety: number
  readonly meanShare: number
  readonly weightedShare: number
  readonly termWeightedShare: number
}

let counterThreshold = 0.5
const COUNTER_LIMIT = 5
/** Above this, a Matcher's profiles no longer fit beside the index. */
const EXHAUSTIVE_LIMIT = 100_000

function counterRuns(n: number): number {
  if (n >= 100_000) return 3
  if (n >= 10_000) return 7
  return 21
}

function counterRows(
  n: number,
  corpusClass: CorpusClass,
  gramSize: number,
  corpus: Corpus,
  built: BuiltIndex,
): CounterRow[] {
  const index = built.index
  const statistics = index.postingStatistics()
  const runs = counterRuns(n)
  const rows: CounterRow[] = []
  for (const metric of METRICS) {
    let matcher: ExhaustiveMatcher | null = null
    let matcherBuildMs: number | null = null
    if (n <= EXHAUSTIVE_LIMIT) {
      const started = process.hrtime.bigint()
      matcher = matcherFor(metric, gramSize, corpus.choices)
      matcherBuildMs = Number(process.hrtime.bigint() - started) / 1e6
      if (matcher.size !== n) throw new Error('matcher lost choices')
    }
    for (const queryClass of QUERY_CLASSES) {
      const query = corpus.queries.get(queryClass)
      if (query === undefined) throw new Error(`missing query class ${queryClass}`)
      // One call to fill the counters, then the timed runs. Each timed run
      // rebuilds the query profile, because a real query would.
      indexedSearch(
        index,
        metric,
        buildProfile(query, gramSize),
        counterThreshold,
        COUNTER_LIMIT,
      )
      const counters = { ...index.counters }
      const indexedMs = timeMedian(runs, () =>
        indexedSearch(
          index,
          metric,
          buildProfile(query, gramSize),
          counterThreshold,
          COUNTER_LIMIT,
        ),
      )
      const held = matcher
      const exhaustiveMs =
        held === null
          ? null
          : timeMedian(runs, () =>
              held.search(query, { limit: COUNTER_LIMIT, threshold: counterThreshold }),
            )
      let filteredMs: number | null = null
      let filteredCounters: IndexCounters | null = null
      if (metric === 'dice') {
        index.dicePrefixSearch(
          buildProfile(query, gramSize),
          counterThreshold,
          COUNTER_LIMIT,
        )
        filteredCounters = { ...index.counters }
        filteredMs = timeMedian(runs, () =>
          index.dicePrefixSearch(
            buildProfile(query, gramSize),
            counterThreshold,
            COUNTER_LIMIT,
          ),
        )
      }
      rows.push({
        kind: 'counters',
        n,
        corpus: corpusClass,
        gramSize,
        metric,
        queryClass,
        separatesFrequency: corpus.separatesFrequency,
        distinctQueryGrams: counters.distinctQueryGrams,
        postingEntriesTouched: counters.postingEntriesTouched,
        postingsPerChoice: counters.postingEntriesTouched / n,
        postingsPerChoicePerGram:
          counters.distinctQueryGrams === 0
            ? 0
            : counters.postingEntriesTouched / (n * counters.distinctQueryGrams),
        candidatesTouched: counters.candidatesTouched,
        candidatesTouchedRatio: counters.candidatesTouched / n,
        candidatesQualified: counters.candidatesQualified,
        indexedMs,
        exhaustiveMs,
        filteredMs,
        filteredPostings:
          filteredCounters === null ? null : filteredCounters.postingEntriesTouched,
        filteredVerifyProbes:
          filteredCounters === null ? null : filteredCounters.verifyProbes,
        filteredVerified:
          filteredCounters === null ? null : filteredCounters.verifiedCandidates,
        indexBuildMs: built.buildMs,
        matcherBuildMs,
        gramVariety: index.gramVariety(),
        meanShare: statistics.meanShare,
        weightedShare: statistics.weightedShare,
        termWeightedShare: statistics.termWeightedShare,
      })
    }
  }
  return rows
}

type Arm = 'index' | 'profiles'

interface MemoryRow {
  readonly kind: 'memory'
  readonly n: number
  readonly corpus: CorpusClass
  readonly gramSize: number
  readonly arm: Arm
  readonly bytes: number
  readonly bytesPerChoice: number
}

/**
 * One arm, from a heap that has held nothing else — which is why this is its own
 * process rather than a row in the sweep.
 *
 * Measured in-process alongside the counters, every delta came out *negative* at
 * 100k: the counters leave a Matcher's worth of collectable profiles behind, so
 * the baseline reads high and the collection that happens during the build
 * reclaims more than the build allocates. Three `gc()` calls do not fix that;
 * not having allocated it does.
 */
function measureArm(
  n: number,
  corpusClass: CorpusClass,
  gramSize: number,
  arm: Arm,
): MemoryRow {
  // 331 MiB per 100k prepared bigram profiles extrapolates to ~3.3 GiB at a
  // million, which measures the collector rather than the representation. The
  // 1M profile figure belongs in the writeup as an extrapolation, labelled.
  if (arm === 'profiles' && n > 100_000) {
    throw new RangeError('the profile arm stops at 100k — extrapolate above it')
  }
  const corpus = corpusOf(corpusClass, n, gramSize)
  collect()
  const before = retainedBytes()
  let held: number = 0
  if (arm === 'index') {
    const built = buildIndex(corpus.choices, gramSize)
    collect()
    held = retainedBytes() - before
    if (built.index.choiceCount !== n) throw new Error('index lost choices')
    if (n <= 100_000 && retainsProfile(built.index, 0)) {
      throw new Error('the index retained an NGramProfile — the memory claim is void')
    }
  } else {
    const matcher = matcherFor('dice', gramSize, corpus.choices)
    collect()
    held = retainedBytes() - before
    if (matcher.size !== n) throw new Error('matcher lost choices')
  }
  return {
    kind: 'memory',
    n,
    corpus: corpusClass,
    gramSize,
    arm,
    bytes: held,
    bytesPerChoice: held / n,
  }
}

/**
 * One corpus, described the way it needs to be read: what the index is, and
 * what one query of each class costs against it, beside the number of
 * candidates the exhaustive Matcher would have scored.
 */
function summarise(n: number, corpusClass: CorpusClass, gramSize: number): void {
  const corpus = corpusOf(corpusClass, n, gramSize)
  const built = buildIndex(corpus.choices, gramSize)
  const index = built.index
  const stats = index.postingStatistics()
  const percent = (value: number): string => `${(value * 100).toFixed(1)}%`
  process.stdout.write(
    `\n  ${corpusClass}, gram size ${gramSize}\n` +
      `    choices                ${n.toLocaleString().padStart(12)}\n` +
      `    distinct grams         ${stats.distinctGrams.toLocaleString().padStart(12)}\n` +
      `    posting entries        ${stats.totalEntries.toLocaleString().padStart(12)}\n` +
      `    counts width           ${`${stats.countsWidthBytes} byte`.padStart(12)}\n` +
      `    max count              ${String(stats.maxCount).padStart(12)}\n` +
      `    entries with count 1   ${percent(stats.singletonEntryShare).padStart(12)}\n` +
      `    lists all count 1      ${percent(stats.singletonListShare).padStart(12)}\n` +
      `    mean posting share     ${stats.meanShare.toFixed(5).padStart(12)}\n` +
      `    weighted share         ${stats.weightedShare.toFixed(5).padStart(12)}\n` +
      `    term-weighted share    ${stats.termWeightedShare.toFixed(5).padStart(12)}\n` +
      `    index build            ${`${built.buildMs.toFixed(0)} ms`.padStart(12)}\n` +
      `\n    query              postings   candidates   verified   scored by Matcher\n`,
  )
  for (const queryClass of QUERY_CLASSES) {
    const query = corpus.queries.get(queryClass)
    if (query === undefined) throw new Error(`missing query class ${queryClass}`)
    index.dicePrefixSearch(buildProfile(query, gramSize), 0.8, 5)
    const counters = { ...index.counters }
    process.stdout.write(
      `    ${queryClass.padEnd(17)}${counters.postingEntriesTouched.toLocaleString().padStart(9)}` +
        `${counters.candidatesTouched.toLocaleString().padStart(13)}` +
        `${counters.verifiedCandidates.toLocaleString().padStart(11)}` +
        `${n.toLocaleString().padStart(20)}\n`,
    )
  }
}

// ---------------------------------------------------------------- entry point

const SIZES: readonly number[] = [100, 1_000, 10_000, 100_000, 1_000_000]

type Mode = 'parity' | 'counters' | 'memory' | 'summary'

interface Options {
  readonly mode: Mode
  readonly max: number
  readonly runs: number
  readonly n: number | null
  readonly corpus: CorpusClass | null
  readonly gramSize: number | null
  readonly arm: Arm | null
  readonly threshold: number
  readonly sweep: boolean
  readonly packedKeys: boolean
  readonly startRadix: number | null
  readonly directBuild: boolean
}

function corpusClassOf(value: string): CorpusClass {
  const found = [...CORPUS_CLASSES, ...SWEEP_CLASSES].find(
    (candidate) => candidate === value,
  )
  if (found === undefined) throw new Error(`unknown corpus ${value}`)
  return found
}

function armOf(value: string): Arm {
  if (value === 'index' || value === 'profiles') return value
  throw new Error(`unknown arm ${value}`)
}

function numberOf(argument: string, prefix: string): number {
  const value = Number(argument.slice(prefix.length))
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${prefix} needs a positive safe integer`)
  }
  return value
}

function parseOptions(): Options {
  let mode: Mode = 'counters'
  let max = 1_000_000
  let runs = 400
  let n: number | null = null
  let corpus: CorpusClass | null = null
  let gramSize: number | null = null
  let arm: Arm | null = null
  let threshold = 0.5
  let sweep = false
  let packedKeys = true
  let startRadix: number | null = null
  let directBuild = false
  for (const argument of process.argv.slice(2)) {
    if (argument === '--parity') mode = 'parity'
    else if (argument === '--counters') mode = 'counters'
    else if (argument === '--memory') mode = 'memory'
    else if (argument === '--summary') mode = 'summary'
    else if (argument === '--sweep') sweep = true
    else if (argument === '--build=direct') directBuild = true
    else if (argument === '--build=profile') directBuild = false
    else if (argument === '--keys=string') packedKeys = false
    else if (argument === '--keys=packed') packedKeys = true
    else if (argument === '--keys=bmp') startRadix = 0x1_0000
    else if (argument.startsWith('--max=')) max = numberOf(argument, '--max=')
    else if (argument.startsWith('--runs=')) runs = numberOf(argument, '--runs=')
    else if (argument.startsWith('--n=')) n = numberOf(argument, '--n=')
    else if (argument.startsWith('--gram=')) gramSize = numberOf(argument, '--gram=')
    else if (argument.startsWith('--threshold=')) {
      threshold = Number(argument.slice('--threshold='.length))
      if (!(threshold > 0 && threshold <= 1)) {
        throw new RangeError('--threshold must be inside (0, 1]')
      }
    } else if (argument.startsWith('--corpus=')) {
      corpus = corpusClassOf(argument.slice('--corpus='.length))
    } else if (argument.startsWith('--arm=')) {
      arm = armOf(argument.slice('--arm='.length))
    } else throw new Error(`unknown argument ${argument}`)
  }
  return {
    mode,
    max,
    runs,
    n,
    corpus,
    gramSize,
    arm,
    threshold,
    sweep,
    packedKeys,
    startRadix,
    directBuild,
  }
}

/**
 * Gram size 2 stops at 100k. At a million choices one depth answers the
 * structural question, and building both doubles a pass already measured in
 * minutes.
 */
function gramSizesFor(n: number): readonly number[] {
  return n > 100_000 ? [3] : [2, 3]
}

const options = parseOptions()
counterThreshold = options.threshold
packedKeys = options.packedKeys
startRadix = options.startRadix
directBuild = options.directBuild

if (options.mode === 'parity') {
  await parity(options.runs)
} else if (options.mode === 'memory') {
  const { n, corpus, gramSize, arm } = options
  if (n === null || corpus === null || gramSize === null || arm === null) {
    throw new Error('--memory needs --n, --corpus, --gram and --arm')
  }
  process.stdout.write(`${JSON.stringify(measureArm(n, corpus, gramSize, arm))}\n`)
} else if (options.mode === 'summary') {
  for (const n of SIZES) {
    if (n > options.max) continue
    if (options.n !== null && n !== options.n) continue
    for (const corpusClass of options.sweep ? SWEEP_CLASSES : CORPUS_CLASSES) {
      if (options.corpus !== null && corpusClass !== options.corpus) continue
      for (const gramSize of gramSizesFor(n)) {
        if (options.gramSize !== null && gramSize !== options.gramSize) continue
        summarise(n, corpusClass, gramSize)
      }
    }
  }
} else {
  for (const n of SIZES) {
    if (n > options.max) continue
    if (options.n !== null && n !== options.n) continue
    for (const corpusClass of options.sweep ? SWEEP_CLASSES : CORPUS_CLASSES) {
      if (options.corpus !== null && corpusClass !== options.corpus) continue
      for (const gramSize of gramSizesFor(n)) {
        if (options.gramSize !== null && gramSize !== options.gramSize) continue
        const corpus = corpusOf(corpusClass, n, gramSize)
        const built = buildIndex(corpus.choices, gramSize)
        for (const row of counterRows(n, corpusClass, gramSize, corpus, built)) {
          process.stdout.write(`${JSON.stringify(row)}\n`)
        }
      }
    }
  }
}
