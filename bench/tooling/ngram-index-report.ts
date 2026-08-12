/**
 * Parity, structural counters and retained memory for the inverted n-gram index
 * prototype, at corpus sizes the bench harness cannot hold.
 *
 * Run through `ngram-index-scale.ts`, which bundles this file — it reaches into
 * `src/`, so node cannot execute it directly.
 *
 * Every measuring mode runs a smoke parity suite first, under the same build and
 * key configuration it is about to measure, and throws on the first mismatch: a
 * latency number for a wrong result is worth nothing. `--parity` is the
 * exhaustive version of the same check. Rows are written as JSON lines as they
 * are produced, so an interrupted run still leaves usable output, and each row
 * carries the configuration that produced it — two runs differing only in
 * `--build` or `--keys` are otherwise indistinguishable after the fact.
 */

import { type Dirent, readdirSync } from 'node:fs'
import process from 'node:process'
import { pathToFileURL } from 'node:url'

import { similarity as cosineMetric } from '../../src/algorithms/cosine/index.js'
import { similarity as diceMetric } from '../../src/algorithms/dice/index.js'
import { buildProfile, NGramProfile } from '../../src/algorithms/shared/ngram.js'
import { createMatcher, createScorer } from '../../src/index.js'
import {
  DENSE_CUTOFF,
  feasibleRadices,
  NGramIndex,
  type IndexCounters,
  type Scored,
} from './ngramIndex.js'

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

// `createMatcher` takes a `readonly TItem[]`, so the corpus goes in as it is. A
// defensive copy here would be charged to `matcherBuildMs`, and no caller
// comparing an index against a Matcher would ever make one.
function matcherFor(
  metric: Metric,
  gramSize: number,
  choices: readonly string[],
): ExhaustiveMatcher {
  return metric === 'dice'
    ? createMatcher(choices, { scorer: createScorer(diceMetric, { gramSize }) })
    : createMatcher(choices, { scorer: createScorer(cosineMetric, { gramSize }) })
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
type CorpusClass = `alphabet-${number}` | 'zipf-words' | 'file-paths'

const CORPUS_CLASSES: readonly CorpusClass[] = [
  'alphabet-2',
  'alphabet-26',
  'zipf-words',
  'file-paths',
]

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

function queryClassOf(value: string): QueryClass {
  for (const queryClass of QUERY_CLASSES) {
    if (queryClass === value) return queryClass
  }
  throw new RangeError(`--query must be one of ${QUERY_CLASSES.join(', ')}`)
}

/**
 * How many distinct queries each class carries. One query per class, timed in a
 * loop, touches exactly the same posting slices on every iteration — which the
 * index likes and the exhaustive arm, walking the whole corpus regardless,
 * cannot benefit from. Cycling through several spreads the footprint. Counters
 * are still read from the first, since they describe one query.
 */
const QUERY_VARIANTS = 8

interface Corpus {
  readonly choices: string[]
  readonly queries: ReadonlyMap<QueryClass, readonly string[]>
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

/**
 * `count` substitutions at `count` *distinct* positions, so the "2 typos" row
 * measures two typos.
 *
 * Picking each position independently let two edits land on the same character,
 * and on a small alphabet the second could restore the first — a "2 typos" query
 * that was an exact hit. A partial Fisher-Yates over the positions samples
 * without replacement, which is the only version of this the row's label is true
 * of.
 */
function substitute(
  next: () => number,
  source: string,
  count: number,
  alphabet: readonly string[],
): string {
  const characters = [...source]
  const positions: number[] = characters.map((_value, index) => index)
  const edits = Math.min(count, characters.length)
  for (let edit = 0; edit < edits; edit++) {
    const picked = edit + Math.floor(next() * (positions.length - edit))
    const swap = positions[edit]
    positions[edit] = positions[picked]
    positions[picked] = swap
    const at = positions[edit]
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
  const hits = spread(choices, vocabulary[0])
  const variants = (build: (hit: string, at: number) => string): string[] =>
    hits.map(build)
  return {
    choices,
    queries: new Map<QueryClass, readonly string[]>([
      ['exact hit', hits],
      ['1 typo', variants((hit) => substitute(next, hit, 1, LOWER))],
      ['2 typos', variants((hit) => substitute(next, hit, 2, LOWER))],
      ['unrelated', variants(() => `${word(next, 6, LOWER)} ${word(next, 7, LOWER)}`)],
      ['short', variants((hit) => hit.slice(0, gramSize + 2))],
      // Ranked vocabularies, so "common" is the head of the Zipf curve and
      // "rare" its tail — the pair the uniform corpora cannot separate.
      ['common substring', variants((_hit, at) => `${vocabulary[at]} ${vocabulary[at]}`)],
      [
        'rare substring',
        variants((_hit, at) => {
          const rare = vocabulary[vocabulary.length - 1 - at]
          return `${rare} ${rare}`
        }),
      ],
    ]),
    separatesFrequency: true,
  }
}

/**
 * `QUERY_VARIANTS` entries drawn evenly across the corpus rather than from one
 * end: adjacent choices in these corpora were generated back to back, so a
 * contiguous slice would share more posting lists with itself than the corpus
 * average.
 */
function spread(choices: readonly string[], fallback: string): string[] {
  const picked: string[] = []
  for (let at = 0; at < QUERY_VARIANTS; at++) {
    const index = Math.floor((choices.length * (at + 0.5)) / QUERY_VARIANTS)
    picked.push(choices[index] ?? fallback)
  }
  return picked
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
  const hits = spread(choices, word(next, 24, alphabet))
  const variants = (build: (hit: string, at: number) => string): string[] =>
    hits.map(build)
  return {
    choices,
    queries: new Map<QueryClass, readonly string[]>([
      ['exact hit', hits],
      ['1 typo', variants((hit) => substitute(next, hit, 1, alphabet))],
      ['2 typos', variants((hit) => substitute(next, hit, 2, alphabet))],
      ['unrelated', variants(() => word(next, 24, alphabet))],
      ['short', variants((hit) => hit.slice(0, gramSize + 2))],
      ['common substring', variants((hit) => hit.slice(0, 12))],
      ['rare substring', variants((hit) => hit.slice(12))],
    ]),
    separatesFrequency: false,
  }
}

/**
 * Real file paths, which is what a fuzzy finder is actually pointed at: mixed
 * case, punctuation, long shared prefixes, and a segment-frequency curve nobody
 * chose. The synthetic sweep says where the crossover is; this says which side
 * of it a real workload sits on.
 *
 * Derived from the checkout rather than committed — `node_modules` after
 * `pnpm install` is ~13k paths — and sorted, so a run is reproducible on the
 * same install. It does not tile to reach a requested size: repeating a corpus
 * would hand the index duplicate posting entries it would never see in life.
 */
function filePathCorpus(count: number, gramSize: number): Corpus {
  const paths = realPaths()
  if (paths.length < count) {
    throw new RangeError(
      `the file-path corpus holds ${paths.length} entries, short of ${count}`,
    )
  }
  const choices = sampleEvenly(paths, count)
  const next = rng(0x0c0f_fee1)
  const hits = spread(choices, 'index.js')
  const segments = new Map<string, number>()
  for (const each of choices) {
    for (const segment of each.split('/')) {
      segments.set(segment, (segments.get(segment) ?? 0) + 1)
    }
  }
  const ranked = [...segments.entries()]
    .filter(([segment]) => segment.length >= gramSize)
    .sort((left, right) => right[1] - left[1] || (left[0] < right[0] ? -1 : 1))
  const at = (index: number): string => ranked[index]?.[0] ?? 'index.js'
  const variants = (build: (hit: string, index: number) => string): string[] =>
    hits.map(build)
  return {
    choices,
    queries: new Map<QueryClass, readonly string[]>([
      ['exact hit', hits],
      ['1 typo', variants((hit) => substitute(next, hit, 1, LOWER))],
      ['2 typos', variants((hit) => substitute(next, hit, 2, LOWER))],
      ['unrelated', variants(() => `${word(next, 8, LOWER)}/${word(next, 11, LOWER)}`)],
      [
        'short',
        variants((hit) =>
          hit.slice(hit.lastIndexOf('/') + 1, hit.lastIndexOf('/') + 1 + gramSize + 2),
        ),
      ],
      ['common substring', variants((_hit, index) => at(index))],
      ['rare substring', variants((_hit, index) => at(ranked.length - 1 - index))],
    ]),
    separatesFrequency: true,
  }
}

/**
 * `count` entries spread proportionally across the whole of `source`.
 *
 * An integer stride does not do this: at 10,000 of 12,947 the stride floors to
 * 1, which is the sorted prefix the striding was there to avoid, and at 1,000 it
 * is 12, leaving the last 958 paths unreachable. Proportional positions cover
 * the range at every size, and are the identity when the whole corpus is asked
 * for. Same rule as `spread`.
 */
function sampleEvenly(source: readonly string[], count: number): string[] {
  const picked: string[] = new Array<string>(count)
  for (let at = 0; at < count; at++) {
    picked[at] = source[Math.floor((source.length * (at + 0.5)) / count)]
  }
  return picked
}

let cachedPaths: readonly string[] | null = null

function realPaths(): readonly string[] {
  if (cachedPaths !== null) return cachedPaths
  // From the working directory, not `import.meta.url`: the launcher bundles this
  // file into a temp directory, so a module-relative path would resolve there.
  const root = new URL('node_modules/', pathToFileURL(`${process.cwd()}/`))
  const found: string[] = []
  const walk = (directory: URL, prefix: string): void => {
    let entries: readonly Dirent[]
    try {
      entries = readdirSync(directory, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      const name = `${prefix}${entry.name}`
      if (entry.isDirectory()) walk(new URL(`${entry.name}/`, directory), `${name}/`)
      else if (entry.isFile()) found.push(name)
    }
  }
  walk(root, '')
  found.sort()
  cachedPaths = found
  return found
}

/**
 * Real corpora do not tile, so they cap the sizes they can appear at. Returned
 * rather than thrown, because a run that asks for every size should measure the
 * real corpus where it exists and skip it above that, not fail.
 */
function maxSizeFor(kind: CorpusClass): number {
  return kind === 'file-paths' ? realPaths().length : Number.POSITIVE_INFINITY
}

function corpusOf(kind: CorpusClass, count: number, gramSize: number): Corpus {
  if (kind === 'zipf-words') return zipfCorpus(count, gramSize)
  if (kind === 'file-paths') return filePathCorpus(count, gramSize)
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
 * `direct` extracts grams straight from the text; `profile` goes the long way
 * round, building an `NGramProfile` per choice and dropping it — which is what a
 * Matcher pays during construction, and so the fair comparison for build cost.
 */
type BuildMode = 'profile' | 'direct'

/**
 * Which rung of the packed-key ladder an index starts on, or `string` for the
 * joined-string keys. `auto` starts on the narrowest rung the gram size allows —
 * a byte, up to gram size 6 — and widens when an element does not fit.
 */
type KeyMode = 'auto' | 'bmp' | 'full' | 'string'

/**
 * One value rather than three loose flags, and it travels into every row this
 * script emits. Two runs differing only in `--keys` produce byte-identical rows
 * otherwise, so a JSON file that does not carry this is a file whose numbers
 * cannot be attributed afterwards.
 */
interface ExperimentConfig {
  readonly buildMode: BuildMode
  readonly keyMode: KeyMode
  /**
   * The share of the corpus a posting list covers before it is stored inverted,
   * or `null` for the all-sparse representation this is measured against.
   */
  readonly denseCutoff: number | null
  /**
   * Posting ids in the narrowest word the corpus fits in. Part of the config
   * rather than a bare flag because the fixed parity corpora are all small
   * enough for `Uint16`, so nothing would ever exercise the wide arm otherwise.
   */
  readonly narrowIds: boolean
  /**
   * Dice's shared counts in an `Int32Array`. Dice only — an index built this way
   * refuses Cosine, so the parity sweep runs half the metrics for these.
   */
  readonly narrowAccumulator: boolean
}

const START_RADIX: Readonly<Record<KeyMode, number | null>> = {
  auto: null,
  bmp: 0x1_0000,
  full: 0x11_0000,
  string: null,
}

/**
 * Whether a pinned rung can hold a gram of this depth. The ladder's reach falls
 * as depth rises — a byte to six elements, a BMP word to three, the full
 * code-point range to two — so `--keys=full` is a trigram request the packing
 * cannot answer, and a parity sweep has to skip that pair rather than fail on it.
 */
function supports(config: ExperimentConfig, gramSize: number): boolean {
  const rung = START_RADIX[config.keyMode]
  return (
    config.keyMode === 'string' ||
    rung === null ||
    feasibleRadices(gramSize).includes(rung)
  )
}

function indexFor(
  gramSize: number,
  choiceCount: number,
  config: ExperimentConfig,
): NGramIndex {
  return new NGramIndex(
    gramSize,
    choiceCount,
    config.keyMode !== 'string',
    START_RADIX[config.keyMode],
    config.denseCutoff,
    config.narrowIds,
    config.narrowAccumulator,
  )
}

function buildIndex(
  choices: readonly string[],
  gramSize: number,
  config: ExperimentConfig,
): BuiltIndex {
  const started = process.hrtime.bigint()
  const index = indexFor(gramSize, choices.length, config)
  if (config.buildMode === 'direct') {
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
  readonly config: ExperimentConfig
}

const BUILD_MODES = ['profile', 'direct'] as const
const KEY_MODES = ['auto', 'bmp', 'full', 'string'] as const

/**
 * The whole product, generated rather than listed — a hand-written list said it
 * covered the product while missing `profile + bmp` and `direct + full`, and the
 * two builders are not interchangeable: `add` widens in a loop, `addSequence`
 * restarts through recursion, so the same pinned rung exercises different code
 * on each. `supports` drops the pairs a gram size cannot hold.
 */
const DENSE_CUTOFFS: readonly (number | null)[] = [null, DENSE_CUTOFF]
const ID_WIDTHS: readonly boolean[] = [true, false]
const ACCUMULATOR_WIDTHS: readonly boolean[] = [false, true]

const CONFIGS: readonly ExperimentConfig[] = BUILD_MODES.flatMap((buildMode) =>
  KEY_MODES.flatMap((keyMode) =>
    DENSE_CUTOFFS.flatMap((denseCutoff) =>
      ID_WIDTHS.flatMap((narrowIds) =>
        ACCUMULATOR_WIDTHS.map((narrowAccumulator) => ({
          buildMode,
          keyMode,
          denseCutoff,
          narrowIds,
          narrowAccumulator,
        })),
      ),
    ),
  ),
)

function label(each: ParityCase, call: 'search' | 'best'): string {
  return `${call} ${JSON.stringify(each)}`
}

function checkCase(each: ParityCase): void {
  const { metric, gramSize, choices, query, threshold, limit, config } = each
  const matcher = matcherFor(metric, gramSize, choices)
  const index = buildIndex(choices, gramSize, config).index
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
  // positive score and a posting-list hit are the same event. A dense list
  // suspends it by design — it hands every candidate a base frequency, so no
  // candidate is untouched and the equivalence has nothing left to say.
  if (profile.gramCount > 0) {
    const all = indexedSearch(index, metric, profile, null, null)
    const positives = all.filter((entry) => entry.score > 0).length
    if (
      !index.counters.scannedAllCandidates &&
      positives !== index.counters.candidatesTouched
    ) {
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

function fixedParity(
  configs: readonly ExperimentConfig[],
  corpora: readonly (readonly string[])[],
  queries: readonly string[],
): number {
  let cases = 0
  for (const config of configs) {
    for (const metric of METRICS) {
      // A narrow accumulator is a Dice representation and throws on Cosine.
      if (config.narrowAccumulator && metric !== 'dice') continue
      for (const gramSize of [2, 3]) {
        if (!supports(config, gramSize)) continue
        for (const choices of corpora) {
          for (const query of queries) {
            for (const threshold of THRESHOLDS) {
              for (const limit of LIMITS) {
                checkCase({ metric, gramSize, choices, query, threshold, limit, config })
                cases++
              }
            }
          }
        }
      }
    }
  }
  return cases
}

/**
 * The cheap check every measuring mode runs before it measures anything: the two
 * corpora that have caught real bugs — astral characters with a lone surrogate,
 * and a repeated-character corpus — against the configuration about to be timed.
 * Milliseconds, against a mode that runs for minutes.
 */
function smokeParity(config: ExperimentConfig): number {
  return fixedParity(
    [config],
    [FIXED_CORPORA[5], FIXED_CORPORA[6], FIXED_CORPORA[7]],
    ['abc', 'banana', '😀abc', 'aaaa'],
  )
}

/**
 * The `Uint16` id width is chosen at `choiceCount <= 0x1_0000`, and every
 * corpus in the matrix above is small enough that only one side of that
 * comparison is ever taken. A wrong bound there does not throw — it wraps an id
 * and answers the wrong choice — so the two sizes either side of it are pinned
 * directly, with the only match placed last so the largest id is the one that
 * has to survive.
 */
function idWidthBoundary(): number {
  let cases = 0
  for (const choiceCount of [0x1_0000, 0x1_0001]) {
    const choices: string[] = new Array(choiceCount).fill('')
    const last = choiceCount - 1
    choices[last] = 'abc'
    for (const config of [
      {
        buildMode: 'direct',
        keyMode: 'auto',
        denseCutoff: DENSE_CUTOFF,
        narrowIds: true,
        narrowAccumulator: false,
      },
      {
        buildMode: 'direct',
        keyMode: 'auto',
        denseCutoff: DENSE_CUTOFF,
        narrowIds: true,
        narrowAccumulator: true,
      },
    ] satisfies ExperimentConfig[]) {
      const index = buildIndex(choices, 3, config).index
      const found = index.diceBest(buildProfile('abc', 3), 0.5)
      if (found === undefined || found.id !== last || found.score !== 1) {
        throw new Error(
          `id width boundary: ${choiceCount} choices answered ${JSON.stringify(found)}, expected id ${last} at score 1`,
        )
      }
      cases++
    }
  }
  return cases
}

/**
 * "An index built narrow refuses Cosine" has to hold for every query shape or
 * it is not a contract. A gramless query is answered before accumulation is
 * reached, so guarding the loop alone left `cosineSearch('')` succeeding on an
 * index that refused every other query.
 */
function narrowRefusesCosine(): number {
  const config: ExperimentConfig = {
    buildMode: 'direct',
    keyMode: 'auto',
    denseCutoff: DENSE_CUTOFF,
    narrowIds: true,
    narrowAccumulator: true,
  }
  const index = buildIndex(['abc', 'abcd', ''], 3, config).index
  let cases = 0
  for (const query of ['abc', '', 'a']) {
    const profile = buildProfile(query, 3)
    for (const call of [
      () => index.cosineBest(profile, null),
      () => index.cosineSearch(profile, null, 3),
      () => index.cosineSearch(profile, 0.5, null),
    ]) {
      let refused = false
      try {
        call()
      } catch (error) {
        refused = error instanceof TypeError
      }
      if (!refused) {
        throw new Error(`a narrow index answered Cosine for ${JSON.stringify(query)}`)
      }
      cases++
    }
  }
  return cases
}

async function parity(runs: number): Promise<void> {
  const cases = fixedParity(CONFIGS, FIXED_CORPORA, FIXED_QUERIES)
  process.stdout.write(
    `${JSON.stringify({ kind: 'parity', mode: 'fixed', cases, configs: CONFIGS.length })}\n`,
  )
  process.stdout.write(
    `${JSON.stringify({
      kind: 'parity',
      mode: 'boundary',
      idWidth: idWidthBoundary(),
      narrowCosine: narrowRefusesCosine(),
    })}\n`,
  )

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
      fc.constantFrom(...CONFIGS),
      (choices, query, threshold, limit, gramSize, metric, config) => {
        if (!supports(config, gramSize)) return true
        if (config.narrowAccumulator && metric !== 'dice') return true
        checkCase({ metric, gramSize, choices, query, threshold, limit, config })
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

function quantile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return 0
  const at = Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))
  return sorted[at]
}

/**
 * `p95` is `null` below `P95_MINIMUM_RUNS`, and that is the honest reading. With
 * 5 samples `sorted[floor(0.95 * 5)]` is `sorted[4]` — the maximum, reported
 * under a percentile's name; with 15 it is still the maximum. A tail number for
 * the large corpora belongs to `bench/ngramIndex.bench.ts`, where adaptive
 * sampling takes hundreds of samples per case, not to a script whose exhaustive
 * arm costs tens of milliseconds a call.
 */
interface Latency {
  readonly p50: number
  readonly p95: number | null
}

const P95_MINIMUM_RUNS = 40

/** Where every timed body's result goes, so V8 cannot delete the work. */
const sink: { value: unknown } = { value: undefined }

/**
 * Warmed before it is timed, and the warmup is the same size as the measurement.
 * Without it the 10k and 100k exhaustive arms reported the same milliseconds for
 * ten times the work — the first call of three was carrying the median. These are
 * still indicative numbers; `bench/ngramIndex.bench.ts` is where the adaptive
 * sampling lives.
 *
 * The body is handed a run number rather than closing over one query, so a case
 * can rotate through several. Timing one query repeatedly rewarms exactly the
 * posting slices that query touches, which flatters the index and does nothing
 * for the exhaustive arm — it walks the whole corpus either way.
 */
function timeQuantiles(runs: number, body: (run: number) => unknown): Latency {
  const warmups = Math.max(3, runs)
  for (let run = 0; run < warmups; run++) sink.value = body(run)
  const samples: number[] = []
  for (let run = 0; run < runs; run++) {
    const started = process.hrtime.bigint()
    sink.value = body(run)
    samples.push(Number(process.hrtime.bigint() - started) / 1e6)
  }
  samples.sort((left, right) => left - right)
  return {
    p50: quantile(samples, 0.5),
    p95: samples.length >= P95_MINIMUM_RUNS ? quantile(samples, 0.95) : null,
  }
}

interface CounterRow {
  readonly kind: 'counters'
  readonly n: number
  readonly corpus: CorpusClass
  readonly gramSize: number
  readonly metric: Metric
  readonly queryClass: QueryClass
  readonly separatesFrequency: boolean
  readonly buildMode: BuildMode
  readonly keyMode: KeyMode
  readonly denseCutoff: number | null
  readonly narrowIds: boolean
  readonly narrowAccumulator: boolean
  readonly threshold: number
  readonly limit: number
  readonly queryVariants: number
  readonly distinctQueryGrams: number
  readonly postingEntriesTouched: number
  readonly postingsPerChoice: number
  readonly postingsPerChoicePerGram: number
  readonly candidatesTouched: number
  readonly candidatesTouchedRatio: number
  /** Candidates the accumulation wrote to; null under a dense scan, which no
   * longer tracks them. */
  readonly modifiedCandidates: number | null
  readonly candidatesQualified: number
  readonly indexedMs: number
  /** Null below `P95_MINIMUM_RUNS` samples — see {@link Latency}. */
  readonly indexedP95Ms: number | null
  readonly exhaustiveMs: number | null
  readonly exhaustiveP95Ms: number | null
  /** The prefix-filtered Dice path, where it applies; null for Cosine. */
  readonly filteredMs: number | null
  readonly filteredP95Ms: number | null
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

const COUNTER_LIMIT = 5
/** Above this, a Matcher's profiles no longer fit beside the index. */
const EXHAUSTIVE_LIMIT = 100_000

/**
 * Below 10k a query is cheap enough to sample past `P95_MINIMUM_RUNS`, so those
 * rows carry a real tail. Above it the exhaustive arm is tens of milliseconds a
 * call and would set the length of the whole pass; those rows report `p50` and a
 * null `p95` rather than a maximum pretending to be one.
 */
function counterRuns(n: number): number {
  if (n >= 100_000) return 5
  if (n >= 10_000) return 15
  return 41
}

function counterRows(
  n: number,
  corpusClass: CorpusClass,
  gramSize: number,
  corpus: Corpus,
  built: BuiltIndex,
  config: ExperimentConfig,
  threshold: number,
): CounterRow[] {
  const index = built.index
  const statistics = index.postingStatistics()
  const runs = counterRuns(n)
  const rows: CounterRow[] = []
  for (const metric of METRICS) {
    if (config.narrowAccumulator && metric !== 'dice') continue
    let matcher: ExhaustiveMatcher | null = null
    let matcherBuildMs: number | null = null
    if (n <= EXHAUSTIVE_LIMIT) {
      const started = process.hrtime.bigint()
      matcher = matcherFor(metric, gramSize, corpus.choices)
      matcherBuildMs = Number(process.hrtime.bigint() - started) / 1e6
      if (matcher.size !== n) throw new Error('matcher lost choices')
    }
    for (const queryClass of QUERY_CLASSES) {
      const variants = corpus.queries.get(queryClass)
      if (variants === undefined || variants.length === 0) {
        throw new Error(`missing query class ${queryClass}`)
      }
      // Counters describe one query, so they come from the first variant. The
      // timed runs rotate, and each rebuilds the query profile, because a real
      // query would.
      const rotate = (run: number): string => variants[run % variants.length]
      indexedSearch(
        index,
        metric,
        buildProfile(variants[0], gramSize),
        threshold,
        COUNTER_LIMIT,
      )
      const counters = { ...index.counters }
      const indexed = timeQuantiles(runs, (run) =>
        indexedSearch(
          index,
          metric,
          buildProfile(rotate(run), gramSize),
          threshold,
          COUNTER_LIMIT,
        ),
      )
      const held = matcher
      const exhaustive =
        held === null
          ? null
          : timeQuantiles(runs, (run) =>
              held.search(rotate(run), { limit: COUNTER_LIMIT, threshold }),
            )
      let filtered: Latency | null = null
      let filteredCounters: IndexCounters | null = null
      if (metric === 'dice') {
        index.dicePrefixSearch(
          buildProfile(variants[0], gramSize),
          threshold,
          COUNTER_LIMIT,
        )
        filteredCounters = { ...index.counters }
        filtered = timeQuantiles(runs, (run) =>
          index.dicePrefixSearch(
            buildProfile(rotate(run), gramSize),
            threshold,
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
        buildMode: config.buildMode,
        keyMode: config.keyMode,
        denseCutoff: config.denseCutoff,
        narrowIds: config.narrowIds,
        narrowAccumulator: config.narrowAccumulator,
        threshold,
        limit: COUNTER_LIMIT,
        queryVariants: variants.length,
        distinctQueryGrams: counters.distinctQueryGrams,
        postingEntriesTouched: counters.postingEntriesTouched,
        postingsPerChoice: counters.postingEntriesTouched / n,
        postingsPerChoicePerGram:
          counters.distinctQueryGrams === 0
            ? 0
            : counters.postingEntriesTouched / (n * counters.distinctQueryGrams),
        candidatesTouched: counters.candidatesTouched,
        candidatesTouchedRatio: counters.candidatesTouched / n,
        modifiedCandidates: counters.modifiedCandidates,
        candidatesQualified: counters.candidatesQualified,
        indexedMs: indexed.p50,
        indexedP95Ms: indexed.p95,
        exhaustiveMs: exhaustive === null ? null : exhaustive.p50,
        exhaustiveP95Ms: exhaustive === null ? null : exhaustive.p95,
        filteredMs: filtered === null ? null : filtered.p50,
        filteredP95Ms: filtered === null ? null : filtered.p95,
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

/**
 * Three arms, because there are two honest comparisons and they answer different
 * questions.
 *
 * `index` against `profiles` is the **representation** question — one corpus-wide
 * inverted structure against the N prepared `NGramProfile`s it would replace, and
 * nothing else on either side. That is the architectural claim.
 *
 * `index` against `matcher` is the **product** question — what a caller actually
 * retains. A Matcher also keeps a row per choice holding the item, its key and
 * its prepared value, so this arm is larger than `profiles` by an amount that
 * grows with N. Reporting only this one against a bare index would flatter the
 * index by charging the Matcher for bookkeeping an indexed Matcher would still
 * have to do.
 */
type Arm = 'index' | 'profiles' | 'matcher'

interface MemoryRow {
  readonly kind: 'memory'
  readonly n: number
  readonly corpus: CorpusClass
  readonly gramSize: number
  readonly arm: Arm
  readonly buildMode: BuildMode
  readonly keyMode: KeyMode
  readonly denseCutoff: number | null
  readonly narrowIds: boolean
  readonly narrowAccumulator: boolean
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
  config: ExperimentConfig,
): MemoryRow {
  // 331 MiB per 100k prepared bigram profiles extrapolates to ~3.3 GiB at a
  // million, which measures the collector rather than the representation. The
  // 1M profile figure belongs in the writeup as an extrapolation, labelled.
  if (arm !== 'index' && n > 100_000) {
    throw new RangeError('the profile arms stop at 100k — extrapolate above them')
  }
  const corpus = corpusOf(corpusClass, n, gramSize)
  collect()
  const before = retainedBytes()
  let held: number = 0
  if (arm === 'index') {
    const built = buildIndex(corpus.choices, gramSize, config)
    collect()
    held = retainedBytes() - before
    if (built.index.choiceCount !== n) throw new Error('index lost choices')
    if (n <= 100_000 && retainsProfile(built.index, 0)) {
      throw new Error('the index retained an NGramProfile — the memory claim is void')
    }
  } else if (arm === 'profiles') {
    const profiles: NGramProfile[] = new Array<NGramProfile>(n)
    for (let id = 0; id < n; id++)
      profiles[id] = buildProfile(corpus.choices[id], gramSize)
    collect()
    held = retainedBytes() - before
    if (profiles.length !== n) throw new Error('profiles lost choices')
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
    buildMode: config.buildMode,
    keyMode: config.keyMode,
    denseCutoff: config.denseCutoff,
    narrowIds: config.narrowIds,
    narrowAccumulator: config.narrowAccumulator,
    bytes: held,
    bytesPerChoice: held / n,
  }
}

interface PeakRow {
  readonly kind: 'peak'
  readonly n: number
  readonly corpus: CorpusClass
  readonly gramSize: number
  readonly buildMode: BuildMode
  readonly keyMode: KeyMode
  readonly denseCutoff: number | null
  readonly narrowIds: boolean
  readonly narrowAccumulator: boolean
  /**
   * Everything held before the build starts: the corpus, plus the runtime and
   * module heap under it. Not the corpus's size — nothing here measures that.
   */
  readonly baselineBytes: number
  readonly peakHeapBytes: number
  readonly peakArrayBufferBytes: number
  readonly peakRssBytes: number
  /** `heapUsed + arrayBuffers` at its highest, the same sum `--memory` reports. */
  readonly peakRetainedBytes: number
  /** Peak retained over the baseline: what the build itself needed. */
  readonly peakBuildBytes: number
  /** Retained over the baseline, matching what `--memory` reports. */
  readonly retainedBytes: number
  readonly buildMs: number
}

/**
 * What the build costs at its worst, not what it leaves behind.
 *
 * Until `compact()` runs, every posting list is a pair of growable JS arrays
 * inside a Map, so the heap at that moment is nothing like the CSR arrays the
 * index settles into. The retained figure is the architectural claim; this is
 * the number that decides whether a corpus can be indexed at all on a given
 * machine, and it is the one thing the retained measurement cannot show.
 *
 * Sampled inside the build loop rather than on a timer: the build is one
 * synchronous run, so nothing else would ever get to look.
 */
function measurePeak(
  n: number,
  corpusClass: CorpusClass,
  gramSize: number,
  config: ExperimentConfig,
): PeakRow {
  const corpus = corpusOf(corpusClass, n, gramSize)
  collect()
  // A million 24-character strings are tens of megabytes on their own, and they
  // are the caller's, not the index's. Every figure below is over this line.
  const baseline = retainedBytes()
  let peakHeap = 0
  let peakBuffers = 0
  let peakRss = 0
  // Tracked as one sum per sample, not as two independent maxima. Subtracting a
  // `heapUsed + arrayBuffers` baseline from a `heapUsed` peak is not the build's
  // peak of anything — it undercounts by whatever the corpus already held in
  // buffers, and the two maxima need not occur at the same instant either.
  let peakRetained = baseline
  const sample = (): void => {
    const usage = process.memoryUsage()
    const retained = usage.heapUsed + usage.arrayBuffers
    if (retained > peakRetained) peakRetained = retained
    if (usage.heapUsed > peakHeap) peakHeap = usage.heapUsed
    if (usage.arrayBuffers > peakBuffers) peakBuffers = usage.arrayBuffers
    if (usage.rss > peakRss) peakRss = usage.rss
  }
  const every = Math.max(1, Math.floor(n / 1_000))
  const index = indexFor(gramSize, n, config)
  const started = process.hrtime.bigint()
  for (let id = 0; id < n; id++) {
    if (config.buildMode === 'direct') index.addSequence(id, corpus.choices[id])
    else index.add(id, buildProfile(corpus.choices[id], gramSize))
    if (id % every === 0) sample()
  }
  sample()
  index.compact()
  sample()
  const buildMs = Number(process.hrtime.bigint() - started) / 1e6
  collect()
  const retained = retainedBytes()
  if (index.choiceCount !== n) throw new Error('index lost choices')
  return {
    kind: 'peak',
    n,
    corpus: corpusClass,
    gramSize,
    buildMode: config.buildMode,
    keyMode: config.keyMode,
    denseCutoff: config.denseCutoff,
    narrowIds: config.narrowIds,
    narrowAccumulator: config.narrowAccumulator,
    baselineBytes: baseline,
    peakHeapBytes: peakHeap,
    peakArrayBufferBytes: peakBuffers,
    peakRssBytes: peakRss,
    peakRetainedBytes: peakRetained,
    peakBuildBytes: peakRetained - baseline,
    retainedBytes: retained - baseline,
    buildMs,
  }
}

/**
 * Would a dense "default frequency 1, store the exceptions" posting pay?
 *
 * A probe before an implementation. The idea attacks the one shape where the
 * index barely beats the Matcher — a gram almost every choice has — by storing
 * the choices that *lack* it. Whether that is worth building depends entirely on
 * how much of the work a query actually does lands in lists dense enough to
 * invert, and nothing measured so far reports that.
 */
function probeDense(
  n: number,
  corpusClass: CorpusClass,
  gramSize: number,
  config: ExperimentConfig,
): void {
  const corpus = corpusOf(corpusClass, n, gramSize)
  // Built all-sparse whatever was asked for: the probe compares the current
  // representation against the one it would become, so it has to start from the
  // current one.
  const sparse: ExperimentConfig = { ...config, denseCutoff: null }
  const index = buildIndex(corpus.choices, gramSize, sparse).index
  const stats = index.postingStatistics()
  const outlook = index.denseOutlook(DENSE_CUTOFF)
  const hybrid = outlook.hybridEntries
  const denseLists = outlook.denseLists
  const share = (value: number): string => `${(value * 100).toFixed(1)}%`
  process.stdout.write(
    `\n  ${corpusClass}, n=${n.toLocaleString()}, gram size ${gramSize}\n` +
      `    posting entries        ${stats.documentEntries.toLocaleString().padStart(12)}\n` +
      `    as a hybrid            ${hybrid.toLocaleString().padStart(12)}` +
      `   (${share(hybrid / Math.max(1, stats.documentEntries))} of it)\n` +
      `    lists worth inverting  ${denseLists.toLocaleString().padStart(12)}` +
      `   of ${stats.distinctGrams.toLocaleString()}\n` +
      `\n    query              grams   dense   sparse work   hybrid work   ratio   touched\n`,
  )
  for (const queryClass of QUERY_CLASSES) {
    const variants = corpus.queries.get(queryClass)
    if (variants === undefined || variants.length === 0) {
      throw new Error(`missing query class ${queryClass}`)
    }
    const probe = index.denseProbe(buildProfile(variants[0], gramSize), DENSE_CUTOFF)
    const ratio =
      probe.hybridWork === 0 ? 1 : probe.sparseWork / Math.max(1, probe.hybridWork)
    process.stdout.write(
      `    ${queryClass.padEnd(17)}${String(probe.queryGrams).padStart(6)}` +
        `${String(probe.denseGrams).padStart(8)}` +
        `${probe.sparseWork.toLocaleString().padStart(14)}` +
        `${probe.hybridWork.toLocaleString().padStart(14)}` +
        `${`${ratio.toFixed(2)}x`.padStart(8)}` +
        `${probe.touched.toLocaleString().padStart(10)}\n`,
    )
  }
}

/**
 * Where the selection scan's time goes, per query class. Dense postings moved
 * `common substring` by 3% after taking 30x off its posting traffic, so this is
 * the loop that decides that query — and nothing had measured inside it.
 */
function profileSelection(
  n: number,
  corpusClass: CorpusClass,
  gramSize: number,
  threshold: number,
  config: ExperimentConfig,
): void {
  const corpus = corpusOf(corpusClass, n, gramSize)
  const index = buildIndex(corpus.choices, gramSize, config).index
  const runs = n >= 100_000 ? 15 : 60
  process.stdout.write(
    `\n  ${corpusClass}, n=${n.toLocaleString()}, gram size ${gramSize}, ` +
      `threshold ${threshold}, dense ${config.denseCutoff === null ? 'off' : 'on'}\n`,
  )
  for (const queryClass of QUERY_CLASSES) {
    const variants = corpus.queries.get(queryClass)
    if (variants === undefined || variants.length === 0) {
      throw new Error(`missing query class ${queryClass}`)
    }
    const phases = index.profilePhases(variants[0], threshold, COUNTER_LIMIT, runs)
    process.stdout.write(`    ${queryClass}\n`)
    let previous = 0
    for (const phase of phases) {
      const step = phase.name.startsWith('accumulate alone')
        ? phase.ms
        : phase.ms - previous
      if (!phase.name.startsWith('accumulate alone')) previous = phase.ms
      process.stdout.write(
        `      ${phase.name.padEnd(34)}${phase.ms.toFixed(4).padStart(9)} ms` +
          `${`(${step >= 0 ? '+' : ''}${step.toFixed(4)})`.padStart(12)}\n`,
      )
    }
    const rows = index.profileSelection(
      buildProfile(variants[0], gramSize),
      threshold,
      runs,
    )
    const floor = rows[0].ms
    for (const row of rows) {
      process.stdout.write(
        `        ${row.name.padEnd(32)}${row.ms.toFixed(4).padStart(9)} ms` +
          `${`+${(row.ms - floor).toFixed(4)}`.padStart(11)}` +
          `${String(row.qualified).padStart(10)}\n`,
      )
    }
  }
}

/**
 * Where the accumulation loop's time goes, per query class, with the posting
 * shape it runs over printed above it — an all-one list share is what decides
 * whether dropping the count load is worth anything.
 */
function profileAccumulation(
  n: number,
  corpusClass: CorpusClass,
  gramSize: number,
  config: ExperimentConfig,
  only: QueryClass | null,
): void {
  const corpus = corpusOf(corpusClass, n, gramSize)
  const index = buildIndex(corpus.choices, gramSize, config).index
  const stats = index.postingStatistics()
  const runs = n >= 100_000 ? 15 : 60
  const percent = (value: number): string => `${(value * 100).toFixed(1)}%`
  process.stdout.write(
    `\n  ${corpusClass}, n=${n.toLocaleString()}, gram size ${gramSize}, ` +
      `dense ${config.denseCutoff === null ? 'off' : 'on'}\n` +
      `    counts width ${stats.countsWidthBytes} byte, ` +
      `entries with count 1 ${percent(stats.singletonEntryShare)}, ` +
      `lists all count 1 ${percent(stats.singletonListShare)}\n`,
  )
  // One class per process is the trustworthy way to read this: the variants are
  // closures over one source each, so profiling a second class reuses whatever
  // optimised code the first class's data shaped — which moved two rungs by
  // 1.8x, reproducibly, on identical loops over identical posting counts.
  for (const queryClass of only === null ? QUERY_CLASSES : [only]) {
    const variants = corpus.queries.get(queryClass)
    if (variants === undefined || variants.length === 0) {
      throw new Error(`missing query class ${queryClass}`)
    }
    const query = buildProfile(variants[0], gramSize)
    index.diceSearch(query, null, COUNTER_LIMIT)
    const counters = { ...index.counters }
    const outlook = index.implicitOutlook(query)
    process.stdout.write(
      `    ${queryClass} — ${counters.distinctQueryGrams} grams, ` +
        `${counters.postingEntriesTouched.toLocaleString()} posting entries, ` +
        `${counters.scannedAllCandidates ? 'dense scan' : 'sparse'}\n` +
        `      count-free lists ${outlook.implicitLists}/${outlook.lists}, ` +
        `entries ${outlook.implicitEntries.toLocaleString()}/${outlook.entries.toLocaleString()}` +
        ` (${percent(outlook.entries === 0 ? 0 : outlook.implicitEntries / outlook.entries)})\n`,
    )
    const rows = index.profileAccumulation(query, runs)
    // The rungs read as steps up from the floor; the two rows past the real
    // method are alternatives to it, so they read against it instead.
    let previous = 0
    let full = 0
    let climbing = true
    for (const row of rows) {
      if (climbing) {
        const step = row.ms - previous
        previous = row.ms
        if (row.name === 'diceAccumulate') {
          full = row.ms
          climbing = false
        }
        process.stdout.write(
          `      ${row.name.padEnd(34)}${row.ms.toFixed(4).padStart(9)} ms` +
            `${`(${step >= 0 ? '+' : ''}${step.toFixed(4)})`.padStart(12)}\n`,
        )
        continue
      }
      process.stdout.write(
        `      ${row.name.padEnd(34)}${row.ms.toFixed(4).padStart(9)} ms` +
          `${`[${(row.ms - full >= 0 ? '+' : '') + (row.ms - full).toFixed(4)}]`.padStart(12)}\n`,
      )
    }
  }
}

/**
 * One corpus, described the way it needs to be read: what the index is, and
 * what one query of each class costs against it, beside the number of
 * candidates the exhaustive Matcher would have scored.
 */
function summarise(
  n: number,
  corpusClass: CorpusClass,
  gramSize: number,
  threshold: number,
  config: ExperimentConfig,
): void {
  const corpus = corpusOf(corpusClass, n, gramSize)
  const built = buildIndex(corpus.choices, gramSize, config)
  const index = built.index
  const stats = index.postingStatistics()
  const percent = (value: number): string => `${(value * 100).toFixed(1)}%`
  process.stdout.write(
    `\n  ${corpusClass}, gram size ${gramSize}, threshold ${threshold}, ` +
      `${config.buildMode} build, ${config.keyMode} keys\n` +
      `    choices                ${n.toLocaleString().padStart(12)}\n` +
      `    distinct grams         ${stats.distinctGrams.toLocaleString().padStart(12)}\n` +
      `    posting entries        ${stats.documentEntries.toLocaleString().padStart(12)}\n` +
      `    stored entries         ${stats.storedEntries.toLocaleString().padStart(12)}\n` +
      `    dense lists            ${stats.denseLists.toLocaleString().padStart(12)}\n` +
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
    const variants = corpus.queries.get(queryClass)
    if (variants === undefined || variants.length === 0) {
      throw new Error(`missing query class ${queryClass}`)
    }
    index.dicePrefixSearch(buildProfile(variants[0], gramSize), threshold, COUNTER_LIMIT)
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

type Mode =
  | 'parity'
  | 'counters'
  | 'memory'
  | 'peak'
  | 'summary'
  | 'dense'
  | 'select'
  | 'accumulate'

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
  readonly query: QueryClass | null
  readonly config: ExperimentConfig
}

function corpusClassOf(value: string): CorpusClass {
  const found = [...CORPUS_CLASSES, ...SWEEP_CLASSES].find(
    (candidate) => candidate === value,
  )
  if (found === undefined) throw new Error(`unknown corpus ${value}`)
  return found
}

function armOf(value: string): Arm {
  if (value === 'index' || value === 'profiles' || value === 'matcher') return value
  throw new Error(`unknown arm ${value} — index, profiles or matcher`)
}

function keyModeOf(value: string): KeyMode {
  if (value === 'auto' || value === 'bmp' || value === 'full' || value === 'string') {
    return value
  }
  throw new Error(`unknown key mode ${value} — auto, bmp, full or string`)
}

function buildModeOf(value: string): BuildMode {
  if (value === 'profile' || value === 'direct') return value
  throw new Error(`unknown build mode ${value} — profile or direct`)
}

function numberOf(argument: string, prefix: string): number {
  const value = Number(argument.slice(prefix.length))
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${prefix} needs a positive safe integer`)
  }
  return value
}

/**
 * `--keys` is one value rather than interacting flags. Two of them used to write
 * to separate variables, so `--keys=bmp --keys=string` left a string-keyed index
 * carrying a pinned BMP rung — a state no single flag asks for and no row
 * recorded.
 */
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
  let query: QueryClass | null = null
  let keyMode: KeyMode = 'auto'
  let buildMode: BuildMode = 'profile'
  let denseCutoff: number | null = DENSE_CUTOFF
  let narrowIds = true
  let narrowAccumulator = false
  for (const argument of process.argv.slice(2)) {
    if (argument === '--parity') mode = 'parity'
    else if (argument === '--counters') mode = 'counters'
    else if (argument === '--memory') mode = 'memory'
    else if (argument === '--peak') mode = 'peak'
    else if (argument === '--summary') mode = 'summary'
    else if (argument === '--dense') mode = 'dense'
    else if (argument === '--select') mode = 'select'
    else if (argument === '--accumulate') mode = 'accumulate'
    else if (argument === '--sweep') sweep = true
    else if (argument === '--wide-ids') narrowIds = false
    else if (argument === '--narrow-accumulator') narrowAccumulator = true
    else if (argument.startsWith('--build=')) {
      buildMode = buildModeOf(argument.slice('--build='.length))
    } else if (argument.startsWith('--keys=')) {
      keyMode = keyModeOf(argument.slice('--keys='.length))
    } else if (argument.startsWith('--dense-cutoff=')) {
      const value = argument.slice('--dense-cutoff='.length)
      if (value === 'off') denseCutoff = null
      else {
        denseCutoff = Number(value)
        if (!(denseCutoff > 0 && denseCutoff <= 1)) {
          throw new RangeError('--dense-cutoff must be inside (0, 1] or "off"')
        }
      }
    } else if (argument.startsWith('--max=')) max = numberOf(argument, '--max=')
    else if (argument.startsWith('--runs=')) runs = numberOf(argument, '--runs=')
    else if (argument.startsWith('--n=')) n = numberOf(argument, '--n=')
    else if (argument.startsWith('--gram=')) {
      gramSize = numberOf(argument, '--gram=')
      // The corpora and the query classes are written for these two depths, and
      // anything else used to parse, then match no size, then print nothing.
      if (gramSize !== 2 && gramSize !== 3) throw new RangeError('--gram must be 2 or 3')
    } else if (argument.startsWith('--threshold=')) {
      threshold = Number(argument.slice('--threshold='.length))
      if (!(threshold > 0 && threshold <= 1)) {
        throw new RangeError('--threshold must be inside (0, 1]')
      }
    } else if (argument.startsWith('--corpus=')) {
      corpus = corpusClassOf(argument.slice('--corpus='.length))
    } else if (argument.startsWith('--arm=')) {
      arm = armOf(argument.slice('--arm='.length))
    } else if (argument.startsWith('--query=')) {
      query = queryClassOf(argument.slice('--query='.length))
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
    query,
    config: { buildMode, keyMode, denseCutoff, narrowIds, narrowAccumulator },
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

/**
 * `--n` names a size, it does not filter the standard ladder. Asking for 50,000
 * used to be accepted, match nothing in `SIZES`, and print an empty run that
 * looked like a finished one.
 */
function sizesFor(chosen: Options): readonly number[] {
  return chosen.n === null ? SIZES.filter((size) => size <= chosen.max) : [chosen.n]
}

function classesFor(chosen: Options): readonly CorpusClass[] {
  const all = chosen.sweep ? SWEEP_CLASSES : CORPUS_CLASSES
  return chosen.corpus === null ? all : [chosen.corpus]
}

function depthsFor(chosen: Options, n: number): readonly number[] {
  return gramSizesFor(n).filter(
    (each) =>
      (chosen.gramSize === null || each === chosen.gramSize) &&
      supports(chosen.config, each),
  )
}

// Before every mode that measures rather than validates, including the memory
// ones: they are not scoring benchmarks, but they build through a specific
// builder and key scheme, and a byte count for a representation that answers
// wrong is no better than a timing for one.
if (options.mode !== 'parity') {
  const smoke = smokeParity(options.config)
  process.stdout.write(
    `${JSON.stringify({ kind: 'parity', mode: 'smoke', cases: smoke, ...options.config })}\n`,
  )
}

if (options.mode === 'parity') {
  await parity(options.runs)
} else if (options.mode === 'memory') {
  const { n, corpus, gramSize, arm, config } = options
  if (n === null || corpus === null || gramSize === null || arm === null) {
    throw new Error('--memory needs --n, --corpus, --gram and --arm')
  }
  process.stdout.write(
    `${JSON.stringify(measureArm(n, corpus, gramSize, arm, config))}\n`,
  )
} else if (options.mode === 'peak') {
  const { n, corpus, gramSize, config } = options
  if (n === null || corpus === null || gramSize === null) {
    throw new Error('--peak needs --n, --corpus and --gram')
  }
  process.stdout.write(`${JSON.stringify(measurePeak(n, corpus, gramSize, config))}\n`)
} else {
  let produced = 0
  for (const n of sizesFor(options)) {
    for (const corpusClass of classesFor(options)) {
      if (n > maxSizeFor(corpusClass)) continue
      for (const gramSize of depthsFor(options, n)) {
        produced++
        if (options.mode === 'summary') {
          summarise(n, corpusClass, gramSize, options.threshold, options.config)
          continue
        }
        if (options.mode === 'dense') {
          probeDense(n, corpusClass, gramSize, options.config)
          continue
        }
        if (options.mode === 'select') {
          profileSelection(n, corpusClass, gramSize, options.threshold, options.config)
          continue
        }
        if (options.mode === 'accumulate') {
          profileAccumulation(n, corpusClass, gramSize, options.config, options.query)
          continue
        }
        const corpus = corpusOf(corpusClass, n, gramSize)
        const built = buildIndex(corpus.choices, gramSize, options.config)
        const rows = counterRows(
          n,
          corpusClass,
          gramSize,
          corpus,
          built,
          options.config,
          options.threshold,
        )
        for (const row of rows) process.stdout.write(`${JSON.stringify(row)}\n`)
      }
    }
  }
  if (produced === 0) {
    throw new Error('no corpus matched the given filters, so nothing was measured')
  }
}
