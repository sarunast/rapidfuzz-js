/**
 * The inverted n-gram index against uFuzzy, on the one job they both do:
 * narrow a large list of strings down to the few worth showing.
 *
 * ```sh
 * node bench/comparison/ngram-index.mjs
 * node bench/comparison/ngram-index.mjs --max=1000000
 * ```
 *
 * **They do not compute the same thing, and this is not a correctness
 * comparison.** uFuzzy matches a subsequence — the needle's characters in
 * order, with bounded gaps — and ranks by where they landed. Dice measures
 * n-gram overlap and ignores position entirely, so `'new york mets'` and
 * `'mets new york'` are near-identical to it and unrelated to uFuzzy. Only the
 * time to filter and rank N candidates is comparable, which is why no agreement
 * check runs here and the match counts are printed instead: they are what shows
 * the two answering different questions.
 *
 * What it is really asking is architectural. Ours visits only candidates that
 * share an n-gram; uFuzzy visits every candidate with a very cheap test — a
 * compiled RegExp run by the engine. Both beat scoring every candidate with a
 * real metric, and the question is by how much, and from what size.
 *
 * The prototype is TypeScript reaching into `src/`, so it is bundled here the
 * way `bench/tooling/ngram-index-scale.ts` bundles its own payload: node cannot
 * resolve the `.js` specifiers `src/` carries.
 */

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import process from 'node:process'
import { fileURLToPath, pathToFileURL } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = dirname(dirname(here))

let uFuzzy
try {
  ;({ default: uFuzzy } = await import('@leeoniya/ufuzzy'))
} catch {
  console.error(
    'uFuzzy is not installed. It is a comparison contender, not a root dependency:\n' +
      '  cd bench/comparison && pnpm install\n',
  )
  process.exit(1)
}

const GRAM_SIZE = 3
const THRESHOLD = 0.5
const LIMIT = 5

// ---------------------------------------------------------------- our arms

/**
 * One bundle holding the prototype, the metric and the Matcher, built on the
 * fly. esbuild resolves `.js` specifiers back to `.ts`; node does not.
 */
async function loadOurs() {
  const { build } = await import('esbuild')
  const directory = mkdtempSync(join(tmpdir(), 'ngram-index-comparison-'))
  const entry = join(directory, 'entry.ts')
  const outfile = join(directory, 'bundle.mjs')
  writeFileSync(
    entry,
    [
      `export { NGramIndex } from ${JSON.stringify(join(root, 'bench/tooling/ngramIndex.ts'))}`,
      `export { buildProfile } from ${JSON.stringify(join(root, 'src/algorithms/shared/ngram.ts'))}`,
      `export { similarity as diceMetric } from ${JSON.stringify(join(root, 'src/algorithms/dice/index.ts'))}`,
      `export { createMatcher, createScorer } from ${JSON.stringify(join(root, 'src/index.ts'))}`,
    ].join('\n'),
  )
  try {
    await build({
      entryPoints: [entry],
      outfile,
      bundle: true,
      platform: 'node',
      format: 'esm',
      target: 'node22',
      minify: false,
      sourcemap: false,
      logLevel: 'silent',
    })
    return await import(pathToFileURL(outfile).href)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
}

const ours = await loadOurs()

// ---------------------------------------------------------------- the corpus

/** xorshift32, so every run sees byte-identical input. */
function rng(seed) {
  let state = seed >>> 0
  return () => {
    state ^= state << 13
    state >>>= 0
    state ^= state >>> 17
    state ^= state << 5
    state >>>= 0
    return state / 0x1_0000_0000
  }
}

const LOWER = [...'abcdefghijklmnopqrstuvwxyz']

function word(next, length) {
  const characters = new Array(length)
  for (let index = 0; index < length; index++) {
    characters[index] = LOWER[Math.floor(next() * LOWER.length)]
  }
  return characters.join('')
}

/**
 * Sentences drawn from a Zipf-weighted vocabulary — a few words carry most of
 * the text, as in real prose. A uniform corpus would give every n-gram the same
 * posting length and flatter the index; it is skew that decides this.
 */
function buildCorpus(count) {
  const next = rng(0x0d15_ea5e)
  const vocabulary = []
  for (let index = 0; index < 400; index++) {
    vocabulary.push(word(next, 3 + Math.floor(next() * 6)))
  }
  const weights = vocabulary.map((_value, rank) => 1 / (rank + 1))
  const total = weights.reduce((sum, weight) => sum + weight, 0)
  const pick = () => {
    let target = next() * total
    for (let rank = 0; rank < weights.length; rank++) {
      target -= weights[rank]
      if (target <= 0) return vocabulary[rank]
    }
    return vocabulary[weights.length - 1]
  }
  const choices = []
  for (let index = 0; index < count; index++) {
    const parts = []
    for (let part = 0; part < 4; part++) parts.push(pick())
    choices.push(parts.join(' '))
  }
  const phrase = choices[Math.floor(count / 2)]
  const typo = (value) => {
    const characters = [...value]
    const at = Math.floor(characters.length / 2)
    characters[at] = characters[at] === 'a' ? 'b' : 'a'
    return characters.join('')
  }
  return {
    choices,
    queries: [
      ['whole phrase', phrase],
      ['common word', vocabulary[0]],
      ['rare word', vocabulary[vocabulary.length - 1]],
      ['typo in a word', typo(vocabulary[0])],
      ['unrelated', 'qxzjvwkqxzjv'],
    ],
  }
}

// ---------------------------------------------------------------- timing

const sink = { value: undefined }

function time(runs, body) {
  for (let run = 0; run < Math.max(3, runs); run++) sink.value = body()
  const samples = []
  for (let run = 0; run < runs; run++) {
    const started = process.hrtime.bigint()
    sink.value = body()
    samples.push(Number(process.hrtime.bigint() - started) / 1e6)
  }
  samples.sort((left, right) => left - right)
  return samples[samples.length >> 1]
}

function formatMs(value) {
  return value >= 10
    ? value.toFixed(1)
    : value >= 0.1
      ? value.toFixed(3)
      : value.toFixed(4)
}

// ---------------------------------------------------------------- the contest

const sizes = [1_000, 10_000, 100_000]
for (const argument of process.argv.slice(2)) {
  if (argument.startsWith('--max=')) {
    const max = Number(argument.slice('--max='.length))
    if (max >= 1_000_000) sizes.push(1_000_000)
  }
}

const scorer = ours.createScorer(ours.diceMetric, { gramSize: GRAM_SIZE })
const searcher = new uFuzzy()

for (const count of sizes) {
  const corpus = buildCorpus(count)
  const index = new ours.NGramIndex(GRAM_SIZE, count)
  const startedIndex = process.hrtime.bigint()
  for (let id = 0; id < count; id++) {
    index.add(id, ours.buildProfile(corpus.choices[id], GRAM_SIZE))
  }
  index.compact()
  const indexBuild = Number(process.hrtime.bigint() - startedIndex) / 1e6

  const startedMatcher = process.hrtime.bigint()
  const matcher = count <= 100_000 ? ours.createMatcher(corpus.choices, { scorer }) : null
  const matcherBuild =
    matcher === null ? null : Number(process.hrtime.bigint() - startedMatcher) / 1e6

  const runs = count >= 100_000 ? 5 : 25
  console.log(
    `\n  ${count.toLocaleString()} choices — setup: index ${formatMs(indexBuild)}ms, ` +
      `Matcher ${matcherBuild === null ? 'n/a' : `${formatMs(matcherBuild)}ms`}, uFuzzy 0ms (it builds nothing)`,
  )
  console.log(
    `  ${'query'.padEnd(16)}${'uFuzzy'.padStart(11)}${'uFuzzy filter'.padStart(15)}` +
      `${'ours: index'.padStart(13)}${'ours: prefix'.padStart(14)}${'ours: scan all'.padStart(16)}   matches (uFuzzy / ours)`,
  )
  for (const [name, query] of corpus.queries) {
    const full = time(runs, () => searcher.search(corpus.choices, query))
    const filtered = time(runs, () => searcher.filter(corpus.choices, query))
    const indexed = time(runs, () =>
      index.diceSearch(ours.buildProfile(query, GRAM_SIZE), THRESHOLD, LIMIT),
    )
    const prefixed = time(runs, () =>
      index.dicePrefixSearch(ours.buildProfile(query, GRAM_SIZE), THRESHOLD, LIMIT),
    )
    const scanned =
      matcher === null
        ? null
        : time(runs, () => matcher.search(query, { limit: LIMIT, threshold: THRESHOLD }))
    const theirMatches = searcher.filter(corpus.choices, query)?.length ?? 0
    const ourMatches = index.diceSearch(
      ours.buildProfile(query, GRAM_SIZE),
      THRESHOLD,
      null,
    ).length
    console.log(
      `  ${name.padEnd(16)}${formatMs(full).padStart(11)}${formatMs(filtered).padStart(15)}` +
        `${formatMs(indexed).padStart(13)}${formatMs(prefixed).padStart(14)}` +
        `${(scanned === null ? 'n/a' : formatMs(scanned)).padStart(16)}   ` +
        `${theirMatches} / ${ourMatches}`,
    )
  }
}

console.log(
  '\n  Times are milliseconds for one query, median of a warmed loop. Lower is better.\n' +
    '  The two answer different questions — see the header of this file — so the\n' +
    '  match counts are there to keep the timings from reading as a like-for-like.\n',
)
