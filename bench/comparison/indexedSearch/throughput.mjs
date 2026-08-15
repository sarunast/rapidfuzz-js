// @ts-check
/**
 * Searching one collection many times, from a small list to a large one: the
 * two matchers against the JavaScript packages that do the same job.
 *
 * ```sh
 * pnpm build && node bench/comparison/indexedSearch/throughput.mjs
 * node bench/comparison/indexedSearch/throughput.mjs --gram=3
 * node bench/comparison/indexedSearch/throughput.mjs --max=1000000
 * ```
 *
 * Eight arms answer the same question — _which of these N strings best matches
 * this one_ — by different means:
 *
 *   - **`search`**, ours, holding nothing between calls;
 *   - **`createMatcher`**, ours, holding every choice's grams;
 *   - **`createIndexedMatcher`**, ours, holding one inverted structure over the
 *     whole collection, which visits only candidates sharing an n-gram;
 *   - **`dice-coefficient`** handed prebuilt gram arrays — the only other
 *     package here that holds anything, and the fastest scan of the field;
 *   - **`string-similarity.findBestMatch`** and **`fuzzball.extract`**, the two
 *     packages that ship a search API, both of which reprocess the collection
 *     per query;
 *   - **uFuzzy** and **Fuse**, which do not compute Dice at all.
 *
 * The first four agree to the last bit and are checked here before anything is
 * timed: same metric, same corpus, different amounts of held work.
 * `string-similarity` is Dice over bigrams too but strips whitespace first, so
 * it scores multi-word text slightly differently — near enough to time against,
 * not near enough to check.
 *
 * **uFuzzy and Fuse are not a like-for-like at all.** uFuzzy matches a
 * subsequence — the needle's characters in order, with bounded gaps — and Fuse
 * is Bitap over a prebuilt index; both rank by where a match landed, where Dice
 * measures n-gram overlap and ignores position, so `'new york mets'` and
 * `'mets new york'` are near-identical to us and unrelated to them. Their match
 * counts are printed beside the times for that reason: what is comparable is
 * the time to narrow N candidates down to a handful, not the answer.
 *
 * Everything except the index and uFuzzy is capped at 100,000 choices, and the
 * cap is printed rather than passed over in silence.
 */

import process from 'node:process'

import { similarity as diceSimilarity } from '../../../dist/algorithms/dice/index.js'
import {
  createIndexedMatcher,
  createMatcher,
  createScorer,
  search,
} from '../../../dist/index.js'
import { time } from '../shared/timing.mjs'
import { buildCorpus } from './corpus.mjs'

let uFuzzy, Fuse, fuzzball, stringSimilarity, diceCoefficient, nGram
try {
  ;({ default: uFuzzy } = await import('@leeoniya/ufuzzy'))
  ;({ default: Fuse } = await import('fuse.js'))
  fuzzball = await import('fuzzball')
  stringSimilarity = await import('string-similarity')
  ;({ diceCoefficient } = await import('dice-coefficient'))
  nGram = await import('n-gram')
} catch {
  console.error(
    'The comparison libraries are not installed. They are contenders, not root\n' +
      'dependencies:\n\n  pnpm install --dir bench/comparison\n',
  )
  process.exit(1)
}

/** @type {(value: string) => string[]} */
const grams = (value) => nGram.nGram(GRAM_SIZE)(value)

// ---------------------------------------------------------------- parameters

/** How many results a caller asks for. Every arm that can be told, is told. */
const LIMIT = 5

/** Dice similarity a choice has to reach. Both our matchers prune on it. */
const THRESHOLD = 0.5

/** The scan arms are O(N) per query with a large constant; past this they stop. */
const SCAN_LIMIT = 100_000

let GRAM_SIZE = 2
const sizes = [100, 1_000, 10_000, 100_000]
for (const argument of process.argv.slice(2)) {
  if (argument.startsWith('--gram=')) GRAM_SIZE = Number(argument.slice('--gram='.length))
  if (argument.startsWith('--max=')) {
    const max = Number(argument.slice('--max='.length))
    if (max >= 1_000_000) sizes.push(1_000_000)
  }
}

// ---------------------------------------------------------------- the arms

/**
 * @typedef {object} Arm
 * @property {string} name
 * @property {number} setup        milliseconds, once per corpus
 * @property {(query: string) => unknown} run
 * @property {(query: string) => number} matches  results at or above the bar
 */

const millisecondsSince = (started) => Number(process.hrtime.bigint() - started) / 1e6

/**
 * Build every arm over one corpus, timing the setup each one needs.
 *
 * @param {string[]} choices
 * @returns {Arm[]}
 */
function buildArms(choices) {
  const scorer = createScorer(diceSimilarity, { gramSize: GRAM_SIZE })
  const options = { limit: LIMIT, threshold: THRESHOLD }

  const startedIndexed = process.hrtime.bigint()
  const indexed = createIndexedMatcher(choices, { scorer })
  const indexedSetup = millisecondsSince(startedIndexed)
  const searcher = new uFuzzy()
  // uFuzzy tolerates no error inside a term at its defaults, so the row above
  // is its speed at finding no typos at all. This is its own single-error
  // preset, and the difference between the two rows is what that costs.
  const typoTolerant = new uFuzzy({
    intraMode: 1,
    intraIns: 1,
    intraSub: 1,
    intraTrn: 1,
    intraDel: 1,
  })

  /** @type {Arm} */
  const indexedArm = {
    name: 'ours: createIndexedMatcher',
    setup: indexedSetup,
    run: (query) => indexed.search(query, options),
    matches: (query) =>
      indexed.search(query, { limit: null, threshold: THRESHOLD }).length,
  }
  /** @type {Arm} */
  const uFuzzyArm = {
    name: 'uFuzzy.filter',
    setup: 0,
    run: (query) => searcher.filter(choices, query),
    matches: (query) => searcher.filter(choices, query)?.length ?? 0,
  }
  // `filter` narrows; it does not rank. Every other arm here returns a ranked
  // best five, so the fair uFuzzy row is its whole pipeline — filter, info,
  // sort — with the error tolerance that makes a typo findable at all.
  /** @type {Arm} */
  const uFuzzyTypoArm = {
    name: 'uFuzzy.search, single error',
    setup: 0,
    run: (query) => typoTolerant.search(choices, query, 0),
    matches: (query) => typoTolerant.search(choices, query, 0)[0]?.length ?? 0,
  }
  if (choices.length > SCAN_LIMIT) return [indexedArm, uFuzzyArm, uFuzzyTypoArm]

  const startedMatcher = process.hrtime.bigint()
  const matcher = createMatcher(choices, { scorer })
  const matcherSetup = millisecondsSince(startedMatcher)

  const startedGrams = process.hrtime.bigint()
  const prebuilt = choices.map((choice) => grams(choice))
  const gramsSetup = millisecondsSince(startedGrams)

  const startedFuse = process.hrtime.bigint()
  const fuse = new Fuse(choices, { includeScore: true, threshold: 0.4 })
  const fuseSetup = millisecondsSince(startedFuse)

  /** The shape a package with no search API is used in: score all, keep the best. */
  const scanBest = (query) => {
    const left = grams(query)
    let best = -1
    for (const right of prebuilt) {
      const value = diceCoefficient(left, right)
      if (value > best) best = value
    }
    return best
  }

  return [
    indexedArm,
    {
      name: 'ours: createMatcher',
      setup: matcherSetup,
      run: (query) => matcher.search(query, options),
      matches: (query) =>
        matcher.search(query, { limit: null, threshold: THRESHOLD }).length,
    },
    {
      name: 'ours: search (one-shot)',
      setup: 0,
      run: (query) => search(query, choices, { scorer, ...options }),
      matches: (query) =>
        search(query, choices, { scorer, limit: null, threshold: THRESHOLD }).length,
    },
    {
      name: 'dice-coefficient, prebuilt grams',
      setup: gramsSetup,
      run: scanBest,
      matches: (query) => {
        const left = grams(query)
        let qualifying = 0
        for (const right of prebuilt)
          if (diceCoefficient(left, right) >= THRESHOLD) qualifying++
        return qualifying
      },
    },
    {
      name: 'string-similarity.findBestMatch',
      setup: 0,
      run: (query) => stringSimilarity.findBestMatch(query, choices),
      matches: (query) =>
        stringSimilarity
          .findBestMatch(query, choices)
          .ratings.filter((rating) => rating.rating >= THRESHOLD).length,
    },
    {
      name: 'fuzzball.extract',
      setup: 0,
      run: (query) =>
        fuzzball.extract(query, choices, {
          scorer: fuzzball.ratio,
          limit: LIMIT,
          cutoff: THRESHOLD * 100,
          full_process: false,
        }),
      matches: (query) =>
        fuzzball.extract(query, choices, {
          scorer: fuzzball.ratio,
          cutoff: THRESHOLD * 100,
          full_process: false,
        }).length,
    },
    uFuzzyArm,
    uFuzzyTypoArm,
    {
      name: 'Fuse, prebuilt index',
      setup: fuseSetup,
      run: (query) => fuse.search(query, { limit: LIMIT }),
      matches: (query) => fuse.search(query).length,
    },
  ]
}

// ---------------------------------------------------------------- agreement

/**
 * The three of ours and `dice-coefficient` compute one number, so a
 * disagreement is a bug rather than a metric difference. Checked on every
 * corpus, before it is timed.
 *
 * @param {string[]} choices
 * @param {[string, string][]} queries
 * @returns {string[]}
 */
function disagreements(choices, queries) {
  const scorer = createScorer(diceSimilarity, { gramSize: GRAM_SIZE })
  const indexed = createIndexedMatcher(choices, { scorer })
  const matcher = createMatcher(choices, { scorer })
  const problems = []
  for (const [name, query] of queries) {
    const options = { limit: LIMIT, threshold: THRESHOLD }
    const fromIndex = indexed.search(query, options)
    const fromMatcher = matcher.search(query, options)
    const shape = (matches) =>
      matches.map((match) => `${match.key}:${match.score.toFixed(12)}`)
    if (shape(fromIndex).join() !== shape(fromMatcher).join()) {
      problems.push(`${name}: the index and the Matcher disagree`)
    }
    const left = grams(query)
    for (const match of fromIndex) {
      const theirs = diceCoefficient(left, grams(choices[Number(match.key)]))
      if (Math.abs(theirs - match.score) > 1e-12) {
        problems.push(`${name}: dice-coefficient says ${theirs}, we say ${match.score}`)
      }
    }
  }
  return problems
}

// ---------------------------------------------------------------- the report

const formatMs = (value) =>
  value >= 100
    ? value.toFixed(0)
    : value >= 10
      ? value.toFixed(1)
      : value >= 1
        ? value.toFixed(2)
        : value >= 0.01
          ? value.toFixed(3)
          : value.toFixed(4)

const NAME_WIDTH = 34
const CELL_WIDTH = 15

console.log(
  `\n  Dice at gramSize ${GRAM_SIZE}, limit ${LIMIT}, threshold ${THRESHOLD}. ` +
    `Milliseconds for one query,\n  median of nine passes. Lower is better.`,
)

for (const count of sizes) {
  const corpus = buildCorpus(count)
  const problems =
    count <= SCAN_LIMIT ? disagreements(corpus.choices, corpus.queries) : []
  const arms = buildArms(corpus.choices)
  const header = corpus.queries.map(([name]) => name.padStart(CELL_WIDTH)).join('')

  console.log(`\n  ${count.toLocaleString()} choices`)
  if (problems.length > 0) for (const problem of problems) console.log(`  ! ${problem}`)
  console.log(`  ${'time per query'.padEnd(NAME_WIDTH)}${header}${'setup'.padStart(10)}`)
  for (const arm of arms) {
    const cells = corpus.queries
      .map(([, query]) => formatMs(time(arm.run.bind(null, query)).median * 1e3))
      .map((cell) => cell.padStart(CELL_WIDTH))
      .join('')
    const setup = arm.setup === 0 ? '—' : formatMs(arm.setup)
    console.log(`  ${arm.name.padEnd(NAME_WIDTH)}${cells}${setup.padStart(10)}`)
  }

  console.log(`\n  ${'results returned'.padEnd(NAME_WIDTH)}${header}`)
  for (const arm of arms) {
    const cells = corpus.queries
      .map(([, query]) => String(arm.matches(query)).padStart(CELL_WIDTH))
      .join('')
    console.log(`  ${arm.name.padEnd(NAME_WIDTH)}${cells}`)
  }
  if (count > SCAN_LIMIT) {
    console.log(
      `\n  Every arm but the index was capped at ${SCAN_LIMIT.toLocaleString()} choices and is` +
        ` not measured here.`,
    )
  }
}

console.log(
  '\n  Setup is milliseconds once per corpus: an index, a prepared collection, a\n' +
    '  gram array per choice, a Bitap index. The four Dice arms agree to 1e-12 and\n' +
    '  are checked above; uFuzzy, Fuse and fuzzball answer a different question, so\n' +
    '  their result counts are printed rather than compared.\n',
)
