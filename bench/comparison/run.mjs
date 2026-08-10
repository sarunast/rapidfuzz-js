// @ts-check
/**
 * How this library compares against other implementations of the same job —
 * the other JavaScript fuzzy-matching packages, and the Python RapidFuzz this
 * is a port of.
 *
 * ```sh
 * pnpm build && node bench/comparison/run.mjs
 * node bench/comparison/run.mjs --python=/path/to/venv/bin/python
 * ```
 *
 * Two things separate this from `bench/compare.mjs`, and they are the reason it
 * is a second script rather than more cases in the first:
 *
 *   - **It measures `dist/`, not `src/`.** Every contender is installed from
 *     npm and runs as its published build, so this one has to as well.
 *     Comparing our TypeScript sources against their bundled output would
 *     charge us for module hops the published package never pays.
 *   - **Every contender runs in one process.** Timing two libraries in two
 *     processes measured a 40% difference between two runs of the *same* code
 *     on this machine. Within one process they share a JIT, a heap and a
 *     thermal state, and the ratio is the thing being reported anyway.
 *
 * The Python leg is opt-in because it needs an interpreter this repository does
 * not vendor:
 *
 * ```sh
 * python3 -m venv .venv && .venv/bin/pip install rapidfuzz
 * node bench/comparison/run.mjs --python=.venv/bin/python
 * ```
 */
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { extractOne, levenshteinDistance, ratio, scoreMatrix } from '../../dist/index.js'

// The contenders are dependencies of `bench/comparison/package.json`, not of
// the library — nothing in `src/`, `tests/` or the baseline benchmarks needs
// them, and a contributor running `pnpm check` should not be made to install
// six fuzzy-matching packages first. Node resolves them out of
// `bench/comparison/node_modules`, and this import is guarded so a missing
// install is a sentence rather than a stack trace.
let fastestLevenshtein, Fuse, fuzzball, jsLevenshtein, leven, stringSimilarity
try {
  ;({ distance: fastestLevenshtein } = await import('fastest-levenshtein'))
  ;({ default: Fuse } = await import('fuse.js'))
  fuzzball = await import('fuzzball')
  ;({ default: jsLevenshtein } = await import('js-levenshtein'))
  ;({ default: leven } = await import('leven'))
  stringSimilarity = await import('string-similarity')
} catch {
  console.error(
    'The comparison libraries are not installed. They are kept out of the root\n' +
      'package.json on purpose. Install them where they belong:\n\n' +
      '  pnpm install --dir bench/comparison\n',
  )
  process.exit(1)
}
const { compareTwoStrings, findBestMatch } = stringSimilarity

/**
 * fuzzball preprocesses both inputs unless told not to — `full_process` lower
 * cases and strips non-alphanumerics, which is `defaultProcess` here and is off
 * by default on this side. Every call below turns it off, so the two libraries
 * are handed the same strings.
 */
const RAW = { full_process: false }
import { buildCorpus, PAIR_LENGTHS } from './corpus.mjs'
import { speedup, time } from './timing.mjs'

const here = fileURLToPath(new URL('.', import.meta.url))

/** @type {string | null} */
let pythonPath = process.env['RAPIDFUZZ_PYTHON'] ?? null
for (const argument of process.argv.slice(2)) {
  if (argument.startsWith('--python=')) pythonPath = argument.slice(9)
}

const corpus = buildCorpus()

/**
 * Before timing anything, check the contenders answer the same question.
 *
 * A distance benchmark against something that computes a different number is
 * not a benchmark, it is a table. The four Levenshtein implementations below
 * are checked against ours on every pair in the corpus; the Dice and Bitap
 * contenders further down are *not* checked, because they genuinely score
 * something else, and the report says so rather than pretending otherwise.
 *
 * `fuzzball` is checked twice over, because it is the one contender descended
 * from the same source as this library — a port of fuzzywuzzy, where RapidFuzz
 * began. Its distances are ours exactly; its `ratio` family is ours rounded to
 * the nearest integer, which is fuzzywuzzy's own convention and the only
 * difference between the two libraries' numbers.
 */
function checkAgreement() {
  /** @type {string[]} */
  const problems = []
  for (const length of PAIR_LENGTHS) {
    for (const [first, second] of corpus.pairs[String(length)]) {
      const ours = levenshteinDistance(first, second)
      /** @type {[string, number][]} */
      const others = [
        ['fastest-levenshtein', fastestLevenshtein(first, second)],
        ['leven', leven(first, second)],
        ['js-levenshtein', jsLevenshtein(first, second)],
        ['fuzzball', fuzzball.distance(first, second, RAW)],
      ]
      for (const [name, theirs] of others) {
        if (theirs !== ours) {
          problems.push(`${name}: ${theirs} vs ${ours} on ${length}-char pair`)
        }
      }
    }
  }

  for (const [first, second] of corpus.sentences) {
    const rounded = Math.round(ratio(first, second))
    const theirs = fuzzball.ratio(first, second, RAW)
    if (theirs !== rounded) {
      problems.push(`fuzzball ratio: ${theirs} vs ${rounded} on "${first}"`)
    }
  }

  return problems
}

/**
 * @typedef {object} Row
 * @property {string} task
 * @property {string} contender
 * @property {number} ours     seconds per pass
 * @property {number} theirs   seconds per pass
 * @property {string} note
 */

/** @type {Row[]} */
const rows = []

/**
 * @param {string} task
 * @param {string} contender
 * @param {() => void} ourRun
 * @param {() => void} theirRun
 * @param {string} [note]
 */
function contest(task, contender, ourRun, theirRun, note = '') {
  const ours = time(ourRun)
  const theirs = time(theirRun)
  rows.push({ task, contender, ours: ours.median, theirs: theirs.median, note })
  const spread = Math.max(ours.spread, theirs.spread)
  const flag = spread > 0.25 ? `  (spread ${(spread * 100).toFixed(0)}%)` : ''
  console.log(
    `  ${task.padEnd(34)} vs ${contender.padEnd(22)} ${speedup(ours.median, theirs.median).padStart(14)}${flag}`,
  )
}

console.log('\nChecking the exact contenders agree with us…')
const disagreements = checkAgreement()
if (disagreements.length > 0) {
  console.log(`  ${disagreements.length} disagreement(s):`)
  for (const problem of disagreements.slice(0, 5)) console.log(`    ${problem}`)
  throw new Error('contenders disagree; a timing comparison would be meaningless')
}
console.log('  all agree on every pair in the corpus\n')

console.log('Levenshtein distance — same number, checked above')
for (const length of PAIR_LENGTHS) {
  const pairs = corpus.pairs[String(length)]
  const label = `${length} chars, ${pairs.length} pairs`
  contest(
    label,
    'fastest-levenshtein',
    () => {
      for (const [a, b] of pairs) levenshteinDistance(a, b)
    },
    () => {
      for (const [a, b] of pairs) fastestLevenshtein(a, b)
    },
  )
  contest(
    label,
    'leven',
    () => {
      for (const [a, b] of pairs) levenshteinDistance(a, b)
    },
    () => {
      for (const [a, b] of pairs) leven(a, b)
    },
  )
  contest(
    label,
    'js-levenshtein',
    () => {
      for (const [a, b] of pairs) levenshteinDistance(a, b)
    },
    () => {
      for (const [a, b] of pairs) jsLevenshtein(a, b)
    },
  )
  contest(
    label,
    'fuzzball',
    () => {
      for (const [a, b] of pairs) levenshteinDistance(a, b)
    },
    () => {
      for (const [a, b] of pairs) fuzzball.distance(a, b, RAW)
    },
  )
}

console.log('\nNormalised similarity')
contest(
  `sentences, ${corpus.sentences.length} pairs`,
  'fuzzball',
  () => {
    for (const [a, b] of corpus.sentences) ratio(a, b)
  },
  () => {
    for (const [a, b] of corpus.sentences) fuzzball.ratio(a, b, RAW)
  },
  'same number, rounded — checked above',
)
contest(
  `sentences, ${corpus.sentences.length} pairs`,
  'string-similarity',
  () => {
    for (const [a, b] of corpus.sentences) ratio(a, b)
  },
  () => {
    for (const [a, b] of corpus.sentences) compareTwoStrings(a, b)
  },
  'ours is Indel-normalised, theirs is Dice over bigrams',
)

console.log('\nBest match among 2,000 choices')
contest(
  `${corpus.queries.length} queries`,
  'fuzzball',
  () => {
    for (const query of corpus.queries) {
      extractOne(query, corpus.choices, { scorer: ratio })
    }
  },
  () => {
    for (const query of corpus.queries) {
      fuzzball.extract(query, corpus.choices, {
        scorer: fuzzball.ratio,
        limit: 1,
        cutoff: 0,
        full_process: false,
      })
    }
  },
  'same scorer on both sides',
)
contest(
  `${corpus.queries.length} queries`,
  'string-similarity',
  () => {
    for (const query of corpus.queries) {
      extractOne(query, corpus.choices, { scorer: ratio })
    }
  },
  () => {
    for (const query of corpus.queries) findBestMatch(query, corpus.choices)
  },
  'ours scores Indel, theirs Dice',
)

// Fuse indexes its haystack, and that cost is paid once for many searches, so
// it is built outside the timed loop. Timing the build inside would be a
// comparison of setup, not of search.
const fuse = new Fuse(corpus.choices, { includeScore: true, threshold: 1 })
contest(
  `${corpus.queries.length} queries`,
  'fuse.js (prebuilt index)',
  () => {
    for (const query of corpus.queries) {
      extractOne(query, corpus.choices, { scorer: ratio })
    }
  },
  () => {
    for (const query of corpus.queries) fuse.search(query, { limit: 1 })
  },
  'ours scans every choice, Fuse uses a prebuilt Bitap index',
)

/**
 * The Python leg. Same corpus, same loop shape, same statistic — the script on
 * the other side restates `timing.mjs` rather than importing anything.
 *
 * @param {string} interpreter
 */
function runPython(interpreter) {
  const directory = mkdtempSync(join(tmpdir(), 'rapidfuzz-comparison-'))
  const corpusPath = join(directory, 'corpus.json')
  writeFileSync(corpusPath, JSON.stringify(corpus))

  const output = execFileSync(
    interpreter,
    [join(here, 'rapidfuzz_bench.py'), corpusPath],
    { encoding: 'utf8' },
  )
  /** @type {unknown} */
  const parsed = JSON.parse(output)
  if (typeof parsed !== 'object' || parsed === null) {
    throw new TypeError('the Python leg did not return an object')
  }
  /** @type {Record<string, number>} */
  const theirs = {}
  for (const [key, value] of Object.entries(parsed)) {
    if (typeof value !== 'number') throw new TypeError(`${key} is not a number`)
    theirs[key] = value
  }
  return theirs
}

if (pythonPath !== null) {
  console.log('\nAgainst Python RapidFuzz — same corpus, same loop shape')
  const python = runPython(pythonPath)

  /** @type {[string, string, () => void][]} */
  const againstPython = []
  for (const length of PAIR_LENGTHS) {
    const pairs = corpus.pairs[String(length)]
    againstPython.push([
      `levenshtein-${length}`,
      `Levenshtein.distance, ${length} chars`,
      () => {
        for (const [a, b] of pairs) levenshteinDistance(a, b)
      },
    ])
  }
  againstPython.push([
    'ratio-sentences',
    'fuzz.ratio, sentences',
    () => {
      for (const [a, b] of corpus.sentences) ratio(a, b)
    },
  ])
  againstPython.push([
    'extract-one',
    'process.extractOne, 2,000 choices',
    () => {
      for (const query of corpus.queries) {
        extractOne(query, corpus.choices, { scorer: ratio })
      }
    },
  ])
  againstPython.push([
    'score-matrix',
    'process.cdist, 50 x 200',
    () => {
      scoreMatrix(corpus.matrixRows, corpus.matrixCols, { scorer: ratio })
    },
  ])

  for (const [key, label, run] of againstPython) {
    const ours = time(run)
    const theirs = python[key]
    if (theirs === undefined) throw new Error(`the Python leg skipped ${key}`)
    rows.push({
      task: label,
      contender: 'rapidfuzz (Python)',
      ours: ours.median,
      theirs,
      note: '',
    })
    console.log(`  ${label.padEnd(38)} ${speedup(ours.median, theirs).padStart(14)}`)
  }
} else {
  console.log(
    '\nSkipping the Python leg. Pass --python=/path/to/python with rapidfuzz installed.',
  )
}

const reportPath = join(here, 'last-run.json')
writeFileSync(reportPath, `${JSON.stringify({ node: process.version, rows }, null, 2)}\n`)
console.log(`\nWrote ${reportPath}`)

// Referenced so the import is not mistaken for unused when the Python leg is
// skipped; reading it back also proves the file is valid JSON before anyone
// quotes a number out of it.
JSON.parse(readFileSync(reportPath, 'utf8'))
