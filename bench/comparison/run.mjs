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
 * Two things separate this from `bench/tooling/compare.ts`, and they are the reason it
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

import { distance as damerauLevenshteinDistance } from '../../dist/algorithms/damerauLevenshtein/index.js'
import { distance as hammingDistance } from '../../dist/algorithms/hamming/index.js'
import { distance as indelDistance } from '../../dist/algorithms/indel/index.js'
import { similarity as jaroSimilarity } from '../../dist/algorithms/jaro/index.js'
import { similarity as jaroWinklerSimilarity } from '../../dist/algorithms/jaroWinkler/index.js'
import {
  editops as lcsEditops,
  similarity as lcsSimilarity,
} from '../../dist/algorithms/lcs/index.js'
import {
  distance as levenshteinDistance,
  editops as levenshteinEditops,
} from '../../dist/algorithms/levenshtein/index.js'
import { distance as osaDistance } from '../../dist/algorithms/osa/index.js'
import { distance as postfixDistance } from '../../dist/algorithms/postfix/index.js'
import { distance as prefixDistance } from '../../dist/algorithms/prefix/index.js'
import {
  weightedSimilarity,
  partialSimilarity,
  similarity as fuzzSimilarity,
  tokenSetSimilarity,
  tokenSortSimilarity,
} from '../../dist/fuzz/index.js'
import {
  bestMatch,
  createMatcher,
  createScorer,
  scoreMatrix,
  scorePairs,
} from '../../dist/index.js'

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
const fixedLevenshteinQuery = corpus.pairs['128'][0][0]
const fixedLevenshteinChoices = corpus.pairs['128'].map(([, choice]) => choice)

// 0.6 has no public prepared handle. A query is held for the length of one
// call — `scoreMatrix` prepares each row once and each column once — and a
// collection is held by a `Matcher`, which prepares every choice at
// construction and reuses them for every later query.
const levenshteinScorer = createScorer(levenshteinDistance)
const fuzzScorer = createScorer(fuzzSimilarity)
const tokenSortScorer = createScorer(tokenSortSimilarity)
const titleMatcher = createMatcher(corpus.titles, { scorer: tokenSortScorer })

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
    const rounded = Math.round(fuzzSimilarity(first, second))
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

const runFixedLevenshteinDirect = () => {
  for (const choice of fixedLevenshteinChoices) {
    levenshteinDistance(fixedLevenshteinQuery, choice)
  }
}
const runFixedLevenshteinHeld = () => {
  scoreMatrix([fixedLevenshteinQuery], fixedLevenshteinChoices, {
    scorer: levenshteinScorer,
  })
}

console.log('\nHeld-query Levenshtein — 1 query, 200 choices at 128 chars')
contest(
  'held query',
  'rapidfuzz-js direct',
  runFixedLevenshteinHeld,
  runFixedLevenshteinDirect,
  'same scorer and numbers; the query is prepared once per call, not per pair',
)
contest(
  'held query',
  'fastest-levenshtein',
  runFixedLevenshteinHeld,
  () => {
    for (const choice of fixedLevenshteinChoices) {
      fastestLevenshtein(fixedLevenshteinQuery, choice)
    }
  },
  'same distance; contender scores every pair from scratch',
)
contest(
  'held query',
  'leven',
  runFixedLevenshteinHeld,
  () => {
    for (const choice of fixedLevenshteinChoices) leven(fixedLevenshteinQuery, choice)
  },
  'same distance; contender scores every pair from scratch',
)
contest(
  'held query',
  'js-levenshtein',
  runFixedLevenshteinHeld,
  () => {
    for (const choice of fixedLevenshteinChoices) {
      jsLevenshtein(fixedLevenshteinQuery, choice)
    }
  },
  'same distance; contender scores every pair from scratch',
)
contest(
  'held query',
  'fuzzball',
  runFixedLevenshteinHeld,
  () => {
    for (const choice of fixedLevenshteinChoices) {
      fuzzball.distance(fixedLevenshteinQuery, choice, RAW)
    }
  },
  'same distance; contender scores every pair from scratch',
)

console.log('\nNormalised similarity')
contest(
  `sentences, ${corpus.sentences.length} pairs`,
  'fuzzball',
  () => {
    for (const [a, b] of corpus.sentences) fuzzSimilarity(a, b)
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
    for (const [a, b] of corpus.sentences) fuzzSimilarity(a, b)
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
      bestMatch(query, corpus.choices, { scorer: fuzzScorer })
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
      bestMatch(query, corpus.choices, { scorer: fuzzScorer })
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
      bestMatch(query, corpus.choices, { scorer: fuzzScorer })
    }
  },
  () => {
    for (const query of corpus.queries) fuse.search(query, { limit: 1 })
  },
  'ours scans every choice, Fuse uses a prebuilt Bitap index',
)

const runRawTokenSearch = () => {
  for (const query of corpus.titleQueries) {
    bestMatch(query, corpus.titles, { scorer: tokenSortScorer })
  }
}
const runMatcherTokenSearch = () => {
  for (const query of corpus.titleQueries) titleMatcher.best(query)
}
const runFuzzballTokenSearch = () => {
  for (const query of corpus.titleQueries) {
    fuzzball.extract(query, corpus.titles, {
      scorer: fuzzball.token_sort_ratio,
      limit: 1,
      cutoff: 0,
      full_process: false,
    })
  }
}

console.log('\nMatcher token-sort search — 20 queries, 2,000 multiword titles')
contest(
  'raw choices',
  'fuzzball',
  runRawTokenSearch,
  runFuzzballTokenSearch,
  'same token-sort scorer; fuzzball rounds scores',
)
contest(
  'Matcher',
  'rapidfuzz-js one-shot',
  runMatcherTokenSearch,
  runRawTokenSearch,
  'same scorer and results; holding the collection is the only difference',
)
contest(
  'Matcher',
  'fuzzball',
  runMatcherTokenSearch,
  runFuzzballTokenSearch,
  'same token-sort scorer; fuzzball re-reads the collection every query',
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
  const pairs128 = corpus.pairs['128']
  const sentenceLeft = corpus.sentences.map(([left]) => left)
  const sentenceRight = corpus.sentences.map(([, right]) => right)

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
    'fuzz.similarity, sentences',
    () => {
      for (const [a, b] of corpus.sentences) fuzzSimilarity(a, b)
    },
  ])
  againstPython.push([
    'fixed-query-levenshtein-128',
    'fixed-query Levenshtein, direct',
    runFixedLevenshteinDirect,
  ])
  againstPython.push([
    'fixed-query-levenshtein-128',
    'fixed-query Levenshtein, held',
    runFixedLevenshteinHeld,
  ])
  againstPython.push([
    'indel-distance-128',
    'Indel.distance, 128 chars',
    () => {
      for (const [a, b] of pairs128) indelDistance(a, b)
    },
  ])
  againstPython.push([
    'lcs-similarity-128',
    'lcs.similarity, 128 chars',
    () => {
      for (const [a, b] of pairs128) lcsSimilarity(a, b)
    },
  ])
  againstPython.push([
    'osa-distance-128',
    'OSA.distance, 128 chars',
    () => {
      for (const [a, b] of pairs128) osaDistance(a, b)
    },
  ])
  againstPython.push([
    'damerau-distance-128',
    'DamerauLevenshtein.distance, 128 chars',
    () => {
      for (const [a, b] of pairs128) damerauLevenshteinDistance(a, b)
    },
  ])
  againstPython.push([
    'hamming-distance-128',
    'Hamming.distance, 128 chars',
    () => {
      for (const [a, b] of pairs128) hammingDistance(a, b)
    },
  ])
  againstPython.push([
    'jaro-similarity-128',
    'Jaro.similarity, 128 chars',
    () => {
      for (const [a, b] of pairs128) jaroSimilarity(a, b)
    },
  ])
  againstPython.push([
    'jaro-winkler-similarity-128',
    'JaroWinkler.similarity, 128 chars',
    () => {
      for (const [a, b] of pairs128) jaroWinklerSimilarity(a, b)
    },
  ])
  againstPython.push([
    'prefix-distance-128',
    'Prefix.distance, 128 chars',
    () => {
      for (const [a, b] of pairs128) prefixDistance(a, b)
    },
  ])
  againstPython.push([
    'postfix-distance-128',
    'Postfix.distance, 128 chars',
    () => {
      for (const [a, b] of pairs128) postfixDistance(a, b)
    },
  ])
  againstPython.push([
    'partial-ratio-sentences',
    'fuzz.partialSimilarity, sentences',
    () => {
      for (const [a, b] of corpus.sentences) partialSimilarity(a, b)
    },
  ])
  againstPython.push([
    'token-sort-ratio-sentences',
    'fuzz.tokenSortSimilarity, sentences',
    () => {
      for (const [a, b] of corpus.sentences) tokenSortSimilarity(a, b)
    },
  ])
  againstPython.push([
    'token-set-ratio-sentences',
    'fuzz.tokenSetSimilarity, sentences',
    () => {
      for (const [a, b] of corpus.sentences) tokenSetSimilarity(a, b)
    },
  ])
  againstPython.push([
    'w-ratio-sentences',
    'fuzz.weightedSimilarity, sentences',
    () => {
      for (const [a, b] of corpus.sentences) weightedSimilarity(a, b)
    },
  ])
  againstPython.push([
    'extract-one',
    'bestMatch, 2,000 choices',
    () => {
      for (const query of corpus.queries) {
        bestMatch(query, corpus.choices, { scorer: fuzzScorer })
      }
    },
  ])
  againstPython.push([
    'extract-one-token-sort',
    'bestMatch tokenSort, 2,000 titles',
    runRawTokenSearch,
  ])
  againstPython.push([
    'extract-one-token-sort',
    'Matcher tokenSort, 2,000 titles',
    runMatcherTokenSearch,
  ])
  againstPython.push([
    'score-matrix',
    'scoreMatrix, 50 x 200',
    () => {
      scoreMatrix(corpus.matrixRows, corpus.matrixCols, { scorer: fuzzScorer })
    },
  ])
  againstPython.push([
    'score-matrix-token-sort',
    'scoreMatrix tokenSort, 50 x 200',
    () => {
      scoreMatrix(corpus.titles.slice(0, 50), corpus.titles.slice(50, 250), {
        scorer: tokenSortScorer,
      })
    },
  ])
  againstPython.push([
    'score-pairs-ratio',
    'scorePairs, 200 pairs',
    () => {
      scorePairs(sentenceLeft, sentenceRight, { scorer: fuzzScorer })
    },
  ])
  againstPython.push([
    'levenshtein-editops-128',
    'Levenshtein.editops, 128 chars',
    () => {
      for (const [a, b] of pairs128) levenshteinEditops(a, b)
    },
  ])
  againstPython.push([
    'lcs-editops-128',
    'lcs.editops, 128 chars',
    () => {
      for (const [a, b] of pairs128) lcsEditops(a, b)
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
