// @ts-check
/**
 * Does it find the thing? — the half of a search comparison a timing table
 * cannot show.
 *
 * ```sh
 * pnpm build && node bench/comparison/indexedSearch/quality.mjs
 * ```
 *
 * `throughput.mjs` measures how fast each library narrows a list. Fast is only
 * half an answer: a search that returns in a microsecond without the item the
 * user typed has not done the job. This script asks the other half. It takes
 * known entries out of the corpus, damages them the way a person does — a
 * substituted letter, a dropped one, two swapped, words in the wrong order, a
 * half-remembered phrase — and asks each library for its best five. The entry
 * it came from either comes back or it does not.
 *
 * **Every library is configured to be as fuzzy as it can be**, which matters
 * more than it sounds. uFuzzy tolerates no error inside a word by default —
 * `intraMode: 0` — so a single typo makes it return nothing, and Fuse's default
 * `location`/`distance` pair only matches near the start of a string. Measuring
 * either at its defaults and calling it a fuzzy search would be a straw man, so
 * both appear twice: as they arrive, and configured for the job.
 *
 * The metrics are the two a caller cares about. **Hit@1** is how often the
 * intended entry was ranked first, and **hit@5** how often it was anywhere in
 * the five results a UI would show. Text is compared rather than index, because
 * a Zipf corpus repeats phrases and an equal string is an equally right answer.
 */

import process from 'node:process'

import { similarity as diceSimilarity } from '../../../dist/algorithms/dice/index.js'
import { createIndexedMatcher, createMatcher, createScorer } from '../../../dist/index.js'
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

const CHOICES = 10_000
const GRAM_SIZE = 2
const LIMIT = 5
const TARGETS = 40

const grams = nGram.nGram(GRAM_SIZE)

// ---------------------------------------------------------------- the queries

/**
 * The ways a remembered string differs from the stored one. Each takes an entry
 * and returns what someone might type instead.
 *
 * Every character edit lands inside a word, never on a space and never on a
 * word's first letter. Both exclusions are fairness rather than realism: a typo
 * on a space merges two words, and uFuzzy's single-error mode declines an error
 * at a term's start (`intraSlice`), so either one would measure a corner of its
 * configuration instead of its matcher.
 *
 * @type {[string, (value: string) => string][]}
 */
const DAMAGE = [
  ['exact', (value) => value],
  [
    'substitution',
    (value) =>
      editWord(
        value,
        (word, at) =>
          word.slice(0, at) + (word[at] === 'a' ? 'e' : 'a') + word.slice(at + 1),
      ),
  ],
  [
    'deletion',
    (value) => editWord(value, (word, at) => word.slice(0, at) + word.slice(at + 1)),
  ],
  [
    'transposition',
    (value) =>
      editWord(
        value,
        (word, at) => word.slice(0, at) + word[at + 1] + word[at] + word.slice(at + 2),
      ),
  ],
  ['words reordered', (value) => reorder(value)],
  ['half the phrase', (value) => value.split(' ').slice(0, 2).join(' ')],
]

/**
 * Apply `edit` inside the longest word of `value`, at a position that is
 * neither its first nor its last character.
 */
function editWord(value, edit) {
  const words = value.split(' ')
  let longest = 0
  for (let index = 1; index < words.length; index++) {
    if (words[index].length > words[longest].length) longest = index
  }
  const word = words[longest]
  if (word.length < 4) return value
  words[longest] = edit(word, Math.floor(word.length / 2))
  return words.join(' ')
}

function reorder(value) {
  const words = value.split(' ')
  return [words[words.length - 1], ...words.slice(0, -1)].join(' ')
}

// ---------------------------------------------------------------- the arms

const corpus = buildCorpus(CHOICES)
const choices = corpus.choices
const scorer = createScorer(diceSimilarity, { gramSize: GRAM_SIZE })

const indexed = createIndexedMatcher(choices, { scorer })
const matcher = createMatcher(choices, { scorer })
const prebuilt = choices.map((choice) => grams(choice))

const defaultUFuzzy = new uFuzzy()
// uFuzzy's own "single error" preset: one insertion, substitution, transposition
// or deletion inside a term. Without it a typo is invisible to it by design.
const fuzzyUFuzzy = new uFuzzy({
  intraMode: 1,
  intraIns: 1,
  intraSub: 1,
  intraTrn: 1,
  intraDel: 1,
})
const defaultFuse = new Fuse(choices, { includeScore: true, threshold: 0.4 })
const fuzzyFuse = new Fuse(choices, {
  includeScore: true,
  threshold: 0.6,
  ignoreLocation: true,
})

/** Rank by score, take the best `LIMIT`, return the texts. */
function bestScored(score) {
  const ranked = []
  for (let index = 0; index < choices.length; index++) {
    ranked.push([score(index), choices[index]])
  }
  ranked.sort((left, right) => right[0] - left[0])
  return ranked.slice(0, LIMIT).map((entry) => entry[1])
}

/** uFuzzy's ranked form: filter, then info, then its own sort. */
function rankUFuzzy(searcher, query, outOfOrder) {
  const found = searcher.search(choices, query, outOfOrder)
  const [, info, order] = found
  if (info === null || order === null) return []
  return order.slice(0, LIMIT).map((position) => choices[info.idx[position]])
}

/** @type {[string, (query: string) => string[]][]} */
const ARMS = [
  [
    'ours: createIndexedMatcher',
    (query) => indexed.search(query, { limit: LIMIT }).map((match) => match.item),
  ],
  [
    'ours: createMatcher',
    (query) => matcher.search(query, { limit: LIMIT }).map((match) => match.item),
  ],
  [
    'dice-coefficient scan',
    (query) => {
      const left = grams(query)
      return bestScored((index) => diceCoefficient(left, prebuilt[index]))
    },
  ],
  [
    'string-similarity',
    (query) =>
      [...stringSimilarity.findBestMatch(query, choices).ratings]
        .map((rating, index) => [rating.rating, index])
        .sort((left, right) => right[0] - left[0])
        .slice(0, LIMIT)
        .map(([, index]) => choices[index]),
  ],
  [
    'fuzzball.extract',
    (query) =>
      fuzzball
        .extract(query, choices, {
          scorer: fuzzball.ratio,
          limit: LIMIT,
          full_process: false,
        })
        .map((entry) => entry[0]),
  ],
  ['uFuzzy, defaults', (query) => rankUFuzzy(defaultUFuzzy, query, 0)],
  ['uFuzzy, single error + out of order', (query) => rankUFuzzy(fuzzyUFuzzy, query, 5)],
  [
    'Fuse, defaults',
    (query) => defaultFuse.search(query, { limit: LIMIT }).map((entry) => entry.item),
  ],
  [
    'Fuse, ignoreLocation',
    (query) => fuzzyFuse.search(query, { limit: LIMIT }).map((entry) => entry.item),
  ],
]

// ---------------------------------------------------------------- the contest

const targets = []
for (let index = 0; index < TARGETS; index++) {
  targets.push(choices[Math.floor(((index + 0.5) / TARGETS) * choices.length)])
}

const NAME_WIDTH = 36
const CELL_WIDTH = 18

console.log(
  `\n  ${CHOICES.toLocaleString()} choices, ${TARGETS} entries taken out of them and damaged.\n` +
    `  Each cell is hit@1 / hit@5: how often the entry the query came from was\n` +
    `  ranked first, and how often it was in the best ${LIMIT}. Higher is better.`,
)
console.log(
  `\n  ${'library'.padEnd(NAME_WIDTH)}${DAMAGE.map(([name]) => name.padStart(CELL_WIDTH)).join('')}` +
    `${'ms/query'.padStart(11)}`,
)

for (const [name, run] of ARMS) {
  const cells = []
  let elapsed = 0
  for (const [, damage] of DAMAGE) {
    let first = 0
    let within = 0
    for (const target of targets) {
      const query = damage(target)
      const started = process.hrtime.bigint()
      const results = run(query)
      elapsed += Number(process.hrtime.bigint() - started) / 1e6
      if (results[0] === target) first++
      if (results.includes(target)) within++
    }
    cells.push(`${first}/${within}`.padStart(CELL_WIDTH))
  }
  const perQuery = elapsed / (TARGETS * DAMAGE.length)
  console.log(
    `  ${name.padEnd(NAME_WIDTH)}${cells.join('')}${perQuery.toFixed(3).padStart(11)}`,
  )
}

console.log(
  `\n  Out of ${TARGETS} per cell. Times are one warm pass over every query, not a\n` +
    '  median — read them as an order of magnitude and take the real figures from\n' +
    '  throughput.mjs. uFuzzy and Fuse appear twice because their defaults refuse\n' +
    '  intra-word errors and off-start matches respectively.\n',
)
