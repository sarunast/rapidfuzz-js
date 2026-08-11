// @ts-check
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const pkg = JSON.parse(
  await readFile(fileURLToPath(new URL('../package.json', import.meta.url)), 'utf8'),
)

/** @type {Record<string, string[]>} */
const EXPECTED = {
  '.': [
    'createScorer',
    'createMatcher',
    'bestMatch',
    'search',
    'scoreMatrix',
    'scorePairs',
    'isMatch',
    'scoreIfMatch',
    'normalizeText',
  ],
  './fuzz': [
    'similarity',
    'partialSimilarity',
    'partialSimilarityAlignment',
    'tokenSortSimilarity',
    'tokenSetSimilarity',
    'tokenSimilarity',
    'partialTokenSortSimilarity',
    'partialTokenSetSimilarity',
    'partialTokenSimilarity',
    'fuzzySimilarity',
  ],
  './levenshtein': ['distance', 'similarity', 'editops', 'opcodes'],
  './indel': ['distance', 'similarity', 'editops', 'opcodes'],
  './lcs': ['distance', 'similarity', 'editops', 'opcodes'],
  './hamming': ['distance', 'similarity', 'editops', 'opcodes'],
  './osa': ['distance', 'similarity'],
  './damerau-levenshtein': ['distance', 'similarity'],
  './jaro': ['similarity'],
  './jaro-winkler': ['similarity'],
  './prefix': ['distance', 'similarity'],
  './postfix': ['distance', 'similarity'],
}

const REMOVED = [
  'configure',
  'extract',
  'extractOne',
  'extractIter',
  'prepareChoices',
  'prepareQuery',
  'prepareChoice',
  'matchScore',
  'defaultProcess',
  'ratio',
  'wRatio',
  'qRatio',
  'normalizedDistance',
  'normalizedSimilarity',
]

const declared = Object.keys(pkg.exports).filter((subpath) => subpath !== './package.json')
const missingChecks = declared.filter((subpath) => !(subpath in EXPECTED))
if (missingChecks.length > 0) {
  throw new Error(`missing export checks for ${missingChecks.join(', ')}`)
}

let failures = 0
for (const [subpath, names] of Object.entries(EXPECTED)) {
  const specifier = subpath === '.' ? pkg.name : `${pkg.name}/${subpath.slice(2)}`
  let module
  try {
    module = await import(specifier)
  } catch (error) {
    console.error(
      `✗ ${specifier} failed to import: ${error instanceof Error ? error.message : error}`,
    )
    failures++
    continue
  }
  const absent = names.filter((name) => typeof module[name] !== 'function')
  const present = REMOVED.filter((name) => module[name] !== undefined)
  if (absent.length > 0 || present.length > 0) {
    if (absent.length > 0) console.error(`✗ ${specifier} missing ${absent.join(', ')}`)
    if (present.length > 0) console.error(`✗ ${specifier} exposes ${present.join(', ')}`)
    failures++
  } else {
    console.log(`✓ ${specifier} (${names.length} names)`)
  }
}

if (failures > 0) process.exit(1)
console.log(`\nall ${declared.length} declared subpaths resolve`)
