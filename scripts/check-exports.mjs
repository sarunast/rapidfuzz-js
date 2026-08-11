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
    'searchIter',
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
  './levenshtein': [
    'distance',
    'similarity',
    'normalizedDistance',
    'normalizedSimilarity',
    'editops',
    'opcodes',
  ],
  './indel': [
    'distance',
    'similarity',
    'normalizedDistance',
    'normalizedSimilarity',
    'editops',
    'opcodes',
  ],
  './lcs': [
    'distance',
    'similarity',
    'normalizedDistance',
    'normalizedSimilarity',
    'editops',
    'opcodes',
  ],
  './hamming': [
    'distance',
    'similarity',
    'normalizedDistance',
    'normalizedSimilarity',
    'editops',
    'opcodes',
  ],
  './osa': ['distance', 'similarity', 'normalizedDistance', 'normalizedSimilarity'],
  './damerau-levenshtein': [
    'distance',
    'similarity',
    'normalizedDistance',
    'normalizedSimilarity',
  ],
  './jaro': ['distance', 'similarity', 'normalizedDistance', 'normalizedSimilarity'],
  './jaro-winkler': [
    'distance',
    'similarity',
    'normalizedDistance',
    'normalizedSimilarity',
  ],
  './prefix': ['distance', 'similarity', 'normalizedDistance', 'normalizedSimilarity'],
  './postfix': ['distance', 'similarity', 'normalizedDistance', 'normalizedSimilarity'],
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
]

const REMOVED_SUBPATHS = [
  'distance',
  'process',
  'utils',
  'operations',
  'scorer',
  'fuzz-v2',
  'Levenshtein',
  'JaroWinkler',
]

const declared = Object.keys(pkg.exports).filter(
  (subpath) => subpath !== './package.json',
)
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
  const expected = new Set(names)
  const unexpected = Object.keys(module).filter((name) => !expected.has(name))
  const present = REMOVED.filter((name) => module[name] !== undefined)
  if (absent.length > 0 || unexpected.length > 0 || present.length > 0) {
    if (absent.length > 0) console.error(`✗ ${specifier} missing ${absent.join(', ')}`)
    if (unexpected.length > 0) {
      console.error(`✗ ${specifier} unexpectedly exposes ${unexpected.join(', ')}`)
    }
    if (present.length > 0) console.error(`✗ ${specifier} exposes ${present.join(', ')}`)
    failures++
  } else {
    console.log(`✓ ${specifier} (${names.length} names)`)
  }
}

for (const subpath of REMOVED_SUBPATHS) {
  const specifier = `${pkg.name}/${subpath}`
  try {
    await import(specifier)
    console.error(`✗ removed subpath ${specifier} still resolves`)
    failures++
  } catch (error) {
    if (
      error instanceof Error &&
      'code' in error &&
      error.code === 'ERR_PACKAGE_PATH_NOT_EXPORTED'
    ) {
      console.log(`✓ removed subpath ${specifier} is blocked`)
    } else {
      console.error(
        `✗ ${specifier} failed for the wrong reason: ${error instanceof Error ? error.message : error}`,
      )
      failures++
    }
  }
}

if (failures > 0) process.exit(1)
console.log(`\nall ${declared.length} declared subpaths resolve`)
