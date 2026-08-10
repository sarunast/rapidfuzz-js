// @ts-check
/**
 * Assert every declared subpath resolves and carries the names it should.
 *
 * Deliberately a script run after `build` rather than a Vitest file. `test`
 * runs before `build`, so a test importing `../src/search.js` would not
 * exercise the export map at all, and one importing `rapidfuzz-js/search`
 * would resolve against whatever stale `dist/` happened to be on disk. Nothing
 * else covers the barrels or the namespace modules at run time.
 *
 * The imports are self-references: Node resolves them through this package's
 * own `exports` field, which is the thing under test.
 */
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const pkg = JSON.parse(
  await readFile(fileURLToPath(new URL('../package.json', import.meta.url)), 'utf8'),
)

/** Subpath -> names that must be present and callable. */
/** @type {Record<string, string[]>} */
const EXPECTED = {
  '.': [
    'configure',
    'matchScore',
    'isMatch',
    'extract',
    'extractOne',
    'extractIter',
    'scoreMatrix',
    'scorePairs',
    'prepareChoices',
    'ratio',
    'wRatio',
    'levenshteinDistance',
    'defaultProcess',
    'Editops',
    'Opcodes',
  ],
  './fuzz': ['ratio', 'partialRatio', 'wRatio', 'qRatio', 'tokenSortRatio'],
  './search': [
    'extract',
    'extractOne',
    'extractIter',
    'scoreMatrix',
    'scorePairs',
    'prepareChoices',
  ],
  './match': ['matchScore', 'isMatch'],
  './utils': ['defaultProcess'],
  // The namespace barrel mirroring Python's `rapidfuzz.distance` package, so
  // these are the module objects rather than flat scorer names.
  './distance': [
    'Indel',
    'LCSseq',
    'Levenshtein',
    'DamerauLevenshtein',
    'OSA',
    'Hamming',
    'Jaro',
    'JaroWinkler',
    'Prefix',
    'Postfix',
    'Editops',
    'Opcodes',
  ],
}

/** Every `./distance/<Name>` module exposes the same four entry points. */
const METRIC = ['distance', 'similarity', 'normalizedDistance', 'normalizedSimilarity']
for (const subpath of Object.keys(pkg.exports)) {
  if (subpath.startsWith('./distance/')) EXPECTED[subpath] = METRIC
}

const declared = Object.keys(pkg.exports).filter((s) => s !== './package.json')
const missing = declared.filter((s) => !(s in EXPECTED))
if (missing.length > 0) {
  throw new Error(
    `package.json declares ${missing.join(', ')} but check-exports.mjs does not check them`,
  )
}

let failures = 0
for (const [subpath, names] of Object.entries(EXPECTED)) {
  const specifier = subpath === '.' ? pkg.name : `${pkg.name}/${subpath.slice(2)}`
  let module
  try {
    module = await import(specifier)
  } catch (error) {
    console.error(
      `✗ ${specifier} failed to import: ` +
        `${error instanceof Error ? error.message : error}`,
    )
    failures++
    continue
  }

  const absent = names.filter((name) => module[name] === undefined)
  if (absent.length > 0) {
    console.error(`✗ ${specifier} is missing ${absent.join(', ')}`)
    failures++
    continue
  }

  // `cdist` and `cpdist` were renamed; leaving a stale one behind would mean
  // the barrel and the module had drifted apart.
  //
  // The two `fromValidated` functions are a different question: they are not
  // stale, they are the door past every check `fromOperations` makes, and a
  // barrel that widened to `export *` would publish them. Reaching them has to
  // stay a matter of importing `src/distance/editops.js` directly, which the
  // export map does not offer.
  const removed = [
    'cdist',
    'cpdist',
    'scorerKwargs',
    'editopsFromValidated',
    'opcodesFromValidated',
  ].filter((name) => module[name] !== undefined)
  if (removed.length > 0) {
    console.error(`✗ ${specifier} still exports ${removed.join(', ')}`)
    failures++
    continue
  }

  console.log(`✓ ${specifier} (${names.length} names)`)
}

if (failures > 0) {
  console.error(`\n${failures} subpath(s) failed`)
  process.exit(1)
}
console.log(`\nall ${declared.length} declared subpaths resolve`)
