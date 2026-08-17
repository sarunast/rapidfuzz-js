// @ts-check
// Typechecks a consumer against the BUILT declarations rather than `src/`.
//
// A metric's brand is its id literal, `'levenshtein.distance'`, written once
// in the type of the metric it names. If declaration emit ever stopped
// writing those brands out, or wrote them somewhere a consumer cannot name,
// `PreparedChoiceOf` would resolve to an unusable type — and nothing inside
// `src/` could tell.
// Declaration emit erasing the type of a `private` member is exactly how that
// went wrong once; a brand spelled as an internal wrapper type is how a
// consumer's own emit went wrong once, which is what the third check watches.
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
// Inside the package rather than the system temp directory: `rapidfuzz-js`
// resolves to `dist` through the package's own `exports`, and that only
// happens for a file within it.
const output = mkdtempSync(join(root, '.consumer-check-'))
const consumer = join(output, 'consumer.ts')
const config = join(output, 'tsconfig.json')

const source = `import {
  createIndexedMatcher,
  createMatcher,
  createScorer,
  searchIter,
} from 'rapidfuzz-js'
import type { PreparedChoice, PreparedChoiceOf, Scorer, ScorerOf } from 'rapidfuzz-js'
import { ratio, tokenSetRatio } from 'rapidfuzz-js/fuzz'
import { distance } from 'rapidfuzz-js/levenshtein'
import { distance as jaroWinklerDistance } from 'rapidfuzz-js/jaro-winkler'
import { similarity as diceSimilarity } from 'rapidfuzz-js/dice'
import { similarity as tverskySimilarity } from 'rapidfuzz-js/tversky'
import type {
  TverskyElementSimilarity,
  TverskyExplainConfiguration,
} from 'rapidfuzz-js/tversky'
import { normalizedSimilarity as indelNormalizedSimilarity } from 'rapidfuzz-js/indel'

const scorer = createScorer(tokenSetRatio)

export interface Stored {
  readonly name: string
  readonly prepared: PreparedChoiceOf<typeof scorer>
}

export const store = (name: string): Stored => ({
  name,
  prepared: scorer.prepareChoice(name),
})

export const stream = (query: string, rows: Iterable<Stored>) =>
  searchIter(query, rows, { scorer, getPrepared: (row) => row.prepared })

export const matcher = (rows: readonly Stored[]) =>
  createMatcher(rows, { scorer, getPrepared: (row) => row.prepared })

// A matcher keeps the brand of the scorer it was built from, so a handle made
// through it is the same kind of handle as one made directly.
export const fromMatcher = (rows: readonly Stored[]): Stored['prepared'] =>
  matcher(rows).scorer.prepareChoice('alpha')

// Unannotated on purpose: an indexed matcher's type is inferred straight into
// a consumer's own declaration emit, so anything it names has to be reachable
// through the export map.
export const indexed = createIndexedMatcher(['alpha', 'beta'], {
  scorer: createScorer(diceSimilarity, { gramSize: 3 }),
})
export const indexedBest = indexed.best('alpha', { threshold: 0.5 })

// Also unannotated, and deliberately WITHOUT importing \`ExplainableScorer\` or
// \`TverskyEvidence\`: naming them here would hand declaration emit a name it
// already has, which is the thing being tested. Inference has to reach both
// through the export map on its own.
export const company = createScorer(tverskySimilarity, {
  gramSize: 1,
  alpha: 1,
  beta: 0.1,
  elementWeights: new Map<unknown, number>([['ag', 0.1]]),
})
export const explained = company.explain(['swisscom', 'ag'], ['swisscom'])

// The capability survives a configuration hoisted with \`satisfies\` rather than
// written inline, which is the documented way to keep it.
const hoisted = { gramSize: 1, alpha: 1, beta: 0 } satisfies TverskyExplainConfiguration
export const containment = createScorer(tverskySimilarity, hoisted)
export const containmentEvidence = containment.explain(['swisscom'], ['swisscom', 'ag'])

// A nested scorer is a second inference surface: the option holds a
// \`Scorer<'similarity'>\` and the result still has to emit as an explainable
// scorer without naming anything a consumer cannot reach.
const elementSimilarity: TverskyElementSimilarity = {
  scorer: createScorer(indelNormalizedSimilarity),
  threshold: 0.8,
}
export const fuzzyCompany = createScorer(tverskySimilarity, {
  gramSize: 1,
  elementSimilarity,
})
export const fuzzyEvidence = fuzzyCompany.explain(['swisscom', 'ag'], ['swisscomm', 'ag'])

// Widening stays possible: \`Scorer<D>\` is still the type that holds a scorer
// of any metric, which is what most annotations want.
export const held: Scorer<'distance'> = createScorer(distance)
export const many: Scorer<'similarity'>[] = [
  createScorer(ratio),
  createScorer(tokenSetRatio),
]
export const scoreWith = (held: Scorer<'similarity'>) => held.score('a', 'b')

// A loop over metrics compiles each of them, including three whose
// configurations have no key in common: without a configuration argument there
// is nothing for \`Config\` to be inferred from, and inferring it anyway is what
// used to refuse this.
export const eachMetric = (
  [distance, jaroWinklerDistance, diceSimilarity] as const
).map((metric) => createScorer(metric))

// A configured scorer's handle type is nameable too, and \`gramSize\` reaches the
// consumer as an ordinary optional number rather than through an internal type.
const grams = createScorer(diceSimilarity, { gramSize: 3 })
export const gramHandle = (name: string): PreparedChoiceOf<typeof grams> =>
  grams.prepareChoice(name)

// A scorer's exact type is nameable from the metric alone, brand included —
// the annotation a stored scorer wants without \`typeof\` gymnastics.
export const titled: ScorerOf<typeof tokenSetRatio> =
  createScorer(tokenSetRatio)

// Both spellings of a handle's type are nameable from outside the package.
export type Handle = PreparedChoiceOf<typeof scorer>
export type OpaqueHandle = PreparedChoice

// Opaque: nothing to read off it but its type.
export const keysOf = (prepared: Handle): string[] => Object.keys(prepared)
`

// The other half of the contract: a handle from one metric must not typecheck
// where another metric's scorer is used. A brand that stopped surviving
// declaration emit would still compile the file above — it would compile this
// one too, which is how the difference is detected.
const CROSSED_LINE = 13
const crossed = `import { bestMatch, createScorer } from 'rapidfuzz-js'
import { tokenSetRatio } from 'rapidfuzz-js/fuzz'
import { distance } from 'rapidfuzz-js/levenshtein'

const titles = createScorer(tokenSetRatio)
const companies = createScorer(distance)
const rows = [{ prepared: titles.prepareChoice('a') }]

// Line ${CROSSED_LINE} below — the getPrepared line — is the one the
// diagnostic has to name: this scorer cannot take these prepared choices.
export const crossed = bestMatch('a', rows, {
  scorer: companies,
  getPrepared: (row) => row.prepared,
})
`

const DISTANCE_LINE = 9
const distanceIndex = `import { createIndexedMatcher, createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/dice'

const lengths = createScorer(distance)

// Line ${DISTANCE_LINE} below — the call — is the one the diagnostic has to
// name: an index ranks by how much two sequences share, so a distance scorer is
// refused by the type rather than at construction, and no overload accepts one.
export const ranked = createIndexedMatcher(['a'], {
  scorer: lengths,
})
`

const tsconfig = {
  compilerOptions: {
    module: 'nodenext',
    moduleResolution: 'nodenext',
    target: 'es2022',
    strict: true,
    exactOptionalPropertyTypes: true,
    noEmit: true,
    skipLibCheck: false,
    types: [],
  },
  include: [consumer],
}

/**
 * @param {string} configPath
 * @returns {string | null} the compiler's output, or null when it succeeded
 */
function typecheck(configPath) {
  try {
    execFileSync(resolve(root, 'node_modules/.bin/tsc'), ['-p', configPath], {
      cwd: root,
      encoding: 'utf8',
      stdio: 'pipe',
    })
    return null
  } catch (error) {
    const result = error instanceof Error ? Reflect.get(error, 'stdout') : undefined
    return typeof result === 'string' ? result : ''
  }
}

try {
  writeFileSync(consumer, source)
  writeFileSync(config, JSON.stringify(tsconfig))
  const failure = typecheck(config)
  if (failure !== null) {
    throw new Error(`the consumer did not typecheck against dist:\n${failure}`)
  }
  console.log('✓ packed declarations typecheck from a consumer')

  const crossedFile = join(output, 'crossed.ts')
  const crossedConfig = join(output, 'tsconfig.crossed.json')
  writeFileSync(crossedFile, crossed)
  writeFileSync(crossedConfig, JSON.stringify({ ...tsconfig, include: [crossedFile] }))
  const refusal = typecheck(crossedConfig)
  if (refusal === null) {
    throw new Error("another metric's prepared choice compiled")
  }
  // Any compiler error would end up here otherwise — an unrelated one in a
  // later release would then read as this check passing.
  if (!refusal.includes(`crossed.ts(${CROSSED_LINE},`)) {
    throw new Error(
      `expected the refusal to name crossed.ts line ${CROSSED_LINE}:\n${refusal}`,
    )
  }
  if (!refusal.includes('PreparedChoice')) {
    throw new Error(`expected the refusal to name PreparedChoice:\n${refusal}`)
  }
  console.log("✓ another metric's prepared choice is refused at compile time")

  const distanceFile = join(output, 'distanceIndex.ts')
  const distanceConfig = join(output, 'tsconfig.distance.json')
  writeFileSync(distanceFile, distanceIndex)
  writeFileSync(distanceConfig, JSON.stringify({ ...tsconfig, include: [distanceFile] }))
  const distanceRefusal = typecheck(distanceConfig)
  if (distanceRefusal === null) {
    throw new Error('an indexed matcher accepted a distance scorer')
  }
  if (!distanceRefusal.includes(`distanceIndex.ts(${DISTANCE_LINE},`)) {
    throw new Error(
      `expected the refusal to name distanceIndex.ts line ${DISTANCE_LINE}:\n${distanceRefusal}`,
    )
  }
  // The positive control lives in the main fixture above, which builds an
  // indexed matcher from a similarity scorer and has already typechecked — so
  // this refusal is about the direction rather than about the call shape.
  console.log('✓ an indexed matcher refuses a distance scorer at compile time')

  // The third check: the consumer's OWN declaration emit. An inferred scorer
  // type in an exported const makes the compiler spell the brand out; if that
  // spelling ever needs a type of ours that the export map does not reach, the
  // emit degrades to a deep `import("...")` into the package — declarations
  // that typecheck here and break the moment they are published.
  const emitDir = join(output, 'emit')
  const emitConfig = join(output, 'tsconfig.emit.json')
  writeFileSync(
    emitConfig,
    JSON.stringify({
      ...tsconfig,
      compilerOptions: {
        ...tsconfig.compilerOptions,
        noEmit: false,
        declaration: true,
        emitDeclarationOnly: true,
        outDir: emitDir,
      },
    }),
  )
  const emitFailure = typecheck(emitConfig)
  if (emitFailure !== null) {
    throw new Error(`the consumer's declaration emit failed:\n${emitFailure}`)
  }
  const emitted = readFileSync(join(emitDir, 'consumer.d.ts'), 'utf8')
  if (emitted.includes('dist/')) {
    throw new Error(
      `the consumer's declarations deep-import into the package:\n${emitted}`,
    )
  }
  if (!emitted.includes(`"fuzz.tokenSetRatio"`)) {
    throw new Error(
      `expected the emitted declarations to spell the brand as a literal:\n${emitted}`,
    )
  }
  // The capability has to survive inference too. Nothing above would fail if a
  // metric quietly stopped declaring one — the scorer would simply emit as
  // `Scorer<…>` and still typecheck.
  if (!emitted.includes('ExplainableScorer<"similarity", "tversky.similarity"')) {
    throw new Error(
      `expected an inferred explainable scorer to keep its capability:\n${emitted}`,
    )
  }
  console.log("✓ the consumer's own declaration emit stays portable")
} finally {
  rmSync(output, { recursive: true, force: true })
}
