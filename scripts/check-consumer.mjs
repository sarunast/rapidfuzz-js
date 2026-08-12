// @ts-check
// Typechecks a consumer against the BUILT declarations rather than `src/`.
//
// A metric's brand is `MetricBrand<'levenshtein.distance'>`, written once in
// the type of the metric it names. If declaration emit ever stopped writing
// those brands out, or wrote them somewhere a consumer cannot name,
// `PreparedChoiceOf` would resolve to an unusable type — and nothing inside
// `src/` could tell.
// Declaration emit erasing the type of a `private` member is exactly how that
// went wrong once.
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
// Inside the package rather than the system temp directory: `rapidfuzz-js`
// resolves to `dist` through the package's own `exports`, and that only
// happens for a file within it.
const output = mkdtempSync(join(root, '.consumer-check-'))
const consumer = join(output, 'consumer.ts')
const config = join(output, 'tsconfig.json')

const source = `import { createMatcher, createScorer, searchIter } from 'rapidfuzz-js'
import type { PreparedChoice, PreparedChoiceOf, Scorer } from 'rapidfuzz-js'
import { similarity, tokenSetSimilarity } from 'rapidfuzz-js/fuzz'
import { distance } from 'rapidfuzz-js/levenshtein'

const scorer = createScorer(tokenSetSimilarity)

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

// Widening stays possible: \`Scorer<D>\` is still the type that holds a scorer
// of any metric, which is what most annotations want.
export const held: Scorer<'distance'> = createScorer(distance)
export const many: Scorer<'similarity'>[] = [
  createScorer(similarity),
  createScorer(tokenSetSimilarity),
]
export const scoreWith = (held: Scorer<'similarity'>) => held.score('a', 'b')

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
import { tokenSetSimilarity } from 'rapidfuzz-js/fuzz'
import { distance } from 'rapidfuzz-js/levenshtein'

const titles = createScorer(tokenSetSimilarity)
const companies = createScorer(distance)
const rows = [{ prepared: titles.prepareChoice('a') }]

// Line ${CROSSED_LINE} below — the getPrepared line — is the one the
// diagnostic has to name: this scorer cannot take these prepared choices.
export const crossed = bestMatch('a', rows, {
  scorer: companies,
  getPrepared: (row) => row.prepared,
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
} finally {
  rmSync(output, { recursive: true, force: true })
}
