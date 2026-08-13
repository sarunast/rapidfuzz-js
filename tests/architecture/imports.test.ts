import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'

import { describe, expect, it } from 'vitest'

const source = resolve(import.meta.dirname, '../../src')

function typeScriptFiles(directory: string): string[] {
  const files: string[] = []
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) files.push(...typeScriptFiles(path))
    else if (entry.isFile() && entry.name.endsWith('.ts')) files.push(path)
  }
  return files
}

function contents(directory: string): string {
  return typeScriptFiles(join(source, directory))
    .map((path) => readFileSync(path, 'utf8'))
    .join('\n')
}

// Both forms, because a side-effect `import './setup.js'` has no `from` and
// would otherwise be invisible to every graph built here.
function sourceImports(path: string): string[] {
  const imports: string[] = []
  const text = readFileSync(path, 'utf8')
  const specifiers = [
    ...text.matchAll(/from\s+['"]([^'"]+)['"]/g),
    ...text.matchAll(/import\s+['"]([^'"]+)['"]/g),
  ]
  for (const match of specifiers) {
    const specifier = match[1]
    if (!specifier.startsWith('.')) continue
    imports.push(resolve(dirname(path), specifier.replace(/\.js$/, '.ts')))
  }
  return imports
}

describe('dependency direction', () => {
  // Everything below reads specifiers off disk without resolving them, so a
  // typo silently drops an edge instead of failing: the cycle walk sees one
  // less edge, and reachability still passes whenever another importer keeps
  // both files in the graph. This is what makes the rest of them mean
  // something.
  it('keeps every relative static source import resolvable', () => {
    const missing: string[] = []
    for (const path of typeScriptFiles(source)) {
      for (const dependency of sourceImports(path)) {
        if (!existsSync(dependency)) {
          missing.push(`${relative(source, path)} -> ${relative(source, dependency)}`)
        }
      }
    }
    expect(missing).toEqual([])
  })

  it('keeps core algorithm-blind', () => {
    expect(contents('core')).not.toMatch(/from ['"][^'"]*(algorithms|fuzz|search|batch)/)
  })

  // The reason the subdirectory exists: types, sequence and normalize are what
  // scoring is built from, so an edge back into it would make the two halves
  // one layer again. Structural rather than a listing of the files directly in
  // `core/`, so a later `core/sequence/` cannot escape it.
  it('keeps core primitives below the scoring subsystem', () => {
    const scoring = join(source, 'core', 'scoring') + sep
    const crossings: string[] = []
    for (const path of typeScriptFiles(join(source, 'core'))) {
      if (path.startsWith(scoring)) continue
      for (const dependency of sourceImports(path)) {
        if (dependency.startsWith(scoring)) {
          crossings.push(`${relative(source, path)} -> ${relative(source, dependency)}`)
        }
      }
    }
    expect(crossings).toEqual([])
  })

  it('keeps search and batch independent from algorithms', () => {
    expect(contents('search')).not.toMatch(/from ['"][^'"]*(algorithms|fuzz)/)
    expect(contents('batch')).not.toMatch(/from ['"][^'"]*(algorithms|fuzz|search)/)
  })

  // search/ is grouped by lifetime: shared/ is input plumbing both modes use,
  // oneShot/ retains no corpus state, matcher/ builds it once and reuses it.
  // Only `-> shared` crosses; the two execution modes never reach into each
  // other. Judged by the directory an import *starts* in, so search/index.ts
  // naming both modes is the barrel doing its job rather than a cross edge.
  const SEARCH_SUBSYSTEMS: readonly string[] = ['shared', 'oneShot', 'matcher']

  function searchSubsystem(path: string): string | null {
    const within = relative(join(source, 'search'), path)
    if (within.startsWith('..')) return null
    const top = within.split(sep)[0]
    return SEARCH_SUBSYSTEMS.includes(top) ? top : null
  }

  it('keeps the search subsystems from tangling', () => {
    const crossings: string[] = []
    for (const path of typeScriptFiles(join(source, 'search'))) {
      const from = searchSubsystem(path)
      if (from === null) continue
      for (const dependency of sourceImports(path)) {
        const to = searchSubsystem(dependency)
        if (to === null || to === from || to === 'shared') continue
        crossings.push(`${relative(source, path)} -> ${relative(source, dependency)}`)
      }
    }
    expect(crossings).toEqual([])
  })

  it('keeps the root entrypoint orchestration-only', () => {
    const root = readFileSync(join(source, 'index.ts'), 'utf8')
    expect(root).not.toMatch(/from ['"][^'"]*(algorithms|fuzz)/)
  })

  it('keeps the complete source graph acyclic', () => {
    const files = typeScriptFiles(source)
    const sourceFiles = new Set(files)
    const graph = new Map(
      files.map((path) => [
        path,
        sourceImports(path).filter((dependency) => sourceFiles.has(dependency)),
      ]),
    )
    const complete = new Set<string>()
    const active = new Set<string>()
    const stack: string[] = []
    const cycles: string[] = []

    const visit = (path: string): void => {
      if (active.has(path)) {
        const start = stack.indexOf(path)
        cycles.push(
          stack
            .slice(start)
            .concat(path)
            .map((entry) => relative(source, entry))
            .join(' -> '),
        )
        return
      }
      if (complete.has(path)) return
      active.add(path)
      stack.push(path)
      for (const dependency of graph.get(path) ?? []) visit(dependency)
      stack.pop()
      active.delete(path)
      complete.add(path)
    }

    for (const path of files) visit(path)
    expect(cycles).toEqual([])
  })

  it('keeps every source module reachable from a public entrypoint', () => {
    const entries = [
      'index.ts',
      'fuzz/index.ts',
      'algorithms/cosine/index.ts',
      'algorithms/damerauLevenshtein/index.ts',
      'algorithms/dice/index.ts',
      'algorithms/hamming/index.ts',
      'algorithms/indel/index.ts',
      'algorithms/jaro/index.ts',
      'algorithms/jaroWinkler/index.ts',
      'algorithms/lcs/index.ts',
      'algorithms/levenshtein/index.ts',
      'algorithms/osa/index.ts',
      'algorithms/postfix/index.ts',
      'algorithms/prefix/index.ts',
    ].map((path) => join(source, path))
    const reachable = new Set<string>()
    const visit = (path: string): void => {
      if (reachable.has(path)) return
      reachable.add(path)
      for (const dependency of sourceImports(path)) {
        if (existsSync(dependency)) visit(dependency)
      }
    }
    for (const entry of entries) visit(entry)

    expect(
      typeScriptFiles(source)
        .filter((path) => !reachable.has(path))
        .map((path) => relative(source, path)),
    ).toEqual([])
  })

  it('keeps Levenshtein internals below its public orchestration modules', () => {
    const directory = join(source, 'algorithms/levenshtein')
    const forbidden = (names: readonly string[]): Set<string> =>
      new Set(names.map((name) => join(directory, `${name}.ts`)))
    const dependenciesInto = (paths: readonly string[], denied: Set<string>): string[] =>
      paths.flatMap(sourceImports).filter((dependency) => denied.has(dependency))

    expect(
      dependenciesInto(
        typeScriptFiles(join(directory, 'internal')),
        forbidden(['compile', 'editops', 'metric', 'prepare']),
      ),
    ).toEqual([])
    expect(
      dependenciesInto(
        [join(directory, 'prepare.ts')],
        forbidden(['compile', 'editops', 'metric']),
      ),
    ).toEqual([])
    expect(
      dependenciesInto(
        [join(directory, 'editops.ts')],
        forbidden(['compile', 'metric', 'prepare']),
      ),
    ).toEqual([])
  })

  it('keeps shared bitmask code representation-only', () => {
    const directory = join(source, 'algorithms/shared/bitmask')
    expect(readdirSync(directory).sort()).toEqual([
      'blockMasks.ts',
      'lookup.ts',
      'pattern.ts',
    ])
  })

  it('keeps the n-gram subsystem laid out as its layers', () => {
    const ngram = join(source, 'algorithms/shared/ngram')
    expect(readdirSync(ngram).sort()).toEqual([
      'README.md',
      'compare.ts',
      'gramSize.ts',
      'inverted',
      'kernel.ts',
      'key.ts',
      'packing.ts',
      'profile.ts',
    ])
    expect(readdirSync(join(ngram, 'inverted')).sort()).toEqual([
      'builder.ts',
      'cosine.ts',
      'dice.ts',
      'keys.ts',
      'query.ts',
    ])
  })

  it('keeps the n-gram semantics below the index built on them', () => {
    // The inverted index is an optional acceleration strategy over n-gram
    // semantics, never the foundation: a profile, a comparison or a prepared
    // kernel that reached into `inverted/` would invert that.
    const ngram = join(source, 'algorithms/shared/ngram')
    const inverted = join(ngram, 'inverted')
    const semantics = typeScriptFiles(ngram).filter(
      (path) => !path.startsWith(`${inverted}${sep}`),
    )
    expect(
      semantics.flatMap((path) =>
        sourceImports(path)
          .filter((dependency) => dependency.startsWith(`${inverted}${sep}`))
          .map(
            (dependency) =>
              `${relative(source, path)} -> ${relative(source, dependency)}`,
          ),
      ),
    ).toEqual([])
  })

  it('keeps the n-gram index off the semantics it accelerates', () => {
    // The other direction of the rule above, and the stronger half: the index
    // reaches back into `ngram/` for the key arithmetic and nothing else, so it
    // shares an encoding with the profiles without sharing a representation.
    const ngram = join(source, 'algorithms/shared/ngram')
    const inverted = join(ngram, 'inverted')
    const permitted = join(ngram, 'key.ts')
    expect(
      typeScriptFiles(inverted).flatMap((path) =>
        sourceImports(path)
          .filter(
            (dependency) =>
              dependency.startsWith(`${ngram}${sep}`) &&
              !dependency.startsWith(`${inverted}${sep}`) &&
              dependency !== permitted,
          )
          .map(
            (dependency) =>
              `${relative(source, path)} -> ${relative(source, dependency)}`,
          ),
      ),
    ).toEqual([])
  })

  it('keeps the n-gram key arithmetic and option parsing free of dependencies', () => {
    // Both are leaves by construction — one is integer arithmetic over a radix
    // ladder, the other reads a single option — and an edge out of either is
    // the first sign that policy has leaked into them.
    const ngram = join(source, 'algorithms/shared/ngram')
    expect(sourceImports(join(ngram, 'key.ts'))).toEqual([])
    expect(sourceImports(join(ngram, 'gramSize.ts'))).toEqual([])
  })

  it('keeps fuzz families physically and directionally isolated', () => {
    const directory = join(source, 'fuzz')
    expect(readdirSync(directory).sort()).toEqual([
      'index.ts',
      'internal',
      'partial.ts',
      'partialToken.ts',
      'partialTokenSet.ts',
      'partialTokenSort.ts',
      'similarity.ts',
      'token.ts',
      'tokenSet.ts',
      'tokenSort.ts',
      'types.ts',
      'weighted.ts',
    ])
    expect(readFileSync(join(directory, 'similarity.ts'), 'utf8')).not.toMatch(
      /from ['"]\.\/(weighted|partial|partialToken|partialTokenSet|partialTokenSort|token|tokenSet|tokenSort)\.js/,
    )
    expect(readFileSync(join(directory, 'partial.ts'), 'utf8')).not.toMatch(
      /from ['"]\.\/(weighted|partialToken|partialTokenSet|partialTokenSort|token|tokenSet|tokenSort)\.js/,
    )
    for (const name of [
      'partialToken.ts',
      'partialTokenSet.ts',
      'partialTokenSort.ts',
      'token.ts',
      'tokenSet.ts',
      'tokenSort.ts',
    ]) {
      expect(readFileSync(join(directory, name), 'utf8')).not.toMatch(
        /from ['"][^'"]*weighted/,
      )
    }
  })
})

describe('removed architecture', () => {
  it.each([
    '_common.ts',
    'configure.ts',
    'distance/index.ts',
    'internal',
    'match.ts',
    'search.ts',
  ])('does not restore src/%s', (path) => {
    expect(existsSync(join(source, path))).toBe(false)
  })

  it('does not restore the old distance tree', () => {
    expect(existsSync(join(source, 'distance'))).toBe(false)
  })
})

describe('metric identity', () => {
  const declarations = [contents('algorithms'), contents('fuzz')].join('\n')

  it('gives every built-in metric a name of its own', () => {
    const ids = [...declarations.matchAll(/BuiltInMetric<\s*'([\w.]+)'/g)].map(
      (match) => match[1],
    )
    // A copy-pasted name would silently make two metrics accept each other's
    // prepared choices, which is the one mistake the type cannot catch.
    expect(ids.length).toBeGreaterThan(40)
    expect([...new Set(ids)]).toHaveLength(ids.length)
  })

  it('lets a known alias share an identity rather than mint one', () => {
    // Cosine, Dice, Jaro and Jaro-Winkler are normalized by construction, so
    // their normalized exports are the same metrics. `typeof` is what says so,
    // and an alias that named itself instead would appear below as an id.
    const aliases = [
      ...declarations.matchAll(/export const (\w+): typeof (\w+) = (\w+)/g),
    ]
    expect(aliases).toHaveLength(8)
    for (const [, name, annotation, initializer] of aliases) {
      expect(annotation).toBe(initializer)
      expect(name).not.toBe(initializer)
    }
    const ids = [...declarations.matchAll(/BuiltInMetric<\s*'([\w.]+)'/g)].map(
      (match) => match[1],
    )
    for (const family of ['cosine', 'dice', 'jaro', 'jaroWinkler']) {
      expect(ids).not.toContain(`${family}.normalizedDistance`)
      expect(ids).not.toContain(`${family}.normalizedSimilarity`)
    }
  })

  it('leaves the mechanics of identity to the adapter', () => {
    expect(declarations).not.toMatch(/declare const \w+: unique symbol/)
  })
})
