import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'

import { describe, expect, it } from 'vitest'

const source = resolve(import.meta.dirname, '../../src')

// Every rule in this file describes the architecture of the shipping library,
// so the module-owned tests living beside it are not part of the graph: they
// import vitest, they reach across subsystems on purpose, and they are
// reachable from no public entrypoint. Filtering here rather than in each rule
// is what keeps that one decision in one place — `contents()` and every walk
// below run through it.
function typeScriptFiles(directory: string): string[] {
  const files: string[] = []
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) files.push(...typeScriptFiles(path))
    else if (
      entry.isFile() &&
      entry.name.endsWith('.ts') &&
      !entry.name.endsWith('.test.ts')
    ) {
      files.push(path)
    }
  }
  return files
}

// The physical-layout assertions read a directory directly rather than through
// the walk above, so they need the same exclusion: they pin what the subsystem
// ships, not how many files test it.
function shippedEntries(directory: string): string[] {
  return readdirSync(directory)
    .filter((name) => !name.endsWith('.test.ts'))
    .sort()
}

function contents(directory: string): string {
  return typeScriptFiles(join(source, directory))
    .map((path) => readFileSync(path, 'utf8'))
    .join('\n')
}

// A cross-subsystem import is written `#core/sequence.js` rather than climbing
// out with `../../`, so resolving those here is what keeps every rule below
// seeing the whole graph. Skipping them instead would not fail anything
// loudly — it would drop edges, and the cycle walk and the foundation rule
// would quietly stop covering them.
function resolveAlias(specifier: string): string | null {
  const match = /^#([^/]+)\/(.+)$/.exec(specifier)
  return match === null ? null : join(source, match[1] ?? '', match[2] ?? '')
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
    const aliased = resolveAlias(specifier)
    if (aliased !== null) {
      imports.push(aliased.replace(/\.js$/, '.ts'))
      continue
    }
    if (!specifier.startsWith('.')) continue
    imports.push(resolve(dirname(path), specifier.replace(/\.js$/, '.ts')))
  }
  return imports
}

// The public subpaths, and the one list two rules read: reachability walks
// from them, and `algorithmDirectory` derives from them what counts as a
// public algorithm directory.
const ENTRYPOINTS: readonly string[] = [
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
  'algorithms/tversky/index.ts',
]

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

  // The rule colocation makes necessary. Filtering `*.test.ts` out of every
  // graph above leaves a blind spot: a production module could import one, or
  // reach into `testing/`, and nothing here would see it. So state the whole
  // containment instead — shipping source reaches shipping source and nothing
  // else, not a test beside it, not the machinery those tests share. At zero
  // today, which makes it a guard against a future edit rather than a cleanup.
  it('keeps shipping source importing only shipping source', () => {
    const escapes: string[] = []
    for (const path of typeScriptFiles(source)) {
      for (const dependency of sourceImports(path)) {
        if (
          !dependency.startsWith(`${source}${sep}`) ||
          dependency.endsWith('.test.ts')
        ) {
          escapes.push(`${relative(source, path)} -> ${relative(source, dependency)}`)
        }
      }
    }
    expect(escapes).toEqual([])
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
    const entries = ENTRYPOINTS.map((path) => join(source, path))
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

  const FOUNDATION_EDGES: readonly string[] = [
    'indel -> lcs',
    'jaroWinkler -> jaro',
    'levenshtein -> lcs',
  ]

  // A public algorithm directory is one with a published subpath, derived from
  // the entrypoint list rather than named again here. The foundations —
  // `affix.ts`, `bitmask/`, `ngram/` — have no `index.ts` and are therefore not
  // algorithms for the purposes of the cross-algorithm rule below; the rule
  // after it is what governs them instead.
  const ALGORITHM_DIRECTORIES: readonly string[] = ENTRYPOINTS.filter((path) =>
    path.startsWith('algorithms/'),
  ).map((path) => path.split('/')[1] ?? '')

  const ALGORITHM_FOUNDATIONS: readonly string[] = ['affix.ts', 'bitmask', 'ngram']

  function algorithmDirectory(path: string): string | null {
    const within = relative(join(source, 'algorithms'), path)
    if (within.startsWith('..') || !within.includes(sep)) return null
    const top = within.split(sep)[0] ?? ''
    return ALGORITHM_DIRECTORIES.includes(top) ? top : null
  }

  it('lets an algorithm depend only on the foundations it is defined on', () => {
    const crossings: string[] = []
    const seen = new Set<string>()
    for (const path of typeScriptFiles(join(source, 'algorithms'))) {
      const from = algorithmDirectory(path)
      if (from === null) continue
      for (const dependency of sourceImports(path)) {
        const to = algorithmDirectory(dependency)
        if (to === null || to === from) continue
        const edge = `${from} -> ${to}`
        seen.add(edge)
        if (!FOUNDATION_EDGES.includes(edge)) {
          crossings.push(`${relative(source, path)} -> ${relative(source, dependency)}`)
        }
      }
    }
    expect(crossings).toEqual([])
    expect([...seen].sort()).toEqual([...FOUNDATION_EDGES].sort())
  })

  // The other half of the rule above, and what keeps its silence about the
  // foundations from being three unstated exemptions. `affix.ts`, `bitmask/`
  // and `ngram/` are below the algorithms built on them: an edge back up would
  // be caught by nothing otherwise, because `algorithmDirectory` answers null
  // for their own files.
  it('keeps the algorithm foundations below the algorithms built on them', () => {
    const roots = ALGORITHM_FOUNDATIONS.map((name) => join(source, 'algorithms', name))
    const inFoundation = (path: string): boolean =>
      roots.some((root) => path === root || path.startsWith(`${root}${sep}`))

    expect(
      typeScriptFiles(join(source, 'algorithms'))
        .filter(inFoundation)
        .flatMap((path) =>
          sourceImports(path)
            .filter((dependency) => algorithmDirectory(dependency) !== null)
            .map(
              (dependency) =>
                `${relative(source, path)} -> ${relative(source, dependency)}`,
            ),
        ),
    ).toEqual([])

    // And that a fourth foundation cannot appear outside the rule: every entry
    // directly under `algorithms/` is a public algorithm or a named foundation.
    expect(
      shippedEntries(join(source, 'algorithms')).filter(
        (name) => !ALGORITHM_DIRECTORIES.includes(name),
      ),
    ).toEqual([...ALGORITHM_FOUNDATIONS])
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

  it('keeps the bitmask foundation representation-only', () => {
    const directory = join(source, 'algorithms/bitmask')
    expect(shippedEntries(directory)).toEqual([
      'blockMasks.ts',
      'pattern.ts',
      'positionMasks.ts',
      'rowBits.ts',
      'words.ts',
    ])
  })

  it('keeps the n-gram subsystem laid out as its layers', () => {
    const ngram = join(source, 'algorithms/ngram')
    expect(shippedEntries(ngram)).toEqual([
      'README.md',
      'compare.ts',
      'gramSize.ts',
      'inverted',
      'kernel.ts',
      'key.ts',
      'packing.ts',
      'profile.ts',
    ])
    expect(shippedEntries(join(ngram, 'inverted'))).toEqual([
      'builder.ts',
      'cosine.ts',
      'dice.ts',
      'keys.ts',
      'ordinals.ts',
      'overlap.ts',
      'query.ts',
      'tversky.ts',
    ])
  })

  it('keeps the n-gram semantics below the index built on them', () => {
    // The inverted index is an optional acceleration strategy over n-gram
    // semantics, never the foundation: a profile, a comparison or a prepared
    // kernel that reached into `inverted/` would invert that.
    const ngram = join(source, 'algorithms/ngram')
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
    const ngram = join(source, 'algorithms/ngram')
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
    const ngram = join(source, 'algorithms/ngram')
    expect(sourceImports(join(ngram, 'key.ts'))).toEqual([])
    expect(sourceImports(join(ngram, 'gramSize.ts'))).toEqual([])
  })

  // The layout carries the scorer graph: a public scorer module is named after
  // the method it exposes, and `token/` is the one genuine family — six public
  // strategies over a token engine they share. Pinning both listings is what
  // stops a seventh scorer landing loose at the root.
  it('keeps fuzz families physically and directionally isolated', () => {
    const directory = join(source, 'fuzz')
    const family = join(directory, 'token')
    expect(shippedEntries(directory)).toEqual([
      'index.ts',
      'metric.ts',
      'partialRatio.ts',
      'partialWindow.ts',
      'preparation.ts',
      'ratio.ts',
      'token',
      'types.ts',
      'weightedRatio.ts',
    ])
    expect(shippedEntries(family)).toEqual([
      'containment.ts',
      'partialTokenRatio.ts',
      'partialTokenSetRatio.ts',
      'partialTokenSortRatio.ts',
      'tokenRatio.ts',
      'tokenSet.ts',
      'tokenSetRatio.ts',
      'tokenSort.ts',
      'tokenSortRatio.ts',
      'tokens.ts',
    ])
    for (const name of ['ratio.ts', 'partialRatio.ts']) {
      expect(readFileSync(join(directory, name), 'utf8')).not.toMatch(
        /from ['"][^'"]*(token|weightedRatio)/,
      )
    }
    for (const name of shippedEntries(family)) {
      expect(readFileSync(join(family, name), 'utf8')).not.toMatch(
        /from ['"][^'"]*weighted/,
      )
    }
    // The two edges that hold the layers apart, and the ones a later edit is
    // most likely to reverse. partialWindow is the primitive both similarity
    // and every token strategy bottom out in, so it must stay usable without
    // loading the token engine; preparation sits above all of them and reaches
    // the reusable cores, never a module that declares a public metric.
    expect(readFileSync(join(directory, 'partialWindow.ts'), 'utf8')).not.toMatch(
      /from ['"][^'"]*token/,
    )
    expect(readFileSync(join(directory, 'preparation.ts'), 'utf8')).not.toMatch(
      /from ['"][^'"]*[Rr]atio\.js/,
    )
  })
})

describe('removed architecture', () => {
  it.each([
    '_common.ts',
    'algorithms/shared',
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
    // Cosine, Dice, Jaro, Jaro-Winkler and Tversky are normalized by
    // construction, so their normalized exports are the same metrics. `typeof`
    // is what says so, and an alias that named itself instead would appear
    // below as an id.
    const aliases = [
      ...declarations.matchAll(/export const (\w+): typeof (\w+) = (\w+)/g),
    ]
    expect(aliases).toHaveLength(10)
    for (const [, name, annotation, initializer] of aliases) {
      expect(annotation).toBe(initializer)
      expect(name).not.toBe(initializer)
    }
    const ids = [...declarations.matchAll(/BuiltInMetric<\s*'([\w.]+)'/g)].map(
      (match) => match[1],
    )
    for (const family of ['cosine', 'dice', 'jaro', 'jaroWinkler', 'tversky']) {
      expect(ids).not.toContain(`${family}.normalizedDistance`)
      expect(ids).not.toContain(`${family}.normalizedSimilarity`)
    }
  })

  it('leaves the mechanics of identity to the adapter', () => {
    expect(declarations).not.toMatch(/declare const \w+: unique symbol/)
  })
})
