import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

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

describe('dependency direction', () => {
  it('keeps core algorithm-blind', () => {
    expect(contents('core')).not.toMatch(/from ['"][^'"]*(algorithms|fuzz|search|batch)/)
  })

  it('keeps search and batch independent from algorithms', () => {
    expect(contents('search')).not.toMatch(/from ['"][^'"]*(algorithms|fuzz)/)
    expect(contents('batch')).not.toMatch(/from ['"][^'"]*(algorithms|fuzz|search)/)
  })

  it('keeps the root entrypoint orchestration-only', () => {
    const root = readFileSync(join(source, 'index.ts'), 'utf8')
    expect(root).not.toMatch(/from ['"][^'"]*(algorithms|fuzz)/)
  })
})

describe('removed architecture', () => {
  it.each(['_common.ts', 'configure.ts', 'match.ts', 'search.ts', 'distance/index.ts'])(
    'does not restore src/%s',
    (path) => {
      expect(existsSync(join(source, path))).toBe(false)
    },
  )

  it('does not restore the old distance tree', () => {
    expect(typeScriptFiles(join(source, 'distance'))).toEqual([])
  })
})
