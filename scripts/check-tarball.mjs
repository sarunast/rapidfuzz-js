// @ts-check
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join, resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const output = mkdtempSync(join(tmpdir(), 'rapidfuzz-pack-'))

/** @typedef {{ path: string }} PackFile */
/**
 * @typedef {{
 *   filename: string,
 *   size: number,
 *   unpackedSize: number,
 *   entryCount: number,
 *   files: PackFile[],
 * }} PackResult
 */

try {
  const raw = execFileSync(
    'npm',
    ['pack', '--json', '--ignore-scripts', '--pack-destination', output],
    {
      cwd: root,
      encoding: 'utf8',
      env: { ...process.env, npm_config_cache: join(output, 'npm-cache') },
    },
  )
  /** @type {PackResult[]} */
  const parsed = JSON.parse(raw)
  const packed = parsed[0]
  if (packed === undefined) throw new Error('npm pack returned no package')

  const files = packed.files ?? []
  if (packed.size > 464_000) {
    throw new Error(`packed size ${packed.size} exceeds the 464 KB limit`)
  }
  const source = files.find((file) => /^src\/.+\.ts$/.test(file.path))
  if (source !== undefined)
    throw new Error(`TypeScript source was packed: ${source.path}`)

  const maps = files.filter((file) => file.path.endsWith('.js.map'))
  if (maps.length === 0) throw new Error('no JavaScript source maps were packed')
  const tarball = join(output, basename(packed.filename))
  for (const map of maps) {
    const text = execFileSync('tar', ['-xOf', tarball, `package/${map.path}`], {
      encoding: 'utf8',
    })
    const payload = JSON.parse(text)
    if (!Array.isArray(payload.sourcesContent) || payload.sourcesContent.length === 0) {
      throw new Error(`${map.path} does not retain sourcesContent`)
    }
  }

  console.log(`✓ packed size: ${packed.size} bytes`)
  console.log(`✓ unpacked size: ${packed.unpackedSize} bytes`)
  console.log(`✓ entry count: ${packed.entryCount}`)
  console.log(`✓ ${maps.length} source maps retain sourcesContent`)
  console.log('✓ no src/**/*.ts files are packed')
} finally {
  rmSync(output, { recursive: true, force: true })
}
