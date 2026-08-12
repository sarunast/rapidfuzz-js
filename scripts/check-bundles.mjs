// @ts-check
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import { gzipSync } from 'node:zlib'

const root = resolve(import.meta.dirname, '..')
const fixtures = resolve(import.meta.dirname, 'bundle-fixtures')
const output = mkdtempSync(join(tmpdir(), 'rapidfuzz-bundles-'))

/**
 * Accepted deterministic gzip sizes plus 2% headroom.
 *
 * Re-record all of them together, never one: the headroom is what absorbs the
 * few dozen bytes a different toolchain gzips differently, and topping up only
 * the entry that failed leaves its siblings a byte from the same surprise.
 */
const budgets = {
  'direct-fuzz.js': 6_052,
  'direct-token-sort.js': 6_124,
  'compiled-fuzz.js': 6_913,
  'compiled-token-sort.js': 6_964,
  'direct-normalized-edit.js': 9_934,
  'compiled-normalized-edit.js': 10_797,
  'direct-raw-distance.js': 9_945,
  'compiled-raw-distance.js': 10_801,
  'one-shot.js': 7_936,
  'one-shot-iter.js': 7_969,
  'prepared-one-shot.js': 7_999,
  'matcher-token-sort.js': 8_740,
  'score-matrix.js': 11_836,
  'score-pairs.js': 11_592,
  'full-fuzz.js': 10_223,
  'all-subpaths.js': 26_969,
}

try {
  const inputs = readdirSync(fixtures)
    .filter((name) => name.endsWith('.ts'))
    .map((name) => join(fixtures, name))
  for (const input of inputs) {
    const name = basename(input, '.ts')
    execFileSync(
      resolve(root, 'node_modules/.bin/tsdown'),
      [
        input,
        '--no-config',
        '--no-dts',
        '--minify',
        '--platform',
        'browser',
        '--deps.always-bundle',
        'rapidfuzz-js',
        '--target',
        'es2022',
        '--out-dir',
        join(output, name),
        '--logLevel',
        'silent',
      ],
      { cwd: root, stdio: 'inherit' },
    )
  }

  let failures = 0
  for (const [file, budget] of Object.entries(budgets)) {
    const name = basename(file, '.js')
    const bytes = gzipSync(readFileSync(join(output, name, file))).byteLength
    const marker = bytes <= budget ? '✓' : '✗'
    console.log(`${marker} ${file}: ${bytes} B gzip (budget ${budget} B)`)
    if (bytes > budget) failures++
  }
  if (failures > 0) process.exitCode = 1
} finally {
  rmSync(output, { recursive: true, force: true })
}
