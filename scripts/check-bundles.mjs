// @ts-check
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import { gzipSync } from 'node:zlib'

const root = resolve(import.meta.dirname, '..')
const fixtures = resolve(import.meta.dirname, 'bundle-fixtures')
const output = mkdtempSync(join(tmpdir(), 'rapidfuzz-bundles-'))

/** Accepted deterministic gzip sizes plus 2% headroom. */
const budgets = {
  'direct-fuzz.js': 5_987,
  'direct-token-sort.js': 6_097,
  'compiled-fuzz.js': 6_747,
  'compiled-token-sort.js': 6_847,
  'direct-normalized-edit.js': 9_880,
  'compiled-normalized-edit.js': 10_613,
  'direct-raw-distance.js': 9_892,
  'compiled-raw-distance.js': 10_624,
  'one-shot.js': 7_506,
  'one-shot-iter.js': 7_433,
  'matcher-token-sort.js': 8_347,
  'score-matrix.js': 11_465,
  'score-pairs.js': 11_283,
  'full-fuzz.js': 10_138,
  'all-subpaths.js': 26_371,
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
