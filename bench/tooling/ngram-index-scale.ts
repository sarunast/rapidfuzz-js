/**
 * Entry point for the inverted n-gram index measurements.
 *
 * ```sh
 * node --expose-gc bench/tooling/ngram-index-scale.ts --parity-only
 * node --expose-gc bench/tooling/ngram-index-scale.ts --max=100000
 * ```
 *
 * All this does is bundle `ngram-index-report.ts` and import it. That indirection
 * is not a style choice: `src/` imports carry `.js` specifiers only a bundler
 * resolves back to `.ts`, so a script that reaches into `src/` cannot be run by
 * node directly — which is why `matcher-memory.ts` reads `dist/` instead. This
 * one needs `src/` internals *and* the bench-only prototype, so it does what
 * `runner.ts` does and bundles with esbuild first. The imports have to live in a
 * separate module because node resolves a module's static imports before it runs
 * a line of it.
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const directory = mkdtempSync(join(tmpdir(), 'ngram-index-scale-'))
const outfile = join(directory, 'report.mjs')

try {
  const { build } = await import('esbuild')
  await build({
    entryPoints: [join(here, 'ngram-index-report.ts')],
    outfile,
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    minify: false,
    sourcemap: false,
    logLevel: 'silent',
  })
  await import(pathToFileURL(outfile).href)
} finally {
  rmSync(directory, { recursive: true, force: true })
}
