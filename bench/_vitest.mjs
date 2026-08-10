// @ts-check
/**
 * Where the benchmark runner lives, for the two scripts that spawn it.
 *
 * `compare.mjs` and `run.mjs` both need the same answer and neither can ask
 * `pnpm` for it: they set environment variables around the child, so they run
 * the binary rather than a package script.
 *
 * Deliberately not a `.ts` file under `bench/`. Everything the fingerprint in
 * `compare.mjs` covers has to be hashed before a run can be compared; these two
 * scripts sit outside that, because how the child was spawned is already
 * recorded in each case's environment stamp.
 */

import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

export const PROJECT_DIR = dirname(dirname(fileURLToPath(import.meta.url)))

/** `--expose-gc` is what lets `_harness.ts` collect between cases. */
export const EXPOSE_GC = '--expose-gc'

/** The environment variable `_harness.ts` reads to shorten every window. */
export const QUICK_ENV = 'BENCH_QUICK'

/** @returns {string} */
export function vitestBin() {
  const name = process.platform === 'win32' ? 'vitest.cmd' : 'vitest'
  return join(PROJECT_DIR, 'node_modules', '.bin', name)
}

/**
 * A `.cmd` shim is not an executable image on Windows, so `spawn` there has to
 * go through the shell. Everywhere else it must not, or the arguments would be
 * re-split by it — a `-t` pattern with a space in it is ordinary.
 */
export const needsShell = process.platform === 'win32'

/** @returns {void} */
export function assertVitestInstalled() {
  if (existsSync(vitestBin())) return
  throw new Error(`no vitest at ${vitestBin()} — run \`pnpm install\` first`)
}
