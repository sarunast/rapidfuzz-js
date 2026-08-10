#!/usr/bin/env node
// @ts-check
/**
 * `pnpm bench:quick` — vitest's own reporter, tenth-length windows.
 *
 * The edit loop's benchmark. Everything after the script name is handed
 * straight to `vitest bench`, so the two filters compose the way they do
 * anywhere else:
 *
 *     pnpm bench:quick                             # the whole suite, ~30s
 *     pnpm bench:quick bench/fuzz.bench.ts         # one file
 *     pnpm bench:quick -t 'partialRatio'           # one group, any file
 *     pnpm bench:quick bench/fuzz.bench.ts -t '128 chars'
 *
 * ## Why a script rather than a line in `package.json`
 *
 * All this does is set {@link QUICK_ENV} for the child, which `_harness.ts`
 * reads. `BENCH_QUICK=1 vitest bench --run` would do the same on a POSIX shell
 * and is not a command at all on Windows — and `compare.mjs` carries `win32`
 * handling throughout, so the suite is meant to run there.
 *
 * ## Why not a flag in `_harness.ts`
 *
 * Because `_harness.ts` is one of the files `compare.mjs` hashes into every
 * case's fingerprint. Editing it to add a nicer spelling of a variable it
 * already reads would mark all of `baseline.json` as measured against a
 * definition that no longer exists, and cost a full re-record to undo.
 *
 * ## What these numbers are not
 *
 * Comparable to anything. Short windows can measure a case before V8 has
 * settled on a tier, and without `--expose-gc` one case's garbage is collected
 * on the next one's time. This answers "is it roughly where I left it" while
 * you are editing. `pnpm bench:compare` is what a claim rests on.
 */

import { spawnSync } from 'node:child_process'
import process from 'node:process'

import {
  assertVitestInstalled,
  needsShell,
  PROJECT_DIR,
  QUICK_ENV,
  vitestBin,
} from './_vitest.mjs'

try {
  assertVitestInstalled()
  const result = spawnSync(vitestBin(), ['bench', '--run', ...process.argv.slice(2)], {
    cwd: PROJECT_DIR,
    stdio: 'inherit',
    shell: needsShell,
    env: { ...process.env, [QUICK_ENV]: '1' },
  })
  if (result.error !== undefined) throw result.error
  // A child killed by a signal reports `status: null`; treating that as success
  // would make a Ctrl-C look like a clean run.
  process.exitCode = result.status ?? 1
} catch (error) {
  process.stderr.write(`\n  ! ${error instanceof Error ? error.message : error}\n\n`)
  process.exitCode = 1
}
