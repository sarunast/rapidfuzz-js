/**
 * What `baseline.json` holds, and how a stored case is identified.
 *
 * Separate from `compare.ts` because this half is a library and that half is a
 * command: `compare.ts` runs `main()` at module load, so anything that imports
 * it measures the suite as a side effect. Everything here can be imported by a
 * tool that only wants to read or rewrite the stored file.
 */

import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const REGRESSION_DIR = dirname(fileURLToPath(import.meta.url))
const BENCH_DIR = dirname(REGRESSION_DIR)
const PROJECT_DIR = dirname(BENCH_DIR)

/** Everything outside the code under test that changes what a number means. */
export interface Environment {
  node: string
  esbuild: string
  platform: string
  cpu: string
  nodeOptions: string
  /** The measurement method's version; see `MEASUREMENT_VERSION` in `compare.ts`. */
  measurement: number
}

/**
 * One entry in `baseline.json`, which is a file on disk that outlives any one
 * run — this interface is the only description of what is in it, and a field
 * quietly added or dropped here is a baseline a later version reads
 * differently.
 *
 * `normalised` is the case's median in milliseconds, corrected to its
 * session's average machine speed. `machine` is that session's yardstick: the
 * per-control median, in milliseconds, keyed by control name.
 */
export interface CaseRecord extends Environment {
  normalised: number
  noise: number
  median: number
  machine: Record<string, number>
  min: number
  samples: number
  batch: number
  /** The shortest timed sample any repeat took, in milliseconds. */
  sample: number
  repeats: number
  source: string
  controls: string
  recordedAt: string
}

/** The stored file: every case ever recorded, by name. */
export interface Baseline {
  cases: Record<string, CaseRecord>
}

/**
 * Everything a case's timing depends on besides `src` and its own bench file.
 *
 * `bench/harness/runner.ts` is in here because the bundling options are decided
 * there, and changing any of them changes what a stored number means
 * without touching a benchmark. `MEASUREMENT_VERSION` covers a deliberate
 * change of method; this covers forgetting to bump it.
 */
export const SHARED_SOURCES = [
  'bench/harness/corpus.ts',
  'bench/harness/harness.ts',
  'bench/harness/runner.ts',
]

/**
 * Identify the definition a case was measured under.
 *
 * A case is named, not numbered, which survives reordering — but a name says
 * nothing about the fixture behind it. Widening `128 chars` to 256 without
 * renaming it would otherwise report as a 2x regression. The fingerprint covers
 * the bench file and everything it shares, so any such edit is visible as an
 * edit rather than as a result.
 */
const fingerprints = new Map<string, string>()
export function fingerprint(benchFile: string): string {
  const cached = fingerprints.get(benchFile)
  if (cached !== undefined) return cached

  const hash = createHash('sha256')
  for (const name of [benchFile, ...SHARED_SOURCES]) {
    const path = join(PROJECT_DIR, name)
    // Treating a missing file as empty content would fingerprint every case
    // by the shared helpers alone, and then editing a fixture would no longer
    // invalidate anything. If the path a case's name implies is not there,
    // the assumption behind these names has broken and the fingerprint is
    // worth nothing.
    if (!existsSync(path)) {
      throw new Error(`cannot fingerprint benchmark source: ${path}`)
    }
    hash.update(readFileSync(path))
  }
  const digest = hash.digest('hex').slice(0, 12)
  fingerprints.set(benchFile, digest)
  return digest
}

/** The bench file a stored case came from — the first segment of its name. */
export function fileOf(name: string): string {
  return name.slice(0, name.indexOf(' > '))
}
