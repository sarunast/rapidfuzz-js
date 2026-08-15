/**
 * Which files are suites, and what each one is called.
 *
 * One owner for both readers. The runner measures what this returns and the
 * comparator judges it, and when they each had their own copy the two could
 * disagree about ordering, about which spelling of a path a filter matched, or
 * about whether the anchor is a subject — none of which would fail anything, it
 * would just compare a run against a baseline built from a different set.
 *
 * Deliberately flat: `bench/suites/*.bench.ts`, never `**`. `runner.ts` names a
 * bundle from the basename alone, so two suites of the same name in different
 * subdirectories would silently overwrite one bundle and measure it twice.
 *
 * This file is hashed into every baseline entry (see `SHARED_SOURCES`): what it
 * returns decides which definitions a stored number was produced from.
 */

import { readdirSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HARNESS_DIR = dirname(fileURLToPath(import.meta.url))
const BENCH_DIR = dirname(HARNESS_DIR)
const SUITES_DIR = join(BENCH_DIR, 'suites')
const PROJECT_DIR = dirname(BENCH_DIR)

/** The file whose cases anchor every other case. See its header. */
export const CONTROL_FILE = 'bench/suites/control.bench.ts'

/**
 * A path argument in the one spelling everything downstream compares against.
 *
 * Stored case names begin with the path the runner reports —
 * `bench/suites/fuzz.bench.ts`, project-relative and slash-separated — and that
 * string is the identity behind three separate things: which baseline entries a
 * partial `--record` replaces, which ones `candidates` considers, and which file
 * gets fingerprinted. Left raw, `./bench/suites/fuzz.bench.ts` and an absolute
 * path each name the same file and match none of them, so a re-record would
 * quietly keep the stale entries it was run to remove. Normalising here means
 * there is only ever one namespace.
 */
export function canonicalFile(arg: string): string {
  // Resolved against the project, not the process's cwd: a bench path has to
  // name the same file whichever directory the command was typed in.
  return relative(PROJECT_DIR, resolve(PROJECT_DIR, arg)).split(sep).join('/')
}

/** Every suite file, project-relative with forward slashes. */
function allSuiteFiles(): string[] {
  return (
    readdirSync(SUITES_DIR)
      .filter((name) => name.endsWith('.bench.ts'))
      // Interpolated, not `join`ed. What is being built here is an identity —
      // the same string the runner puts in a case's name and `baseline.json`
      // stores — and identities do not have a platform. `join` would spell it
      // `bench\suites\fuzz.bench.ts` on Windows, where it would match neither
      // {@link CONTROL_FILE}, so the anchor would be measured as a subject, nor
      // anything {@link canonicalFile} produced, so a filter naming a real file
      // would be rejected as matching none.
      .map((name) => `bench/suites/${name}`)
      .sort()
  )
}

/**
 * The suite files a run should cover.
 *
 * A filter is a substring of a path, in either the spelling that was typed or
 * the canonical one — `fuzz`, `bench/suites/fuzz.bench.ts` and an absolute path
 * all select the same file.
 *
 * @param excludeControl - drop {@link CONTROL_FILE}. The comparator times the
 * controls on their own either side of a pass rather than inside it, so they
 * are never among its subjects; the runner measures whatever it is given.
 */
export function discoverSuiteFiles(
  filters: readonly string[],
  { excludeControl = false }: { excludeControl?: boolean } = {},
): string[] {
  const all = excludeControl
    ? allSuiteFiles().filter((file) => file !== CONTROL_FILE)
    : allSuiteFiles()

  if (all.length === 0) {
    throw new Error(`nothing to measure: ${CONTROL_FILE} is the anchor, not a subject`)
  }
  if (filters.length === 0) return all

  const chosen = all.filter((file) =>
    filters.some(
      (filter) => file.includes(canonicalFile(filter)) || file.includes(filter),
    ),
  )
  // Reported here rather than as an empty run: a filter matching no file would
  // otherwise surface minutes later as a report with no cases in it.
  if (chosen.length === 0) {
    throw new Error(
      `no benchmark file matched ${filters.join(', ')}. Known files: ${all.join(', ')}`,
    )
  }
  return chosen
}
