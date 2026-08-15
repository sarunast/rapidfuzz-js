#!/usr/bin/env node
/**
 * Bundle the requested bench files with esbuild and run them in bare `node`.
 *
 * This is what keeps vite out of the measurement path. `vitest bench` executes
 * suites through vite's SSR transform, which rewrites every ESM import into
 * namespace property access — measured at roughly 2.5x on this suite's case
 * bodies, and asymmetrically across module layouts, so an A/B over a refactor
 * lied. Bundling once with esbuild and executing the output in a plain child
 * leaves plain statically compiled JS with no runtime transform. Not the
 * shipped `dist/` shape — tsdown emits that unbundled — but one stable
 * prebuilt artifact, built identically for every repeat, which is what
 * regression numbers need.
 *
 * Usage mirrors the old vitest spellings so the surrounding scripts kept
 * theirs:
 *
 *     node bench/harness/runner.ts                   # whole suite, full windows
 *     node bench/harness/runner.ts --quick fuzz      # quick windows, one file
 *     node bench/harness/runner.ts -t 'partialRatio' # name filter, any file
 *     node bench/harness/runner.ts --outputJson=out.json  # compare.ts's interface
 *
 * A file may be named by any substring of its path, as everywhere else in the
 * bench tooling. `--testNamePattern=<re>` is accepted as the long spelling of
 * `-t` because a pattern with a space in it must survive Windows' cmd.exe
 * re-splitting as one token.
 *
 * This file is hashed into every baseline entry (see SHARED_SOURCES in
 * `compare.ts`): the bundling options decide what is measured, so editing
 * them means the stored numbers no longer describe the same thing.
 */

import { spawnSync } from 'node:child_process'
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import { discoverSuiteFiles } from './discovery.ts'

const HARNESS_DIR = dirname(fileURLToPath(import.meta.url))
const BENCH_DIR = dirname(HARNESS_DIR)
const PROJECT_DIR = dirname(BENCH_DIR)

/**
 * The generated entry for ONE bench file: name it, import it, run what
 * registered.
 *
 * The dynamic import executes after `setCurrentFile`, which is what lets the
 * harness label the `describe` calls with the project-relative path the
 * baseline stores. esbuild resolves and bundles it statically all the same.
 */
function entryStub(file: string): string {
  return [
    `import { setCurrentFile, runRegisteredBenchmarks } from './bench/harness/harness.js'`,
    `setCurrentFile(${JSON.stringify(file)})`,
    `await import(${JSON.stringify(`./${file}`)})`,
    'runRegisteredBenchmarks()',
  ].join('\n')
}

/**
 * One bundle per bench file, because each file gets its own child process.
 *
 * Isolation is a measurement decision, not a convenience: with every file in
 * one process, the shared kernels' inline caches go megamorphic on the union
 * of every file's input shapes and the heap holds every file's corpus at
 * once. Measured on the first single-process recording, the fuzz cases ran
 * 1.05-1.54x slower than they had standalone and their repeat-to-repeat noise
 * went from under 3% to as much as 52% — the file that ran last paid the
 * most. A child per file is what the old vitest workers provided, and what
 * every stored baseline number assumes.
 *
 * Building is the default; reuse has to be asked for, with
 * `--reusePrepared`. That is what lets `compare.ts` hand every pass the same
 * `--bundleDir`: esbuild runs once under `--prepare`, before the first control
 * is timed, so the CPU work of bundling cannot heat the machine inside a
 * measured pass — and repeat 1 and repeat 2 execute literally the same bytes.
 *
 * Reuse is a claim that the bundle matches the source, and an existing file
 * proves nothing about its age. So only the caller that just prepared the
 * directory may make that claim, within the one run it prepared it for.
 * Anything else — including a second plain invocation against a `--bundleDir`
 * left lying around from an earlier one — rebuilds, because an edit in between
 * would otherwise be measured as the old code and look entirely valid doing it.
 */
async function bundle(bundleDir: string, file: string, reuse: boolean): Promise<string> {
  const outfile = join(bundleDir, `${basename(file, '.bench.ts')}.mjs`)
  if (reuse) {
    // Reuse or fail, never "reuse or quietly build instead": falling back
    // would run esbuild inside a measured pass, which is the one thing
    // preparing the directory exists to prevent.
    if (!existsSync(outfile)) {
      throw new Error(`prepared bundle is missing: ${outfile}`)
    }
    return outfile
  }
  // Imported here rather than at the top so the resolvability check in
  // `main` owns the failure message.
  const { build } = await import('esbuild')
  mkdirSync(bundleDir, { recursive: true })
  await build({
    stdin: {
      contents: entryStub(file),
      resolveDir: PROJECT_DIR,
      sourcefile: 'bench-entry.mjs',
      loader: 'js',
    },
    outfile,
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    // Not minified: the point is a stable, readable shape for V8, not size.
    // Sourcemaps off for the same reason — nothing debugs the bundle, and the
    // map would double what esbuild writes per run.
    minify: false,
    sourcemap: false,
    logLevel: 'silent',
  })
  return outfile
}

function usage(): void {
  process.stdout.write(`
  node bench/harness/runner.ts [options] [file …]

  Bundle the bench suite with esbuild and measure it in bare node.

    --quick                      short windows; not baseline-comparable
    --confirm                    widened windows, for re-measuring a flagged case
    -t, --testNamePattern=<re>   only cases whose 'group > name' matches
    --outputJson=<path>          write the report JSON compare.ts reads
    --bundleDir=<path>           where the bundles go (compare.ts)
    --prepare                    build the bundles into --bundleDir, measure nothing
    --reusePrepared              trust the bundles already in --bundleDir
    --reverse                    run files and cases in reverse order
    --progress                   name each case on stderr as it starts
    -h, --help                   this
`)
}

interface Arguments {
  readonly files: string[]
  /** `-t` regexp source, or null for no filter. */
  readonly name: string | null
  readonly outputJson: string | null
  readonly bundleDir: string | null
  readonly quick: boolean
  readonly confirm: boolean
  readonly prepare: boolean
  readonly reusePrepared: boolean
  readonly reverse: boolean
  readonly progress: boolean
  readonly help: boolean
}

function parseArguments(): Arguments {
  const argv = process.argv.slice(2)
  const files: string[] = []
  let name: string | null = null
  let outputJson: string | null = null
  let bundleDir: string | null = null
  let quick = false
  let confirm = false
  let prepare = false
  let reusePrepared = false
  let reverse = false
  let progress = false
  let help = false
  for (let i = 0; i < argv.length; i++) {
    const argument = argv[i]
    if (argument === '-h' || argument === '--help') help = true
    else if (argument === '--quick') quick = true
    else if (argument === '--confirm') confirm = true
    else if (argument === '--prepare') prepare = true
    else if (argument === '--reusePrepared') reusePrepared = true
    else if (argument === '--reverse') reverse = true
    else if (argument === '--progress') progress = true
    else if (argument === '-t' || argument === '--testNamePattern') {
      i++
      if (i === argv.length) throw new Error(`${argument} needs a pattern`)
      name = argv[i]
    } else if (argument.startsWith('--testNamePattern=')) {
      name = argument.slice('--testNamePattern='.length)
    } else if (argument.startsWith('--outputJson=')) {
      // Absolute from the start: the measuring child runs with its own cwd,
      // and a relative path read back here and written there would name two
      // different files the moment this script is invoked from elsewhere.
      outputJson = resolve(argument.slice('--outputJson='.length))
    } else if (argument.startsWith('--bundleDir=')) {
      bundleDir = resolve(argument.slice('--bundleDir='.length))
    } else if (argument.startsWith('-')) {
      throw new Error(`unknown option ${argument}`)
    } else files.push(argument)
  }
  if (quick && confirm) throw new Error('--quick and --confirm are opposites')
  // Preparing into a directory nobody named builds into a temp directory this
  // process then deletes on the way out — the one spelling of `--prepare`
  // that cannot do what the word means.
  if (prepare && bundleDir === null) {
    throw new Error('--prepare needs a --bundleDir= to prepare into')
  }
  if (reusePrepared && bundleDir === null) {
    throw new Error('--reusePrepared needs the --bundleDir= that was prepared')
  }
  if (prepare && reusePrepared) {
    throw new Error('--prepare builds the bundles; --reusePrepared trusts them')
  }
  return {
    files,
    name,
    outputJson,
    bundleDir,
    quick,
    confirm,
    prepare,
    reusePrepared,
    reverse,
    progress,
    help,
  }
}

/** As much of the harness's report shape as merging the parts needs. */
interface ReportFile {
  readonly groups: readonly unknown[]
}

async function main(): Promise<number> {
  const options = parseArguments()
  if (options.help) {
    usage()
    return 0
  }
  // After `--help`, so the usage text is readable on a checkout that has not
  // installed anything, and before anything else, so the failure names the fix.
  try {
    createRequire(import.meta.url).resolve('esbuild')
  } catch {
    throw new Error('esbuild is not installed — run `pnpm install` first')
  }
  if (options.name !== null) {
    try {
      new RegExp(options.name)
    } catch (error) {
      throw new Error(
        `-t is a regexp and ${JSON.stringify(options.name)} is not one: ` +
          `${error instanceof Error ? error.message : error}`,
      )
    }
  }

  const files = discoverSuiteFiles(options.files)

  // A shared `--bundleDir` belongs to the caller and is left in place — that
  // is the whole point of it. Without one, each invocation bundles into its
  // own fresh temp directory, so two benchmark commands running at once
  // cannot overwrite each other's bundles mid-measurement.
  const ownDir = options.bundleDir === null
  const bundleDir = options.bundleDir ?? mkdtempSync(join(tmpdir(), 'rapidfuzz-bench-'))
  try {
    for (const file of files) await bundle(bundleDir, file, options.reusePrepared)
    if (options.prepare) return 0

    // Set *and* cleared, never merely inherited: a mode variable left over
    // from a parent shell must not put a run in a mode nobody asked for.
    const env: NodeJS.ProcessEnv = { ...process.env }
    if (options.quick) env['BENCH_QUICK'] = '1'
    else delete env['BENCH_QUICK']
    if (options.confirm) env['BENCH_CONFIRM'] = '1'
    else delete env['BENCH_CONFIRM']
    if (options.name !== null) env['BENCH_FILTER'] = options.name
    else delete env['BENCH_FILTER']
    if (options.reverse) env['BENCH_REVERSE'] = '1'
    else delete env['BENCH_REVERSE']
    if (options.progress) env['BENCH_PROGRESS'] = '1'
    else delete env['BENCH_PROGRESS']
    delete env['BENCH_OUTPUT']

    const ordered = options.reverse ? [...files].reverse() : files
    const reports: ReportFile[] = []
    for (const [index, file] of ordered.entries()) {
      const outfile = join(bundleDir, `${basename(file, '.bench.ts')}.mjs`)
      const childOutput =
        options.outputJson === null ? null : `${options.outputJson}.${index}.part`
      try {
        const result = spawnSync(process.execPath, ['--expose-gc', outfile], {
          cwd: PROJECT_DIR,
          stdio: 'inherit',
          env: childOutput === null ? env : { ...env, BENCH_OUTPUT: childOutput },
        })
        if (result.error !== undefined) throw result.error
        // A child killed by a signal reports `status: null`; treating that as
        // success would make a Ctrl-C look like a clean run.
        if (result.status !== 0) return result.status ?? 1
        if (childOutput !== null) {
          const part: { files: ReportFile[] } = JSON.parse(
            readFileSync(childOutput, 'utf8'),
          )
          reports.push(...part.files)
        }
      } finally {
        // Including on the failure paths above: a half-written part file left
        // behind is one a later run with the same `--outputJson` would find
        // and merge.
        if (childOutput !== null) rmSync(childOutput, { force: true })
      }
    }
    if (options.outputJson !== null) {
      writeFileSync(options.outputJson, JSON.stringify({ files: reports }))
    }
    return 0
  } finally {
    if (ownDir) rmSync(bundleDir, { recursive: true, force: true })
  }
}

main().then(
  (code) => {
    process.exitCode = code
  },
  (error: unknown) => {
    process.stderr.write(`\n  ! ${error instanceof Error ? error.message : error}\n\n`)
    process.exitCode = 1
  },
)
