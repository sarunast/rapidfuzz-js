import { mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import { writeHeapSnapshot } from 'node:v8'

import { similarity } from '../../dist/algorithms/dice/index.js'
import { createIndexedMatcher, createScorer } from '../../dist/index.js'
import {
  collectAndSample,
  assertCliOptions,
  median,
  numberOption,
  parseCli,
  positive,
  recoveryDelta,
  retainedTheilSen,
  runIsolated,
  writeJsonArtifact,
  writeStructured,
  type MemorySample,
} from './harness.ts'
import {
  SOAK_CALIBRATION,
  SOAK_SCENARIOS,
  SOAK_THRESHOLDS,
  type SoakScenario,
} from './thresholds.ts'
import { runLateInvalidQuery } from './workloads/exceptionRecovery.ts'
import { runQueryProfileSpike } from './workloads/queryProfile.ts'
import {
  lowercaseBigramCorpus,
  ordinaryQuery,
  runOrdinaryBest,
} from './workloads/shared.ts'
import { runSteadyBatch, STEADY_BATCH_SIZE } from './workloads/steady.ts'
import { runTouchedSetSpike, validateTouchedCorpus } from './workloads/touchedSet.ts'

const DEFAULT_CHOICES = 100_000
const CALIBRATION_RUNS = 20
const RECOVERY_BATCHES = 10
const PLATEAU_SAMPLES = 5
interface RunResult {
  readonly scenario: SoakScenario
  readonly choices: number
  readonly batchOperations: number
  readonly baseline: readonly MemorySample[]
  readonly recovery: readonly MemorySample[]
  readonly samples: readonly MemorySample[]
  readonly slopeBytesPerBatch: number
  readonly recoveryBytes: number
  readonly thresholds: (typeof SOAK_THRESHOLDS)[SoakScenario]
  readonly evaluated: boolean
  readonly evaluationReason: string | null
  readonly passed: boolean | null
  readonly touchedValidation?: ReturnType<typeof validateTouchedCorpus> | undefined
  readonly fixture?: 'slope' | 'recovery' | undefined
}

function isScenario(value: string): value is SoakScenario {
  return SOAK_SCENARIOS.some((scenario) => scenario === value)
}

function scenarioOf(value: string | boolean | undefined): readonly SoakScenario[] {
  if (value === undefined || value === 'all') return SOAK_SCENARIOS
  if (
    typeof value === 'string' &&
    Object.hasOwn(SOAK_THRESHOLDS, value) &&
    isScenario(value)
  ) {
    return [value]
  }
  throw new RangeError(`--scenario must be one of all, ${SOAK_SCENARIOS.join(', ')}`)
}

function evaluationReason(
  choices: number,
  batchOperations: number,
  snapshots: string | undefined,
  diagnostic: boolean,
): string | null {
  if (diagnostic) return 'diagnostic or profiler-instrumented run'
  if (snapshots !== undefined) return 'heap snapshots were enabled'
  if (choices !== DEFAULT_CHOICES || batchOperations !== STEADY_BATCH_SIZE) {
    return 'workload size differs from the committed calibration'
  }
  const nodeMajor = Number.parseInt(process.versions.node, 10)
  if (
    nodeMajor !== SOAK_CALIBRATION.nodeMajor ||
    process.platform !== SOAK_CALIBRATION.platform ||
    process.arch !== SOAK_CALIBRATION.arch
  ) {
    return `environment differs from Node ${SOAK_CALIBRATION.nodeMajor} ${SOAK_CALIBRATION.platform}/${SOAK_CALIBRATION.arch} calibration`
  }
  return null
}

function snapshotDirectory(value: string | boolean | undefined): string | undefined {
  if (value === undefined) return undefined
  if (typeof value === 'string') return resolve(value)
  const timestamp = new Date()
    .toISOString()
    .replaceAll(/[-:]/g, '')
    .replace(/\.\d{3}Z$/, 'Z')
  return resolve('bench/memory/artifacts', `${timestamp}-${process.pid}`)
}

function capture(
  directory: string | undefined,
  scenario: string,
  boundary: string,
): void {
  if (directory === undefined) return
  const scenarioDirectory = join(directory, scenario)
  mkdirSync(scenarioDirectory, { recursive: true })
  writeHeapSnapshot(join(scenarioDirectory, `${boundary}.heapsnapshot`))
}

function runOrdinaryBatch(
  matcher: import('./workloads/shared.ts').IndexedMatcherWorkload,
  batch: number,
  operations: number,
): void {
  const first = batch * operations
  for (let offset = 0; offset < operations; offset++) {
    runOrdinaryBest(matcher, first + offset)
  }
}

function retainedSamples(
  count: number,
  operation: (index: number) => void,
  fixtureSlopeThreshold = 0,
  operationsPerBatch = 0,
): MemorySample[] {
  const samples: MemorySample[] = []
  for (let index = 0; index < count; index++) {
    operation(index)
    if (fixtureSlopeThreshold > 0) {
      const buffers = Math.max(1, Math.floor(operationsPerBatch / 100))
      const bytes = Math.ceil((2 * fixtureSlopeThreshold) / buffers)
      for (let retained = 0; retained < buffers; retained++) {
        retainedFixture.push(new ArrayBuffer(bytes))
      }
    }
    samples.push(collectAndSample())
  }
  return samples
}

const retainedFixture: ArrayBuffer[] = []

function executeScenario(
  scenario: SoakScenario,
  choices: number,
  batchOperations: number,
  snapshots: string | undefined,
  fixture: 'slope' | 'recovery' | undefined,
  diagnostic: boolean,
): RunResult {
  const corpus = lowercaseBigramCorpus(choices)
  const scorer = createScorer(similarity, { gramSize: 2 })
  const matcher = createIndexedMatcher(corpus, { scorer })
  let touchedValidation: ReturnType<typeof validateTouchedCorpus> | undefined
  if (scenario === 'touched-set') touchedValidation = validateTouchedCorpus(corpus)

  // Warm the same matcher and module graph that is subsequently measured.
  runOrdinaryBatch(matcher, 0, Math.min(batchOperations, 1_000))
  collectAndSample()
  if (scenario === 'steady') capture(snapshots, scenario, 'post-warmup')

  const threshold = SOAK_THRESHOLDS[scenario]
  const fixtureSlopeThreshold = fixture === 'slope' ? threshold.slopeBytesPerBatch : 0
  let baseline: MemorySample[]
  let recovery: MemorySample[]
  let samples: MemorySample[]

  if (scenario === 'steady') {
    samples = retainedSamples(
      20,
      (batch) => {
        runSteadyBatch(matcher, batch, batchOperations)
      },
      fixtureSlopeThreshold,
      batchOperations,
    )
    baseline = samples.slice(0, PLATEAU_SAMPLES)
    recovery = samples.slice(-PLATEAU_SAMPLES)
  } else {
    baseline = retainedSamples(PLATEAU_SAMPLES, (batch) =>
      runOrdinaryBatch(matcher, batch + 1, batchOperations),
    )
    capture(snapshots, scenario, 'pre-spike')
    if (scenario === 'query-profile') runQueryProfileSpike(matcher)
    else if (scenario === 'touched-set') runTouchedSetSpike(matcher)
    else runLateInvalidQuery(matcher)
    capture(snapshots, scenario, 'post-spike')

    // Exception cleanup is allowed to wait until one later successful operation.
    if (scenario === 'exception') matcher.best(ordinaryQuery(90_000_000))
    if (fixture === 'recovery') {
      retainedFixture.push(new ArrayBuffer(2 * threshold.recoveryBytes))
    }
    recovery = retainedSamples(
      RECOVERY_BATCHES,
      (batch) => runOrdinaryBatch(matcher, batch + 10, batchOperations),
      fixtureSlopeThreshold,
      batchOperations,
    )
    samples = [...baseline, ...recovery]
  }

  capture(snapshots, scenario, scenario === 'steady' ? 'final' : 'recovered')
  const slopeBytesPerBatch = positive(retainedTheilSen(samples))
  const recoveryBytes = recoveryDelta(baseline, recovery.slice(-PLATEAU_SAMPLES))
  const reason = evaluationReason(choices, batchOperations, snapshots, diagnostic)
  const evaluated = reason === null
  return {
    scenario,
    choices,
    batchOperations,
    baseline,
    recovery,
    samples,
    slopeBytesPerBatch,
    recoveryBytes,
    thresholds: threshold,
    evaluated,
    evaluationReason: reason,
    passed: evaluated
      ? slopeBytesPerBatch <= threshold.slopeBytesPerBatch &&
        recoveryBytes <= threshold.recoveryBytes
      : null,
    touchedValidation,
    fixture,
  }
}

function percentile95(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[Math.ceil(sorted.length * 0.95) - 1]
}

function nextMultipleAbove(value: number, unit: number): number {
  return (Math.floor(value / unit) + 1) * unit
}

function calibratedThreshold(results: readonly RunResult[]) {
  const slopes = results.map((result) => positive(result.slopeBytesPerBatch))
  const recoveries = results.map((result) => positive(result.recoveryBytes))
  return {
    slopeBytesPerBatch: nextMultipleAbove(
      Math.max(8 << 10, 3 * percentile95(slopes)),
      4 << 10,
    ),
    recoveryBytes: nextMultipleAbove(
      Math.max(256 << 10, 3 * percentile95(recoveries)),
      64 << 10,
    ),
    observedSlopeRange: [Math.min(...slopes), Math.max(...slopes)],
    observedRecoveryRange: [Math.min(...recoveries), Math.max(...recoveries)],
  }
}

const options = parseCli(process.argv.slice(2))
assertCliOptions(options, [
  'scenario',
  'choices',
  'batch-operations',
  'snapshot',
  'fixture',
  'calibrate',
  'artifact',
  'child',
  'parent-child',
  'calibration-child',
  'diagnostic',
])
for (const name of [
  'calibrate',
  'child',
  'parent-child',
  'calibration-child',
  'diagnostic',
]) {
  const value = options.get(name)
  if (value !== undefined && value !== true) {
    throw new TypeError(`--${name} does not accept a value`)
  }
}
for (const name of ['scenario', 'fixture', 'artifact']) {
  const value = options.get(name)
  if (value === true) throw new TypeError(`--${name} requires a value`)
}
const scenarios = scenarioOf(options.get('scenario'))
const choices = numberOption(options, 'choices', DEFAULT_CHOICES)
const batchOperations = numberOption(options, 'batch-operations', STEADY_BATCH_SIZE)
const snapshots = snapshotDirectory(options.get('snapshot'))
const fixtureValue = options.get('fixture')
const fixture =
  fixtureValue === 'slope' || fixtureValue === 'recovery' ? fixtureValue : undefined
if (fixtureValue !== undefined && fixture === undefined) {
  throw new RangeError('--fixture must be slope or recovery')
}
if (options.has('calibrate') && fixture !== undefined) {
  throw new TypeError('--fixture cannot be combined with --calibrate')
}
if (options.has('calibrate') && snapshots !== undefined) {
  throw new TypeError('--snapshot cannot be combined with --calibrate')
}
if (options.has('calibrate') && options.has('diagnostic')) {
  throw new TypeError('--diagnostic cannot be combined with --calibrate')
}
if (fixture === 'recovery' && scenarios.includes('steady')) {
  throw new TypeError('the recovery fixture is not supported by the steady scenario')
}

if (options.has('child')) {
  if (scenarios.length !== 1) throw new Error('a child runs exactly one scenario')
  const diagnostic =
    options.has('diagnostic') ||
    process.execArgv.some((argument) => argument.startsWith('--heap-prof'))
  const result = executeScenario(
    scenarios[0],
    choices,
    batchOperations,
    snapshots,
    fixture,
    diagnostic,
  )
  writeStructured(result)
  if (
    result.passed === false &&
    !options.has('calibration-child') &&
    !options.has('parent-child')
  ) {
    process.exitCode = 1
  }
} else {
  const here = fileURLToPath(import.meta.url)
  const childArguments = (scenario: SoakScenario, activeFixture?: string): string[] => [
    `--scenario=${scenario}`,
    `--choices=${choices}`,
    `--batch-operations=${batchOperations}`,
    ...(snapshots === undefined ? [] : [`--snapshot=${snapshots}`]),
    ...(activeFixture === undefined ? [] : [`--fixture=${activeFixture}`]),
    ...(options.has('diagnostic') ? ['--diagnostic'] : []),
    '--parent-child',
  ]
  if (options.has('calibrate')) {
    const calibration = scenarios.map((scenario) => {
      const runs = Array.from({ length: CALIBRATION_RUNS }, () =>
        runIsolated<RunResult>(here, [
          ...childArguments(scenario),
          '--calibration-child',
        ]),
      )
      return { scenario, runs, selected: calibratedThreshold(runs) }
    })
    const report = {
      calibrationRuns: CALIBRATION_RUNS,
      node: process.version,
      platform: `${process.platform}/${process.arch}`,
      choices,
      batchOperations,
      calibration,
    }
    const artifact = options.get('artifact')
    if (typeof artifact === 'string') writeJsonArtifact(artifact, report)
    writeStructured({
      calibrationRuns: report.calibrationRuns,
      node: report.node,
      platform: report.platform,
      choices: report.choices,
      batchOperations: report.batchOperations,
      calibration: calibration.map(({ scenario, selected }) => ({ scenario, selected })),
    })
  } else {
    const results = scenarios.map((scenario) =>
      runIsolated<RunResult>(here, childArguments(scenario, fixture)),
    )
    const report = {
      node: process.version,
      platform: `${process.platform}/${process.arch}`,
      results,
      evaluated: results.every((result) => result.evaluated),
      passed: results.every((result) => result.evaluated)
        ? results.every((result) => result.passed === true)
        : null,
      medianSlopeBytesPerBatch: median(
        results.map((result) => result.slopeBytesPerBatch),
      ),
    }
    const artifact = options.get('artifact')
    if (typeof artifact === 'string') writeJsonArtifact(artifact, report)
    writeStructured({
      node: report.node,
      platform: report.platform,
      results: results.map(
        ({ baseline: _baseline, recovery: _recovery, samples: _samples, ...result }) =>
          result,
      ),
      evaluated: report.evaluated,
      passed: report.passed,
      medianSlopeBytesPerBatch: report.medianSlopeBytesPerBatch,
    })
    if (report.passed === false) process.exitCode = 1
  }
}
