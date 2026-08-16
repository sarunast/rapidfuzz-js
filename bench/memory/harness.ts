import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import process from 'node:process'

export interface MemorySample {
  readonly heapUsed: number
  readonly heapTotal: number
  readonly arrayBuffers: number
  readonly external: number
  readonly rss: number
  readonly retained: number
}

export type CliValue = string | boolean

export function collectGarbage(): void {
  if (globalThis.gc === undefined) throw new Error('this process requires --expose-gc')
  globalThis.gc()
  globalThis.gc()
  globalThis.gc()
}

export function sampleMemory(): MemorySample {
  const usage = process.memoryUsage()
  return {
    heapUsed: usage.heapUsed,
    heapTotal: usage.heapTotal,
    arrayBuffers: usage.arrayBuffers,
    external: usage.external,
    rss: usage.rss,
    retained: usage.heapUsed + usage.arrayBuffers,
  }
}

export function collectAndSample(): MemorySample {
  collectGarbage()
  return sampleMemory()
}

export function median(values: readonly number[]): number {
  if (values.length === 0) throw new RangeError('median needs at least one value')
  const sorted = [...values].sort((left, right) => left - right)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle]
}

/** Median of every pairwise retained-byte slope over the final ten samples. */
export function retainedTheilSen(samples: readonly MemorySample[]): number {
  const tail = samples.slice(-10)
  if (tail.length < 2) throw new RangeError('Theil–Sen needs at least two samples')
  const slopes: number[] = []
  for (let left = 0; left < tail.length - 1; left++) {
    for (let right = left + 1; right < tail.length; right++) {
      slopes.push((tail[right].retained - tail[left].retained) / (right - left))
    }
  }
  return median(slopes)
}

export function positive(value: number): number {
  return value > 0 ? value : 0
}

export function recoveryDelta(
  baseline: readonly MemorySample[],
  recovery: readonly MemorySample[],
): number {
  return positive(
    median(recovery.map((entry) => entry.retained)) -
      median(baseline.map((entry) => entry.retained)),
  )
}

export function parseCli(arguments_: readonly string[]): ReadonlyMap<string, CliValue> {
  const parsed = new Map<string, CliValue>()
  for (const argument of arguments_) {
    if (!argument.startsWith('--'))
      throw new TypeError(`unexpected argument: ${argument}`)
    const separator = argument.indexOf('=')
    const name = separator === -1 ? argument.slice(2) : argument.slice(2, separator)
    if (name.length === 0) throw new TypeError(`invalid option: ${argument}`)
    if (parsed.has(name)) throw new TypeError(`duplicate option: --${name}`)
    if (separator === -1) parsed.set(name, true)
    else {
      const value = argument.slice(separator + 1)
      if (value.length === 0) throw new TypeError(`invalid option: ${argument}`)
      parsed.set(name, value)
    }
  }
  return parsed
}

export function assertCliOptions(
  options: ReadonlyMap<string, CliValue>,
  allowed: readonly string[],
): void {
  const known = new Set(allowed)
  for (const name of options.keys()) {
    if (!known.has(name)) throw new TypeError(`unknown option: --${name}`)
  }
}

export function numberOption(
  options: ReadonlyMap<string, CliValue>,
  name: string,
  fallback: number,
): number {
  const raw = options.get(name)
  if (raw === undefined) return fallback
  if (typeof raw !== 'string') throw new TypeError(`--${name} requires a value`)
  const value = Number(raw)
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`--${name} must be a positive safe integer`)
  }
  return value
}

export function runIsolated<T>(
  entry: string,
  arguments_: readonly string[],
  maxOldSpaceMb = 4096,
): T {
  const output = execFileSync(
    process.execPath,
    [
      '--expose-gc',
      `--max-old-space-size=${maxOldSpaceMb}`,
      entry,
      '--child',
      ...arguments_,
    ],
    { encoding: 'utf8', maxBuffer: 16 << 20, stdio: ['ignore', 'pipe', 'pipe'] },
  )
  try {
    return JSON.parse(output)
  } catch (error) {
    throw new Error(`memory child returned invalid JSON: ${output.slice(0, 500)}`, {
      cause: error,
    })
  }
}

export function writeJsonArtifact(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`)
}

export function writeStructured(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`)
}
