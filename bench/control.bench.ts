/**
 * Control cases: work that this library's code cannot change.
 *
 * `compare.ts` estimates how much faster or slower the machine itself was —
 * thermal state, background load, CPU frequency — and divides it out of every
 * other case's ratio. Estimating that from the suite's own cases cannot work:
 * if every kernel really did get ten percent slower, the estimate absorbs it
 * and the report says nothing moved. These four import nothing from `src`, so a
 * change in their timing is a change in the machine by construction.
 *
 * Between them they cover the resources the kernels actually use: an ALU-bound
 * loop, a streaming read wider than L2, hashed lookup and pointer chasing, and
 * a `charCodeAt` scan, which is how every scorer here reads its input.
 */

import { describe, measure } from './tooling/harness.js'

/**
 * Somewhere for a result to go that the optimiser cannot prove is dead. Without
 * one, a control loop can be eliminated entirely and then measures nothing.
 */
const sink = { value: 0 }

const VALUES = new Float64Array(1 << 19)
for (let i = 0; i < VALUES.length; i++) VALUES[i] = i * 0.5

const LOOKUPS = new Map<number, number>()
for (let i = 0; i < 10_000; i++) LOOKUPS.set(i * 7, i)

const TEXT = 'abcdefghijklmnopqrstuvwxyz0123456789'.repeat(6000)

describe('control', () => {
  measure('integer loop', () => {
    let total = 0
    for (let i = 0; i < 1_000_000; i++) total = (total + i * 3) | 0
    sink.value = total
  })

  measure('array traversal', () => {
    let total = 0
    for (let i = 0; i < VALUES.length; i++) total += VALUES[i]
    sink.value = total
  })

  measure('map lookup', () => {
    let total = 0
    for (let i = 0; i < 100_000; i++) total += LOOKUPS.get((i % 10_000) * 7) ?? 0
    sink.value = total
  })

  measure('charCodeAt scan', () => {
    let total = 0
    for (let i = 0; i < TEXT.length; i++) total += TEXT.charCodeAt(i)
    sink.value = total
  })
})
