import type { MaybeSequence, Sequence } from '../../core/types.js'
import { PREPARE_SCORER, type PreparationFactory } from './preparation.js'
import type { ScorerOptions } from './types.js'

export interface ScorerFlags {
  readonly worstScore: number
  readonly optimalScore: number
  readonly symmetric: boolean
}

export interface Flagged {
  readonly rfScorerFlags: ScorerFlags
}

export interface PreparedCapability {
  readonly [PREPARE_SCORER]: PreparationFactory
}

export interface MetricImplementation<TOptions extends ScorerOptions = ScorerOptions>
  extends Flagged, PreparedCapability {
  (left: Sequence, right: Sequence, options?: TOptions): number
}

export interface MaybeSequenceMetricImplementation<
  TOptions extends ScorerOptions = ScorerOptions,
>
  extends Flagged, PreparedCapability {
  (left: MaybeSequence, right: MaybeSequence, options?: TOptions): number
}

export type ConfigurationSymmetryResolver = (
  options: Readonly<Record<string, unknown>>,
) => boolean

export type ConfigurationCanonicalizer = (
  options: Readonly<Record<string, unknown>>,
) => Readonly<Record<string, unknown>>

export interface ScorerRegistration {
  readonly configurationSymmetry?: ConfigurationSymmetryResolver | undefined
  readonly configurationCanonicalizer?: ConfigurationCanonicalizer | undefined
}

const registrations = new WeakMap<object, ScorerRegistration>()

export function configurationSymmetryOf(
  scorer: object,
): ConfigurationSymmetryResolver | null {
  return registrations.get(scorer)?.configurationSymmetry ?? null
}

export function configurationCanonicalizerOf(
  scorer: object,
): ConfigurationCanonicalizer | null {
  return registrations.get(scorer)?.configurationCanonicalizer ?? null
}

export const DISTANCE_FLAGS: ScorerFlags = /* @__PURE__ */ Object.freeze({
  optimalScore: 0,
  worstScore: Number.POSITIVE_INFINITY,
  symmetric: true,
})

export const SIMILARITY_FLAGS: ScorerFlags = /* @__PURE__ */ Object.freeze({
  optimalScore: Number.POSITIVE_INFINITY,
  worstScore: 0,
  symmetric: true,
})

export const NORMALIZED_DISTANCE_FLAGS: ScorerFlags = /* @__PURE__ */ Object.freeze({
  optimalScore: 0,
  worstScore: 1,
  symmetric: true,
})

export const NORMALIZED_SIMILARITY_FLAGS: ScorerFlags = /* @__PURE__ */ Object.freeze({
  optimalScore: 1,
  worstScore: 0,
  symmetric: true,
})

export const FUZZ_FLAGS: ScorerFlags = /* @__PURE__ */ Object.freeze({
  optimalScore: 100,
  worstScore: 0,
  symmetric: true,
})

export type ErasedMetricImplementation = (
  left: Sequence,
  right: Sequence,
  options?: ScorerOptions,
) => number

export function withPreparedFlags<TImplementation extends ErasedMetricImplementation>(
  implementation: TImplementation,
  flags: ScorerFlags,
  prepare: PreparationFactory,
  registration: ScorerRegistration = {},
): TImplementation & Flagged & PreparedCapability {
  // Decorates the implementation instead of wrapping it, so scoring pays no
  // extra call. Registering unconditionally replaces any earlier registration.
  const scorer = Object.assign(implementation, {
    rfScorerFlags: Object.freeze({ ...flags }),
    [PREPARE_SCORER]: prepare,
  })
  registrations.set(scorer, registration)
  return scorer
}
