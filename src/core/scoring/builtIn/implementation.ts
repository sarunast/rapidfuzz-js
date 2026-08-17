import type { MaybeSequence, Sequence } from '../../types.js'
import type { ScorerOptions } from './options.js'
import { PREPARE_SCORER, type PreparationFactory } from './preparation.js'

interface ScorerFlags {
  readonly worstScore: number
  readonly optimalScore: number
  readonly symmetric: boolean
}

interface Flagged {
  readonly rfScorerFlags: ScorerFlags
}

export interface PreparedCapability<TEvidence = never> {
  readonly [PREPARE_SCORER]: PreparationFactory<TEvidence>
}

export interface MetricImplementation<
  TOptions extends ScorerOptions = ScorerOptions,
  TEvidence = never,
>
  extends Flagged, PreparedCapability<TEvidence> {
  (left: Sequence, right: Sequence, options?: TOptions): number
}

export interface MaybeSequenceMetricImplementation<
  TOptions extends ScorerOptions = ScorerOptions,
  TEvidence = never,
>
  extends Flagged, PreparedCapability<TEvidence> {
  (left: MaybeSequence, right: MaybeSequence, options?: TOptions): number
}

export type ConfigurationSymmetryResolver = (
  options: Readonly<Record<string, unknown>>,
) => boolean

export type ConfigurationCanonicalizer = (
  options: Readonly<Record<string, unknown>>,
) => Readonly<Record<string, unknown>>

interface ScorerRegistration {
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

export function withPreparedFlags<
  TImplementation extends ErasedMetricImplementation,
  TEvidence = never,
>(
  implementation: TImplementation,
  flags: ScorerFlags,
  prepare: PreparationFactory<TEvidence>,
  registration: ScorerRegistration = {},
): TImplementation & Flagged & PreparedCapability<TEvidence> {
  const scorer = Object.assign(implementation, {
    rfScorerFlags: Object.freeze({ ...flags }),
    [PREPARE_SCORER]: prepare,
  })
  registrations.set(scorer, registration)
  return scorer
}
