import type { MaybeSequence, Sequence } from '../../core/types.js'
import { PREPARE_SCORER, type PreparedScorerFactory } from './preparation.js'
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
  readonly [PREPARE_SCORER]: PreparedScorerFactory
}

export interface Scorer<O extends ScorerOptions = ScorerOptions>
  extends Flagged, PreparedCapability {
  (left: Sequence, right: Sequence, options?: O): number
}

export interface NormalizedScorer<O extends ScorerOptions = ScorerOptions>
  extends Flagged, PreparedCapability {
  (left: MaybeSequence, right: MaybeSequence, options?: O): number
}

export type ConfigurationFlagsResolver = (
  options: Readonly<Record<string, unknown>>,
) => ScorerFlags

export type ConfigurationCanonicalizer = (
  options: Readonly<Record<string, unknown>>,
) => Readonly<Record<string, unknown>>

export interface ScorerRegistration {
  readonly configurationFlags?: ConfigurationFlagsResolver | undefined
  readonly configurationCanonicalizer?: ConfigurationCanonicalizer | undefined
}

const configurationFlagResolvers = new WeakMap<object, ConfigurationFlagsResolver>()
const optionCanonicalizers = new WeakMap<object, ConfigurationCanonicalizer>()

export function configurationFlagsOf(scorer: object): ConfigurationFlagsResolver | null {
  return configurationFlagResolvers.get(scorer) ?? null
}

export function configurationCanonicalizerOf(
  scorer: object,
): ConfigurationCanonicalizer | null {
  return optionCanonicalizers.get(scorer) ?? null
}

export const DISTANCE_FLAGS: ScorerFlags = {
  optimalScore: 0,
  worstScore: Number.POSITIVE_INFINITY,
  symmetric: true,
}

export const NORMALIZED_SIMILARITY_FLAGS: ScorerFlags = {
  optimalScore: 1,
  worstScore: 0,
  symmetric: true,
}

export const FUZZ_FLAGS: ScorerFlags = {
  optimalScore: 100,
  worstScore: 0,
  symmetric: true,
}

export type ErasedScorer = (left: never, right: never, options?: never) => number

export interface PreparedErasedScorer extends ErasedScorer {
  readonly [PREPARE_SCORER]: PreparedScorerFactory
}

export function withPreparedFlags<F extends ErasedScorer>(
  implementation: F,
  flags: ScorerFlags,
  prepare: PreparedScorerFactory,
  registration: ScorerRegistration = {},
): F & Flagged & PreparedErasedScorer {
  const scorer = Object.assign(implementation, {
    rfScorerFlags: Object.freeze({ ...flags }),
    [PREPARE_SCORER]: prepare,
  })
  if (registration.configurationFlags !== undefined) {
    configurationFlagResolvers.set(scorer, registration.configurationFlags)
  }
  if (registration.configurationCanonicalizer !== undefined) {
    optionCanonicalizers.set(scorer, registration.configurationCanonicalizer)
  }
  return scorer
}
