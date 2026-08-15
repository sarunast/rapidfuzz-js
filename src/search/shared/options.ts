import type {
  BestOptions,
  MatcherOptions,
  PreparedMatcherOptions,
  SearchOptions,
} from '../types.js'

type AnyMatcherOptionKey =
  | keyof MatcherOptions<unknown>
  | keyof PreparedMatcherOptions<unknown>

export const MATCHER_OPTION_KEYS: readonly string[] = [
  'scorer',
  'getText',
  'getPrepared',
  'normalize',
  'missingItems',
] as const satisfies readonly AnyMatcherOptionKey[]

export const INDEXED_MATCHER_OPTION_KEYS: readonly string[] = [
  'scorer',
  'getText',
  'normalize',
  'missingItems',
]

export const BEST_OPTION_KEYS: readonly string[] = [
  'scorer',
  'getText',
  'getPrepared',
  'normalize',
  'missingItems',
  'threshold',
] as const satisfies readonly (AnyMatcherOptionKey | keyof BestOptions)[]

export const SEARCH_OPTION_KEYS: readonly string[] = [
  'scorer',
  'getText',
  'getPrepared',
  'normalize',
  'missingItems',
  'threshold',
  'limit',
] as const satisfies readonly (AnyMatcherOptionKey | keyof SearchOptions)[]

export const CALL_BEST_KEYS: readonly string[] = [
  'threshold',
] as const satisfies readonly (keyof BestOptions)[]

export const CALL_SEARCH_KEYS: readonly string[] = [
  'threshold',
  'limit',
] as const satisfies readonly (keyof SearchOptions)[]

export function resultLimit(value: number | null | undefined): number | null {
  if (value === null) return null
  const limit = value ?? 5
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new RangeError('limit must be null or a non-negative safe integer')
  }
  return limit
}
