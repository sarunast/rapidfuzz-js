import type { Direction, MaybeSequence } from '../types.js'
import {
  COMPILE,
  type AnyMetricCompilation,
  type MetricCompilation,
} from './compilation.js'
import type { AnyBrand } from './preparedChoice.js'

/**
 * The configuration of a metric that has none.
 *
 * `Record<string, never>` rather than `Record<never, never>`: the second has no
 * keys to check, so every object is assignable to it and `{ threshold: 1 }`
 * would pass for "no configuration". This one has a key type, and every value
 * under it is `never`, so only an empty object satisfies it.
 */
export type NoConfiguration = Readonly<Record<string, never>>

/**
 * A metric this package built, carrying the compilation hook `createScorer`
 * reads. It stays an ordinary callable — `levenshtein.distance(a, b)` is the
 * whole public contract — and the hook is a symbol so that nothing about the
 * internal protocol shows up in inspection or completion.
 *
 * Not a type to implement. A metric of your own is a plain
 * `(a, b) => number` handed to `createScorer` with a configuration describing
 * it; the overload for that takes a bare function and never looks for a hook.
 */
export interface Metric<
  TDirection extends Direction,
  TConfig extends object = NoConfiguration,
  TBrand = AnyBrand,
  TExplains extends TConfig = never,
  TEvidence = never,
> {
  (a: MaybeSequence, b: MaybeSequence): number
  readonly [COMPILE]: (
    configuration: TConfig | undefined,
  ) => MetricCompilation<TDirection, TBrand, TExplains, TEvidence>
}

/**
 * A metric read without its capability types.
 *
 * Every capability-carrying metric is assignable to it, because both erased
 * parameters occur only in covariant positions of the compilation it returns.
 * Anything that merely *accepts* one of this package's metrics takes this;
 * anything that *declares* one keeps the real parameters:
 *
 * ```text
 * construction code    Metric<…real capability…>
 * runtime plumbing     AnyMetric<…> / AnyMetricCompilation<…>
 * public surface       Scorer  |  ExplainableScorer
 * ```
 *
 * It is a separate interface rather than a filled-in `Metric` because
 * `TExplains extends TConfig` makes the inline widening unspellable —
 * `object extends never` is false — and `never` in its place would refuse the
 * very compilations it has to accept.
 */
export interface AnyMetric<
  TDirection extends Direction,
  TConfig extends object = never,
  TBrand = AnyBrand,
> {
  (a: MaybeSequence, b: MaybeSequence): number
  readonly [COMPILE]: (
    configuration: TConfig | undefined,
  ) => AnyMetricCompilation<TDirection, TBrand>
}

/**
 * Whether `value` is one of this package's metrics, asked of `unknown` because
 * it decides a public boundary: `createScorer(null)` reaches here from
 * JavaScript, where the types do not apply, and `COMPILE in null` would throw
 * before the controlled error further down could be raised.
 *
 * `Object.hasOwn` rather than `in`: `COMPILE` is installed on the metric
 * function itself, so an inherited hook is not a metric this package made — it
 * is something that borrowed a prototype from one.
 *
 * It narrows to {@link AnyMetric}: the question it answers is "is this one of
 * ours", which no capability can affect, so this is where a capability is
 * deliberately dropped.
 */
export function isBuiltInMetric<
  TDirection extends Direction,
  TConfig extends object,
  TBrand,
>(value: unknown): value is AnyMetric<TDirection, TConfig, TBrand> {
  if (typeof value !== 'function' || !Object.hasOwn(value, COMPILE)) return false
  const compile: unknown = Reflect.get(value, COMPILE)
  return typeof compile === 'function'
}
