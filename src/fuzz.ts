/**
 * Port of `rapidfuzz.fuzz` (see `src/rapidfuzz/fuzz_py.py`).
 *
 * Every scorer here returns a percentage in `[0, 100]`, and returns `0` — not
 * `null` — when either input is `null`/`undefined`/`NaN`, matching upstream.
 *
 * This file is the public face of the module and holds no algorithm of its own.
 * The implementations live under `_fuzz/`, one module per scorer family;
 * `_fuzz/index.ts` documents which is which. What is assembled here is the pair
 * every scorer needs to be usable from `search`: the flags that tell a distance
 * from a similarity, and the prepared-query hook.
 */
import { FUZZ_FLAGS, withPreparedFlags, type NormalizedScorer } from './_common.js'
import {
  partialRatio_impl,
  partialTokenRatio_impl,
  partialTokenSetRatio_impl,
  partialTokenSortRatio_impl,
  prepareFuzz,
  qRatio_impl,
  ratio_impl,
  tokenRatio_impl,
  tokenSetRatio_impl,
  tokenSortRatio_impl,
  wRatio_impl,
} from './_fuzz/index.js'
import type { FuzzOptions } from './_fuzz/types.js'

export type { FuzzInput, FuzzOptions, ScoreAlignment } from './_fuzz/types.js'
export { partialRatioAlignment } from './_fuzz/index.js'

// Scorer flags let `process` tell distances from similarities.
export const ratio: NormalizedScorer<FuzzOptions> = /* @__PURE__ */ withPreparedFlags(
  ratio_impl,
  FUZZ_FLAGS,
  prepareFuzz('ratio'),
)
export const partialRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialRatio'),
  )
export const tokenSortRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tokenSortRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('tokenSortRatio'),
  )
export const tokenSetRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tokenSetRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('tokenSetRatio'),
  )
export const tokenRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tokenRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('tokenRatio'),
  )
export const partialTokenSortRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenSortRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenSortRatio'),
  )
export const partialTokenSetRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenSetRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenSetRatio'),
  )
export const partialTokenRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenRatio'),
  )
export const wRatio: NormalizedScorer<FuzzOptions> = /* @__PURE__ */ withPreparedFlags(
  wRatio_impl,
  FUZZ_FLAGS,
  prepareFuzz('wRatio'),
)
export const qRatio: NormalizedScorer<FuzzOptions> = /* @__PURE__ */ withPreparedFlags(
  qRatio_impl,
  FUZZ_FLAGS,
  prepareFuzz('qRatio'),
)
