import { FUZZ_FLAGS, withPreparedFlags, type NormalizedScorer } from '../_common.js'
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
} from './index.js'
import type { FuzzOptions } from './types.js'

export type { FuzzInput, FuzzOptions, ScoreAlignment } from './types.js'
export { partialRatioAlignment } from './index.js'

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
