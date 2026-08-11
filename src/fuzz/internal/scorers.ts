/** Internal callable scorer implementations adapted by the public fuzz metrics. */
import {
  FUZZ_FLAGS,
  withPreparedFlags,
  type NormalizedScorer,
} from '../../algorithms/shared/scorerSupport.js'
import type { FuzzOptions } from '../types.js'
import { partialRatioAlignment, partialRatio_impl, ratio_impl } from './basic.js'
import { wRatio_impl } from './composite.js'
import { prepareFuzz } from './prepared.js'
import { prepareRatio } from './prepareRatio.js'
import { prepareTokenSort } from './prepareTokenSort.js'
import {
  partialTokenRatio_impl,
  partialTokenSetRatio_impl,
  partialTokenSortRatio_impl,
  tokenRatio_impl,
  tokenSetRatio_impl,
  tokenSortRatio_impl,
} from './tokenScorers.js'

export type { FuzzInput, FuzzOptions, ScoreAlignment } from '../types.js'
export { partialRatioAlignment }

export const ratio: NormalizedScorer<FuzzOptions> = /* @__PURE__ */ withPreparedFlags(
  ratio_impl,
  FUZZ_FLAGS,
  prepareRatio(),
)
export const partialRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialRatio'),
  )
export const tokenSortRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(tokenSortRatio_impl, FUZZ_FLAGS, prepareTokenSort())
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
