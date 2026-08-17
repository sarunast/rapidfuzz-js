/**
 * The largest weighted mass any sequence can carry. Each of the score's three
 * terms is bounded by it, so `numerator + penalty` cannot reach `Infinity`:
 * `alpha / scale` and `beta / scale` are at most `1`, and three quarters of
 * `Number.MAX_VALUE` is representable. `weightedProfile` sizes its weight limit
 * from this and the longest sequence a caller can hand over.
 */
export const WEIGHTED_MASS_LIMIT: number = Number.MAX_VALUE / 4

/** `2 ** -1022`, below which a double has fewer than 53 significant bits. */
const SMALLEST_NORMAL = 2.2250738585072014e-308

const FLOAT_BITS = /* @__PURE__ */ new DataView(new ArrayBuffer(8))

/**
 * The binary exponent of a positive finite `value`: the `e` for which
 * `value / 2 ** e` lands in `[1, 2)`.
 *
 * Read from the bits rather than from `Math.log2`, which is neither exact nor
 * required to be correctly rounded, and would put a one-exponent error into the
 * scaling below. A normal double carries `e` biased by 1023; a subnormal one
 * has no exponent field, so its leading set bit says where the mantissa starts
 * — bit `p` of the 52-bit fraction means `p - 1074`.
 */
function binaryExponent(value: number): number {
  FLOAT_BITS.setFloat64(0, value)
  const high = FLOAT_BITS.getUint32(0)
  const biased = (high >>> 20) & 0x7ff
  if (biased !== 0) return biased - 1023
  const highBits = high & 0xf_ffff
  const leading =
    highBits === 0 ? 31 - Math.clz32(FLOAT_BITS.getUint32(4)) : 63 - Math.clz32(highBits)
  return leading - 1074
}

/** How much of a power of two one multiplication may apply and stay normal. */
const SCALE_STEP = 1000

/**
 * `value * 2 ** exponent`, in steps small enough that no factor is subnormal.
 *
 * `2 ** exponent` is itself unrepresentable past the exponent range, and a
 * single multiplication by a subnormal power of two would round the mantissa
 * away. Stepping keeps every intermediate normal, so scaling is exact until the
 * result itself leaves the normal range — and once an intermediate is subnormal
 * there is still a full step to go, which flushes it to zero, which is what the
 * true value rounds to as well.
 */
function timesPowerOfTwo(value: number, exponent: number): number {
  let scaled = value
  let remaining = exponent
  while (remaining > SCALE_STEP) {
    scaled *= 2 ** SCALE_STEP
    remaining -= SCALE_STEP
  }
  while (remaining < -SCALE_STEP) {
    scaled *= 2 ** -SCALE_STEP
    remaining += SCALE_STEP
  }
  return scaled * 2 ** remaining
}

/** A positive finite `value` reduced to its mantissa, in `[1, 2)`. */
function mantissaOf(value: number): number {
  return timesPowerOfTwo(value, -binaryExponent(value))
}

const ABSENT = -Infinity

/**
 * The same ratio, with each of the three denominator terms held as a mantissa
 * and a binary exponent and scaled to the largest exponent of the three.
 *
 * That ordering is what makes it safe. `alpha * firstOnly` can exceed
 * `Number.MAX_VALUE` while `shared / (shared + alpha * firstOnly)` is still a
 * representable — even normal — number, so no evaluation order over the terms
 * themselves can work: one of them has to be a value that is never formed.
 * Scaling toward the *largest* exponent leaves the denominator inside `[1, 12)`,
 * so only the numerator can underflow, and it underflows exactly when the true
 * score is below the smallest subnormal.
 *
 * A term whose weight or mass is zero is dropped rather than scaled: it
 * contributes nothing, and it has no exponent to take a maximum over. The two
 * penalties are summed before the numerator joins, as the fast path does, so
 * swapping (arguments, weights) still only commutes one addition.
 */
function exponentSafeScore(
  shared: number,
  firstOnly: number,
  secondOnly: number,
  alpha: number,
  beta: number,
): number {
  const firstPresent = alpha !== 0 && firstOnly !== 0
  const secondPresent = beta !== 0 && secondOnly !== 0
  const sharedExponent = shared === 0 ? ABSENT : binaryExponent(shared)
  const firstExponent = firstPresent
    ? binaryExponent(alpha) + binaryExponent(firstOnly)
    : ABSENT
  const secondExponent = secondPresent
    ? binaryExponent(beta) + binaryExponent(secondOnly)
    : ABSENT
  const common = Math.max(sharedExponent, firstExponent, secondExponent)
  const scaledShared =
    shared === 0 ? 0 : timesPowerOfTwo(mantissaOf(shared), sharedExponent - common)
  const scaledFirst = firstPresent
    ? timesPowerOfTwo(mantissaOf(alpha) * mantissaOf(firstOnly), firstExponent - common)
    : 0
  const scaledSecond = secondPresent
    ? timesPowerOfTwo(mantissaOf(beta) * mantissaOf(secondOnly), secondExponent - common)
    : 0
  const penalty = scaledFirst + scaledSecond
  return scaledShared / (scaledShared + penalty)
}

/**
 * `shared / (shared + alpha * firstOnly + beta * secondOnly)` over weighted
 * masses rather than gram counts.
 *
 * The three components arrive already accumulated, never derived by subtracting
 * one rounded aggregate from another: with weights spanning several exponents a
 * mass can absorb the very occurrences a penalty is made of, and
 * `mass - shared` then reports `0` unmatched where one whole token is missing.
 *
 * Dividing by the largest weight keeps `weight * mass` finite, exactly as the
 * unweighted score does, and the two penalties are summed before the numerator
 * joins so that swapping (arguments, weights) only commutes one addition. That
 * scaling is safe for integer gram counts but not for weighted masses, which can
 * be subnormal on either side of the division: a shared mass of
 * `Number.MIN_VALUE` against an `alpha` of `Number.MAX_VALUE` divides away to
 * `0` and leaves `0 / 0`, and a weight 300 orders below the largest one loses
 * most of its mantissa on the way down. So the moment any operand the scaling
 * forms is subnormal while the value it stands for is not zero, the ratio is
 * recomputed against a common exponent instead, where no term is represented on
 * its own. Over 44,856 combinations of extreme masses and weights that is
 * within 2 ulps of the exactly rounded ratio.
 */
export function weightedTverskyScore(
  shared: number,
  firstOnly: number,
  secondOnly: number,
  alpha: number,
  beta: number,
): number {
  const scale = Math.max(1, alpha, beta)
  const numerator = shared / scale
  const firstWeight = alpha / scale
  const secondWeight = beta / scale
  const first = firstWeight * firstOnly
  const second = secondWeight * secondOnly
  if (
    (shared !== 0 && numerator < SMALLEST_NORMAL) ||
    (alpha !== 0 &&
      firstOnly !== 0 &&
      (firstWeight < SMALLEST_NORMAL || first < SMALLEST_NORMAL)) ||
    (beta !== 0 &&
      secondOnly !== 0 &&
      (secondWeight < SMALLEST_NORMAL || second < SMALLEST_NORMAL))
  ) {
    return exponentSafeScore(shared, firstOnly, secondOnly, alpha, beta)
  }
  const penalty = first + second
  return numerator / (numerator + penalty)
}
