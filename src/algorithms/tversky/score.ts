export function tverskyScore(
  shared: number,
  gramsA: number,
  gramsB: number,
  alpha: number,
  beta: number,
): number {
  // Dividing everything by the largest weight keeps `weight * count` finite
  // for huge coefficients whose true score is still representable; weights at
  // or below 1 leave every operand — and the default Dice arithmetic — as is.
  // The two penalty terms are summed before the numerator joins so that
  // swapping (arguments, weights) only commutes one addition, keeping
  // `T(a, b, α, β)` bit-identical to `T(b, a, β, α)`.
  const scale = Math.max(1, alpha, beta)
  const numerator = shared / scale
  const unmatched =
    (alpha / scale) * (gramsA - shared) + (beta / scale) * (gramsB - shared)
  return numerator / (numerator + unmatched)
}
