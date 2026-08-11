import type {
  ErasedScorer,
  MaybeSequence,
  Processor,
} from '../../src/algorithms/shared/scorerSupport.js'

interface MatrixTestOptions {
  readonly scorer: ErasedScorer
  readonly processor?: Processor | undefined
  readonly scoreCutoff?: number | undefined
  readonly scoreHint?: number | undefined
}

function score(
  scorer: ErasedScorer,
  query: MaybeSequence,
  choice: MaybeSequence,
  options: MatrixTestOptions,
): number {
  const result: unknown = Reflect.apply(scorer, undefined, [query, choice, options])
  if (typeof result !== 'number')
    throw new TypeError('test scorer did not return a number')
  return result
}

/** A deliberately naive matrix used only by ported algorithm tests. */
export function matrixScores(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: MatrixTestOptions,
): number[][] {
  return queries.map((query) =>
    choices.map((choice) => score(options.scorer, query, choice, options)),
  )
}
