import {
  PREPARE_CHOICE,
  PREPARE_SCORER,
  withChoicePreparer,
  type PreparedCapability,
  type PreparedScorerFactory,
} from '../../src/algorithms/shared/scorerSupport.js'
import { validateSequence } from '../../src/core/sequence.js'

/** Test-only adapter that always feeds the kernel its opaque prepared choice. */
export function prepareScorerOf(scorer: PreparedCapability): PreparedScorerFactory {
  const factory = scorer[PREPARE_SCORER]
  return withChoicePreparer((query, configuration) => {
    const score = factory(query, configuration)
    return (choice, threshold) =>
      score(factory[PREPARE_CHOICE](validateSequence(choice)), threshold)
  }, factory[PREPARE_CHOICE])
}
