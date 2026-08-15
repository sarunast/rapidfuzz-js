import type { PreparedCapability } from '../src/core/scoring/builtIn/implementation.js'
import { PREPARE_SCORER } from '../src/core/scoring/builtIn/preparation.js'
import type { PreparedKernel } from '../src/core/scoring/compilation.js'
import { validateSequence } from '../src/core/sequence.js'
import type { Sequence } from '../src/core/types.js'

/** Test-only adapter that always feeds the kernel its opaque prepared choice. */
export function prepareScorerOf(
  scorer: PreparedCapability,
): (query: Sequence, configuration: Readonly<Record<string, unknown>>) => PreparedKernel {
  const factory = scorer[PREPARE_SCORER]
  return (query, configuration) => {
    const preparation = factory(configuration)
    const kernel = preparation.prepareQuery(query)
    return (choice, threshold) =>
      kernel(preparation.prepareChoice(validateSequence(choice)), threshold)
  }
}
