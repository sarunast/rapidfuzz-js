import { expect, it } from 'vitest'

import {
  configurationCanonicalizerOf,
  configurationSymmetryOf,
  DISTANCE_FLAGS,
  withPreparedFlags,
} from './implementation.js'
import { prepareMetric } from './preparation.js'

it('replaces an earlier registration instead of merging into it', () => {
  const prepare = prepareMetric(
    'distance',
    () => 0,
    () => 0,
  )
  const implementation = (): number => 0
  withPreparedFlags(implementation, DISTANCE_FLAGS, prepare, {
    configurationSymmetry: () => false,
    configurationCanonicalizer: (options) => options,
  })
  expect(configurationSymmetryOf(implementation)).not.toBeNull()
  expect(configurationCanonicalizerOf(implementation)).not.toBeNull()
  withPreparedFlags(implementation, DISTANCE_FLAGS, prepare)
  expect(configurationSymmetryOf(implementation)).toBeNull()
  expect(configurationCanonicalizerOf(implementation)).toBeNull()
})

it('reports no registration for an implementation it never saw', () => {
  const stranger = (): number => 0
  expect(configurationSymmetryOf(stranger)).toBeNull()
  expect(configurationCanonicalizerOf(stranger)).toBeNull()
})
