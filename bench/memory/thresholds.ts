export interface ScenarioThreshold {
  readonly slopeBytesPerBatch: number
  readonly recoveryBytes: number
}

export const SOAK_SCENARIOS = [
  'steady',
  'query-profile',
  'touched-set',
  'arbitrary-elements',
] as const

export type SoakScenario = (typeof SOAK_SCENARIOS)[number]

export const SOAK_CALIBRATION = {
  nodeMajor: 26,
  platform: 'darwin',
  arch: 'arm64',
} as const

/** Selected from `soak.ts --calibrate`; see the memory-tooling guide for provenance. */
export const SOAK_THRESHOLDS = {
  steady: { slopeBytesPerBatch: 12_288, recoveryBytes: 327_680 },
  'query-profile': { slopeBytesPerBatch: 12_288, recoveryBytes: 327_680 },
  'touched-set': { slopeBytesPerBatch: 12_288, recoveryBytes: 327_680 },
  'arbitrary-elements': { slopeBytesPerBatch: 12_288, recoveryBytes: 327_680 },
} as const satisfies Record<SoakScenario, ScenarioThreshold>
