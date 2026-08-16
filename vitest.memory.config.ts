import { defineConfig, type ViteUserConfig } from 'vitest/config'

const config: ViteUserConfig = defineConfig({
  test: {
    include: ['tests/memory/*.memory.test.ts'],
    fileParallelism: false,
    pool: 'forks',
    maxWorkers: 1,
    sequence: { concurrent: false },
  },
})

export default config
