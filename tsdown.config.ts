import { codecovRollupPlugin } from '@codecov/rollup-plugin'
import { defineConfig, type UserConfig } from 'tsdown'

// Bundle analysis is off unless a token says otherwise, which keeps it to CI.
// `pnpm build` runs on a laptop, and in the release job that publishes to npm —
// neither should depend on Codecov being reachable, and a build that failed
// over a size measurement would take the publish with it.
const uploadToken = process.env['CODECOV_TOKEN']
// Uploading is for CI. A token present locally still measures, and says so
// without sending anything.
const dryRun = process.env['CI'] !== 'true'

const config: UserConfig = defineConfig({
  entry: [
    'src/index.ts',
    'src/fuzz.ts',
    'src/search.ts',
    'src/match.ts',
    'src/utils.ts',
    'src/distance/index.ts',
    'src/distance/namespaces/Indel.ts',
    'src/distance/namespaces/LCSseq.ts',
    'src/distance/namespaces/Levenshtein.ts',
    'src/distance/namespaces/DamerauLevenshtein.ts',
    'src/distance/namespaces/OSA.ts',
    'src/distance/namespaces/Hamming.ts',
    'src/distance/namespaces/Jaro.ts',
    'src/distance/namespaces/JaroWinkler.ts',
    'src/distance/namespaces/Prefix.ts',
    'src/distance/namespaces/Postfix.ts',
  ],
  root: 'src',
  format: 'esm',
  dts: true,
  // Preserve source-module boundaries so subpath imports stay narrow and
  // browser bundlers can tree-shake at module granularity.
  unbundle: true,
  target: 'es2022',
  platform: 'neutral',
  sourcemap: true,
  plugins: [
    codecovRollupPlugin({
      enableBundleAnalysis: uploadToken !== undefined,
      bundleName: 'rapidfuzz-js',
      dryRun,
      // Spread rather than assigned: under `exactOptionalPropertyTypes` an
      // absent token is an absent property, not a property holding `undefined`,
      // and a cast to bridge the two is banned project-wide.
      ...(uploadToken === undefined ? {} : { uploadToken }),
    }),
  ],
})

export default config
