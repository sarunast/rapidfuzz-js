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
    'src/fuzz/index.ts',
    'src/algorithms/levenshtein/index.ts',
    'src/algorithms/indel/index.ts',
    'src/algorithms/lcs/index.ts',
    'src/algorithms/osa/index.ts',
    'src/algorithms/cosine/index.ts',
    'src/algorithms/damerauLevenshtein/index.ts',
    'src/algorithms/dice/index.ts',
    'src/algorithms/hamming/index.ts',
    'src/algorithms/jaro/index.ts',
    'src/algorithms/jaroWinkler/index.ts',
    'src/algorithms/prefix/index.ts',
    'src/algorithms/postfix/index.ts',
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
  // "No runtime dependencies, and no Node built-ins in `src/`" as a build
  // failure rather than a convention. `neverBundle` keeps any npm import out of
  // the output, so an accidental one survives as a bare import instead of being
  // folded in silently; the empty `onlyImport` whitelist then refuses output
  // that imports anything at all. Built-ins are exempt only under
  // `platform: 'node'`, which this is not.
  deps: { neverBundle: true, onlyImport: [] },
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
