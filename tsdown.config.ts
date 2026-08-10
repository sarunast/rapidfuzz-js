import { defineConfig, type UserConfig } from 'tsdown'

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
})

export default config
