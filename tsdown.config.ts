import { defineConfig, type UserConfig } from 'tsdown'

const config: UserConfig = defineConfig({
  entry: [
    'src/index.ts',
    'src/fuzz.ts',
    'src/search.ts',
    'src/match.ts',
    'src/configure.ts',
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
  format: ['esm', 'cjs'],
  dts: true,
  // Emit one output file per source file so consumers can drop what they
  // don't import. A single bundled index.js defeats tree-shaking for
  // anyone whose bundler can't re-analyse our output.
  unbundle: true,
  target: 'es2022',
  platform: 'neutral',
  clean: true,
  treeshake: true,
  sourcemap: true,
})

export default config
