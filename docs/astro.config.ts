import { readFileSync } from 'node:fs'

import starlight from '@astrojs/starlight'
import { defineConfig } from 'astro/config'
import starlightTypeDoc, { typeDocSidebarGroup } from 'starlight-typedoc'

// The library's own version, so the badge in the header cannot drift from what
// is published. Read from the repository root rather than through the linked
// package, whose `exports` map deliberately does not expose `package.json`.
//
// `JSON.parse` returns `any`, so the field is narrowed rather than trusted:
// this value is inlined into every page, and a malformed read should fail here
// instead of rendering "vundefined" in the header.
const manifest: unknown = JSON.parse(
  readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
)
if (
  typeof manifest !== 'object' ||
  manifest === null ||
  !('version' in manifest) ||
  typeof manifest.version !== 'string'
) {
  throw new Error('../package.json has no string version')
}
const version = manifest.version

// Each file in src/api-entries/ re-exports one public subpath, so the generated
// module names match the import paths users type. tsconfig.typedoc.json then
// resolves those subpaths to ../src rather than to the linked package's dist,
// which is what gives the reference its JSDoc and its source links.
const entries = './src/api-entries'
const entryPoints = [
  `${entries}/rapidfuzz-js.d.ts`,
  `${entries}/fuzz.d.ts`,
  `${entries}/levenshtein.d.ts`,
  `${entries}/indel.d.ts`,
  `${entries}/lcs.d.ts`,
  `${entries}/osa.d.ts`,
  `${entries}/damerau-levenshtein.d.ts`,
  `${entries}/hamming.d.ts`,
  `${entries}/dice.d.ts`,
  `${entries}/cosine.d.ts`,
  `${entries}/jaro.d.ts`,
  `${entries}/jaro-winkler.d.ts`,
  `${entries}/prefix.d.ts`,
  `${entries}/postfix.d.ts`,
]

// GitHub Pages serves a project site under the repository name, so every URL
// carries that prefix. Moving to a custom domain means `site` becomes the
// domain and `base` becomes '/' — nothing else here changes, because the
// rewriting below reads `base` rather than repeating it.
const site = 'https://sarunast.github.io'
const base = '/rapidfuzz-js'

/**
 * Prefixes `base` onto site-absolute links written in Markdown.
 *
 * Starlight applies `base` to the links it generates itself — the sidebar, the
 * header, pagination, the "Edit page" footer — but a link written as
 * `](/guides/comparing-strings/)` in a content file reaches the HTML untouched,
 * and 404s once the site is not at the domain root. That covers both the pages
 * written by hand and the ~440 cross-references TypeDoc emits into the
 * generated API section.
 *
 * Doing it here rather than editing each link keeps content authored the way
 * Starlight's own documentation writes it, and means a new page cannot
 * reintroduce the bug.
 *
 * The tree is walked as `unknown` and narrowed on the way down rather than
 * typed as hast's `Root`. `@types/hast` is not a dependency here, and a
 * transformer that accepts `unknown` is assignable to one that accepts `Root`
 * anyway — so this stays type-safe without adding a package to describe four
 * fields.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function rehypeBasePaths() {
  const walk = (node: unknown): void => {
    if (!isRecord(node)) return

    if (node.type === 'element' && node.tagName === 'a' && isRecord(node.properties)) {
      const { href } = node.properties
      // A protocol-relative `//host/path` is external despite the leading
      // slash, and re-running over an already-prefixed href would double it.
      if (
        typeof href === 'string' &&
        href.startsWith('/') &&
        !href.startsWith('//') &&
        !href.startsWith(`${base}/`)
      ) {
        node.properties.href = `${base}${href}`
      }
    }

    const { children } = node
    if (Array.isArray(children)) {
      for (const child of children) walk(child)
    }
  }
  return walk
}

// https://astro.build/config
export default defineConfig({
  site,
  base,
  markdown: {
    rehypePlugins: [rehypeBasePaths],
  },
  integrations: [
    starlight({
      title: 'rapidfuzz-js',
      components: {
        SiteTitle: './src/components/SiteTitle.astro',
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/sarunast/rapidfuzz-js',
        },
      ],
      plugins: [
        starlightTypeDoc({
          entryPoints,
          tsconfig: './tsconfig.typedoc.json',
          typeDoc: {
            name: 'rapidfuzz-js',
            readme: 'none',
            entryFileName: 'index',
            // TypeDoc emits a source link only for a file git tracks, which is
            // the reason the entry points resolve to `src/` — pointed at the
            // gitignored `dist/`, every "Defined in" was dead text.
            sourceLinkTemplate:
              'https://github.com/sarunast/rapidfuzz-js/blob/{gitRevision}/{path}#L{line}',
            gitRevision: 'main',
          },
        }),
      ],
      sidebar: [
        { label: 'Introduction', slug: 'introduction' },
        { label: 'Getting started', slug: 'getting-started' },
        {
          label: 'Concepts',
          items: [
            { label: 'Metrics', slug: 'concepts/metrics' },
            { label: 'Scorers', slug: 'concepts/scorers' },
            { label: 'Matchers', slug: 'concepts/matchers' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Comparing strings', slug: 'guides/comparing-strings' },
            { label: 'Finding the best match', slug: 'guides/finding-the-best-match' },
            { label: 'Searching collections', slug: 'guides/searching-collections' },
            { label: 'Matching records', slug: 'guides/matching-records' },
            { label: 'Prepared choices', slug: 'guides/prepared-choices' },
            { label: 'Preprocessing', slug: 'guides/preprocessing' },
            { label: 'Performance', slug: 'guides/performance' },
          ],
        },
        {
          label: 'Algorithms',
          items: [
            { label: 'Levenshtein', slug: 'algorithms/levenshtein' },
            { label: 'Indel', slug: 'algorithms/indel' },
            { label: 'LCS', slug: 'algorithms/lcs' },
            { label: 'OSA', slug: 'algorithms/osa' },
            { label: 'Damerau-Levenshtein', slug: 'algorithms/damerau-levenshtein' },
            { label: 'Hamming', slug: 'algorithms/hamming' },
            { label: 'Sørensen-Dice', slug: 'algorithms/dice' },
            { label: 'Cosine', slug: 'algorithms/cosine' },
            { label: 'Jaro', slug: 'algorithms/jaro' },
            { label: 'Jaro-Winkler', slug: 'algorithms/jaro-winkler' },
            { label: 'Prefix and Postfix', slug: 'algorithms/prefix-postfix' },
            { label: 'Fuzz', slug: 'algorithms/fuzz' },
          ],
        },
        { label: 'Errors', slug: 'reference/errors' },
        typeDocSidebarGroup,
        { label: 'Benchmarks', slug: 'benchmarks' },
      ],
    }),
  ],
  vite: {
    define: {
      LIBRARY_VERSION: JSON.stringify(version),
    },
  },
})
