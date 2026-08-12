// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightTypeDoc, { typeDocSidebarGroup } from 'starlight-typedoc';

// The API reference is generated from the built declaration files of the
// linked library (rapidfuzz-js@link:..). Each file in src/api-entries/
// re-exports one public subpath so the generated module names match the
// import paths users type. Run `pnpm build` at the repo root first so
// ../dist exists.
const entries = './src/api-entries';
const entryPoints = [
	`${entries}/rapidfuzz-js.d.ts`,
	`${entries}/fuzz.d.ts`,
	`${entries}/levenshtein.d.ts`,
	`${entries}/indel.d.ts`,
	`${entries}/lcs.d.ts`,
	`${entries}/osa.d.ts`,
	`${entries}/damerau-levenshtein.d.ts`,
	`${entries}/hamming.d.ts`,
	`${entries}/jaro.d.ts`,
	`${entries}/jaro-winkler.d.ts`,
	`${entries}/prefix.d.ts`,
	`${entries}/postfix.d.ts`,
];

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'rapidfuzz-js',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/sarunast/rapidfuzz-js' },
			],
			plugins: [
				starlightTypeDoc({
					entryPoints,
					tsconfig: './tsconfig.typedoc.json',
					typeDoc: {
						name: 'rapidfuzz-js',
						readme: 'none',
						entryFileName: 'index',
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
						{ label: 'Jaro', slug: 'algorithms/jaro' },
						{ label: 'Jaro-Winkler', slug: 'algorithms/jaro-winkler' },
						{ label: 'Prefix and Postfix', slug: 'algorithms/prefix-postfix' },
						{ label: 'Fuzz', slug: 'algorithms/fuzz' },
					],
				},
				typeDocSidebarGroup,
				{ label: 'Benchmarks', slug: 'benchmarks' },
			],
		}),
	],
});
