/**
 * The library version, inlined at build time by the `vite.define` entry in
 * `astro.config.ts`. Declared here because a `define` replacement has no import
 * to carry a type — without this the identifier is an undeclared global, which
 * `tsc` never sees (it does not read `.astro` files) but `astro check` reports.
 */
declare const LIBRARY_VERSION: string
