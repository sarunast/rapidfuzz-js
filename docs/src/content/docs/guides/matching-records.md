---
title: Matching records
description: Deduplicating multi-field records — why blending fields into one string fails, and how to score each field with the measure that suits it.
---

Every example so far compared two strings. Real deduplication compares
**records**: a job posting has a title, a company and a location; a customer
has a name, an address and an email. The instinct is to join the fields into
one string and score that. It's the wrong instinct, and it fails in a way
that's hard to notice — high scores on pairs that aren't duplicates.

This guide works through one real case: deciding whether two job postings
describe the same opening.

## Why blending the fields fails

Take two genuinely different jobs at the same employer, and compare them as
one concatenated blob — title, company, location, and the boilerplate every
posting carries:

```ts
import { tokenSetRatio } from 'rapidfuzz-js/fuzz'

// 'AI Software Engineer' + shared company, location, and benefits blurb
// vs 'Head of Strategic Finance Projects' + the same shared text
tokenSetRatio(blendedA, blendedB) // 94.25
```

94 out of 100 for two unrelated roles. Compare the titles alone — the only
field that actually distinguishes them — and you get `40.74`.

Two things went wrong, and they compound:

- **The boilerplate outvotes the signal.** One bag of tokens lets company,
  location and the benefits blurb — identical on both sides — drown out the
  title. The more text two records share by convention, the more similar
  every pair looks.
- **`tokenSetRatio` returns 100 on containment.** It factors out the
  words both sides share, so whenever one token set _contains_ the other,
  the score is a perfect 100 regardless of how much extra the longer side
  carries.

That second property is worth seeing on its own, because it's the trap:

```ts
import { tokenSetRatio, tokenSortRatio } from 'rapidfuzz-js/fuzz'

tokenSetRatio('data engineer', 'data engineer cloud platform') // 100
tokenSortRatio('data engineer', 'data engineer cloud platform') // 63.41
```

`tokenSetRatio` isn't broken — containment is exactly what you want for
_some_ fields. It's the wrong opinion for a job title, where the extra words
are the difference between two openings.

## Score each field independently

The fix is to stop blending and give each field the measure its noise
deserves:

| Field   | Scorer           | Because                                                                 |
| ------- | ---------------- | ----------------------------------------------------------------------- |
| Title   | `tokenSortRatio` | Length-aware: extra words lower the score, which is what you want       |
| Company | `tokenSetRatio`  | Containment is correct: "Hoval Schweiz" and "Hoval AG" are one employer |

Each field then gets its own threshold, and a pair is a duplicate only when
**every** field agrees. One field's confidence can no longer paper over
another's disagreement — which is precisely what the blended string allowed.

## Normalization carries the domain knowledge

Before either scorer runs, fold the field into a comparable form. This is
where knowledge about _your_ data belongs — the scorer measures what's left
over, and the more the normalizer folds, the less the threshold has to do:

```ts
import { normalizeText } from 'rapidfuzz-js'

const LEGAL_FORMS = /\b(ag|sa|gmbh|holding|group|schweiz|switzerland)\b/g
const foldCompany = (value) =>
  normalizeText(value).replace(LEGAL_FORMS, ' ').replace(/\s+/g, ' ').trim()

const GENDER_TAGS = /\b(m\s*w\s*d|f\s*m\s*d|all genders)\b/g
const foldTitle = (value) =>
  normalizeText(value).replace(GENDER_TAGS, ' ').replace(/\s+/g, ' ').trim()
```

It moves scores in both directions, and both are the point:

| Pair                         | Raw     | Folded  |
| ---------------------------- | ------- | ------- |
| `Hoval Schweiz` / `Hoval AG` | `76.92` | `100`   |
| `Acme AG` / `Beta AG`        | `57.14` | `25.00` |

The same employer goes up because the legal suffix stopped being a
difference. Two _different_ employers go **down**, because the shared `AG`
had been inflating them. A normalizer that only pushed scores up would just
be blurring the data.

## Putting it together

Prepared choices carry the indexed side, so the token work happens once per
record rather than once per comparison. The title — the selective field —
drives the search; the company confirms the survivors:

```ts
import { createScorer, normalizeText, searchIter } from 'rapidfuzz-js'
import { tokenSetRatio, tokenSortRatio } from 'rapidfuzz-js/fuzz'

const titleScorer = createScorer(tokenSortRatio)
const companyScorer = createScorer(tokenSetRatio)

const TITLE_THRESHOLD = 80
const COMPANY_THRESHOLD = 85

function indexPosting(posting) {
  return {
    posting,
    company: foldCompany(posting.company),
    title: titleScorer.prepareChoice(posting.title, { normalize: foldTitle }),
  }
}

function findDuplicate(probe, index) {
  const company = foldCompany(probe.company)

  for (const match of searchIter(probe.title, index, {
    scorer: titleScorer,
    getPrepared: (row) => row.title,
    normalize: foldTitle,
    threshold: TITLE_THRESHOLD,
  })) {
    const companyScore = companyScorer.score(company, match.item.company, {
      threshold: COMPANY_THRESHOLD,
    })
    if (companyScore === undefined) continue
    return { matched: match.item.posting, titleScore: match.score, companyScore }
  }
  return undefined
}
```

On a three-posting index, that separates all three cases correctly:

```text
Senior Data Engineer (m/w/d) @ Hoval Schweiz   → duplicate of "Senior Data Engineer" (title 100, company 100)
AI Software Engineer @ Acme AG                 → new posting
Data Engineer @ Acme AG                        → new posting
```

The last row is the one the blended approach got wrong: `Data Engineer` is
_contained_ in the indexed `Data Engineer Cloud Platform`, so a token-set
comparison calls it a perfect match. Token-sort scores it `63.41`, below the
threshold, and it's correctly kept as a separate opening.

Three details in that code are load-bearing:

- **`searchIter` streams in collection order and stops when you stop** — it
  takes no `limit` for that reason. For "is there _any_ duplicate?" you
  don't need the best match, so the loop returns on the first confirmed one
  and never scores the rest.
- **The normalizer is passed to `prepareChoice`, not applied by hand**, and
  the same function reference goes to the search — see
  [Prepared choices](/guides/prepared-choices/#normalizing-hand-the-function-over-dont-apply-it-yourself).
- **A threshold miss is `undefined`, not `0`**, so `companyScore === undefined`
  is the "company disagreed" branch. Passing the threshold also lets the
  scorer abandon hopeless pairs early ([Performance](/guides/performance/)).

## Choosing the thresholds

Which errors you'd rather make decides the numbers, and the two directions
are rarely symmetric. In deduplication, silently dropping a record is
usually far worse than showing one twice — so tune for **precision**: a pair
must clear every field's bar before you merge it.

Sweep candidate thresholds over a corpus you've labelled by hand, and check
what actually changes. Often less than you'd expect: if a normalizer folds
hard enough, a field's scores become effectively binary — the same employer
lands near 100, a different one far below — and every threshold in a wide
band produces identical verdicts. A number chosen from that band is a
plausible midpoint, not a tuned value, and it's worth writing down which one
you have.

## Guards first, in a generator

Some fields rule a pair out without any string work at all — a location too
far away, an incompatible seniority level, a mismatched country. Those
should run _before_ either scorer, because they cost a comparison where a
scorer costs an alignment.

The searches accept any iterable, so the natural place to put them is a
generator that yields only the candidates worth scoring:

```ts
function* nearbyAndComparable(probe) {
  for (const row of index) {
    if (row.seniority !== probe.seniority) continue
    if (distanceKm(row, probe) > MAX_KM) continue
    yield row
  }
}

for (const match of searchIter(probe.title, nearbyAndComparable(probe), {
  scorer: titleScorer,
  getPrepared: (row) => row.prepared,
  normalize: foldTitle,
  threshold: TITLE_THRESHOLD,
})) {
  // only survivors of the guards ever reach the scorer
}
```

Your policy decides what's worth scoring; the scoring pays nothing to
prepare what it accepts.

### It really is lazy

`searchIter` pulls one candidate at a time, so the guards run only for
candidates actually consumed. Returning on the first confirmed duplicate
means the rest of the collection is never guarded _and_ never scored — over
a four-posting index, a probe that matches the first candidate evaluates one
guard, while a probe that matches nothing necessarily evaluates all four.

The sorted searches share a weaker version of this: `search` and `bestMatch`
stop pulling as soon as a candidate hits the scorer's maximum, since nothing
later can beat it. With `limit: 1` over a generator, a query that scores a
perfect `100` on the first candidate consumes exactly one.

### Two things generators change

- **`key` counts the filtered stream, not your collection.** Yield items 0,
  2 and 3 through a generator and their keys come back `0`, `1`, `2`. Carry
  your own identifier on the item — `match.item.id`, not `match.key` — or
  the guards will silently renumber your records.
- **A generator is consumed once.** Searching the same generator object
  twice gives a full result set and then an empty one. Call the generator
  function per query, as above, rather than storing its result.

## Beyond two fields

The same shape extends: add a field, give it a measure, give it a bar.

- **Fail open on missing data.** If a field can't be resolved, treat it as
  "no opinion" rather than as a zero. A record with an unparseable address
  shouldn't be blocked from matching on everything else.
- **Keep the evidence.** Return the per-field scores alongside the verdict,
  not just a boolean. When someone asks why two records merged, the answer
  should be readable from a log rather than reproducible only by rerunning.
