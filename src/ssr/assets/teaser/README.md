# Teaser demo assets

`input.jpg` is the contributed photo of Old College used in the
project webpage's animated teaser. `search_results.json` records the first five
previously retrieved results, in their original order, from September 21, 2026.
Each result includes its title, original page URL, source, thumbnail URL and
bundled local thumbnail filename. No new search was performed for this release.

The manifest is matched by the SHA-256 of `input.jpg`. Input resizing or
re-encoding changes that hash; record a new manifest rather than silently
returning evidence for an unrecognized input.

The code is MIT licensed. Third-party titles and thumbnail images retain their
original rights and are included as attributed search observations, not as
MIT-licensed original artwork. No raw API responses or credentials are bundled.
