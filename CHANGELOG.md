# Release notes

## 0.6.0 — prepared, not published

`check()` now reads task-specific operational evidence by default. A positive vote score alone no longer produces a trusted verdict. Existing result keys remain; `vote_verdict` preserves the previous vote interpretation. Missing, partial or malformed evidence stays unknown, with warnings. Known observed failures still warn.

`search()` retains its list response. `search_details()` preserves full coverage guidance. Service failures propagate instead of appearing as empty healthy results. Public contribution is optional and needs actual execution evidence and existing publication authorization; registration is access, not authorization.

Migration: consumers that need the old vote-only interpretation should explicitly use `vote_verdict`, and handle structured errors and unknown evidence. The evidence aggregation backend must be deployed before evaluating the new contributor fields. No outcome is automatically published by checks.
