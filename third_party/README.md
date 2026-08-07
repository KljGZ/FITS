# Third-party policy

`LOCK.json` records the exact upstream revisions audited for G-FITS Phase 0.
No upstream source is vendored in this commit and no repository is a Git
submodule yet.

When a later phase needs one of these projects:

1. fetch the repository into a directory ignored by Git;
2. checkout the exact locked commit in detached-HEAD state;
3. record the resolved commit and environment in the experiment artifact;
4. integrate through an adapter or a separately recorded patch;
5. never edit the upstream checkout in place.

A lock is a reproducibility record, not a license grant. Entries with
`NOASSERTION` remain blocked from redistribution or incorporation until their
license is manually resolved. Model weights and datasets require separate
terms review even when repository code has an SPDX license.
