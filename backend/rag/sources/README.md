# Local medical source corpus

This directory contains downloaded public documents and their cleaned Markdown derivatives. `source_manifest.json` is the inclusion registry; only entries marked `source_curated` are indexed. Raw snapshots are retained under `raw/`, and `source_snapshot_manifest.json` records URLs, retrieval times, content types, and SHA-256 hashes.

## Current batch

- National Health Commission: 2024 cerebrovascular disease prevention guideline, 2024 obesity diagnosis and treatment guideline, 2025 pediatric *Mycoplasma pneumoniae* pneumonia guideline, and WS/T 872-2025 adult hypertension screening and diagnosis standard.
- National Healthcare Security Administration and Ministry of Human Resources and Social Security: 2025 basic medical insurance drug catalogue, effective 2026-01-01.
- National Healthcare Security Administration and Ministry of Human Resources and Social Security: 2025 commercial health insurance innovative-drug catalogue. This is separate from basic medical insurance.

The catalogues describe insurance payment scope, not drug labeling, clinical indications, individualized reimbursement, or dosing advice. The corpus does not yet contain a comprehensive, verified set of current package inserts or copyrighted textbook chapters. NMPA's label-query endpoints were not retrievable in this collection run; paid/copyrighted textbooks were not copied. Add those only from authorized or openly licensed sources, preserving product approval number, manufacturer, revision date, and source URL.

WHO snapshots and the earlier manually written summaries remain on disk for provenance/history, but their manifest status is `do_not_use` and they are excluded from the active index.
