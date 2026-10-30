# DevX Command: validate-feature

Validate the explicitly selected feature slug against its generated requirements.

Required input:
- `feature-slug`

Output:
- A pass/fail report for each requirement.
- A short list of missing or partial implementation gaps.
- A recommendation on whether the feature can be marked `done`.

Before validation, run `bash specs/.devx/validate-tracking.sh`.
