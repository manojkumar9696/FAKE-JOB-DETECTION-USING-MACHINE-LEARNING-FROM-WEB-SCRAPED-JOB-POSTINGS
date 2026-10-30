# DevX Command: implement-feature

Implement the explicitly selected feature slug.

Required input:
- `feature-slug`

Guardrails:
- Run `bash specs/.devx/validate-tracking.sh` before implementation.
- Only work inside the selected feature scope.
- Do not modify unrelated specs or DevX tooling.
- Create an implementation plan first.
- Validate every implementation acceptance checklist item before marking the feature done.
- If `requirements.md` contains spec-quality PASS/FAIL results instead of implementation acceptance criteria, stop and ask the user to regenerate or repair the spec bundle.
