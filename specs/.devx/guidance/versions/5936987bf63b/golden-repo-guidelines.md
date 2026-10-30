# Golden Repository Implementation Guidelines

Generated for: DemoTest
Repository: mridul227/testingastra
Generated at: 2026-10-30T10:05:43.866Z
File selection mode: all vectorized files

## Warnings

- None

## How To Use

- Read this file before implementation in the IDE.
- Use `golden-repo-sources.md` for exact source excerpts and file-path attribution.
- Use Golden Repo guidance for conventions, patterns, validation expectations, architecture style, testing expectations, and implementation constraints.
Golden Repo grounding rules:
- If a selected Golden Repo reference states a rule, constraint, term, or pattern that applies to this feature, the spec must reflect it and cite that file path in Source References.
- Do not force unrelated Golden Repo concepts into the specification. If a reference does not apply to this feature, leave it out.
- Do not invent implementation details that are absent from the Golden Repo or the requirement. Exact endpoints, hosts, schemas, tables, queue names, and infrastructure components that are not stated must be marked Validate or left as Open Questions.
- Also incorporate applicable UI design guidelines, architecture guidelines, policies, business rules, standards, and validation expectations when they apply to this feature.
- If a relevant Golden Repo rule applies but the feature source lacks a product decision, write the gap under Open Questions instead of inventing behavior.
- If a user story conflicts with a Golden Repo example, follow the user story for what to build and keep only the Golden Repo terminology and rules that still apply.
- Cite the Golden Repo file paths of the references you actually used in Source References.
- Existing code patterns in the target workspace override generic Golden Repo guidance when they conflict on stack mechanics.
- Do not require live Golden Repository access inside the implementation workspace.

## Extracted Guidance

GOLDEN REPO KNOWLEDGE CONTEXT (spec generation)
Repository: mridul227/testingastra
The following is a distilled, de-duplicated consolidation of Golden Repository conventions. Use a convention only when it is materially relevant to the requested feature. Do not force unrelated concepts into the specification. Do not invent endpoints, hosts, schemas, tables, or queue names that this consolidation does not state; mark those as Validate.

## Architecture & System Design

### Manifest Provider Configuration

- [G1] The repository/source manifest must use **Jira** as the provider.

### Project Identification

- [G2] Use the following project identifiers in the manifest:
  - `projectKey`: `QAS310826`
  - `projectName`: `qas310826`

### Manifest Schema Requirements

- [G3] The manifest must include the following repository/project-level metadata fields:
  - `projectKey`
  - `projectName`
  - `provider`
  - `counts`
  - `artifactKeys`
  - `documentPaths`

## API & Integration Conventions

### Source System Identification

- [G4] Set the source provider to `jira`.
- [G5] Use project key `QAS310826`.
- [G6] Use project name `qas310826`.

### Manifest Schema

- [G7] The manifest must include these fields:
  - `provider`
  - `projectKey`
  - `projectName`
  - `artifactKeys`
  - `counts`
  - `warnings`
  - `truncation`
  - `extractedAt`
  - `documentPaths`

### Manifest Field Conventions

- [G8] In the provided manifest excerpt, `documentPaths` is an empty list.

## Data Model & Validation

### Manifest Schema

- [G9] The manifest includes these fields:
  - `provider`
  - `projectKey`
  - `projectName`
  - `extractedAt`
  - `counts`
  - `artifactKeys`
  - `warnings`
  - `truncation`
  - `documentPaths`

### Field Types and Value Shapes

- [G10] `counts` is present in the manifest and contains artifact totals.
- [G11] `warnings` is an array.
  - In the excerpt, `warnings` is empty.
- [G12] `truncation` is an array.
  - In the excerpt, `truncation` is empty.
- [G13] `documentPaths` is an array.
  - In the excerpt, `documentPaths` is empty.
- [G14] `artifactKeys` is an array of string IDs.
  - IDs are not numeric values.

### Timestamp Validation

- [G15] `extractedAt` is an ISO-8601 timestamp string.
  - Example: `2026-09-07T12:13:04.094Z`

### Counts Validation

- [G16] The manifest `counts` includes these artifact totals:
  - `epic`: 15
  - `task`: 1520
  - `story`: 98
  - `feature`: 48
  - `releases`: 0
  - `documents`: 0

### Artifact Key Format

- [G17] Artifact keys use Jira-style identifier format.
- [G18] IDs follow the pattern `QAS310826-<number>`.
- [G19] IDs are hyphenated strings with:
  - a fixed prefix: `QAS310826-`
  - a numeric suffix
- [G20] The manifest data appears to be a large list of string IDs in `QAS310826-<number>` format.

### Duplicate Handling

- [G21] The excerpt contains repeated duplicate IDs.
- [G22] No uniqueness constraint is stated in the excerpt for `artifactKeys`.

## Testing & Quality Expectations

### Manifest Quality Expectations

- [G23] Manifest reports must include the following counts:
  - `epic`: 15
  - `task`: 1520
  - `story`: 98
  - `feature`: 48
  - `releases`: 0
  - `documents`: 0

### Required Manifest Fields

- [G24] The manifest must include a `warnings` field.
- [G25] The manifest must include a `truncation` field.

### Validation Expectations

- [G26] `warnings` must be empty.
- [G27] `truncation` must be empty.

## Domain & Business Rules

### Repository Metadata

- [G28] Provider: `jira`
- [G29] Project key: `QAS310826`
- [G30] Project name: `qas310826`

### Artifact Inventory Counts

- [G31] The manifest records counts for these artifact types:
  - `epic`: `15`
  - `task`: `1520`
  - `story`: `98`
  - `feature`: `48`
  - `releases`: `0`
  - `documents`: `0`

### Document Presence

- [G32] `documentPaths` is empty in this excerpt.
- [G33] Document count is `0`.

### Source Excerpt Constraint

- [G34] No implementation guidance is present in the excerpt; it is a manifest-like list of item IDs and counts only.

## Naming, Structure & Code Patterns

### Manifest Structure

- [G35] Represent source metadata as a manifest-style JSON structure.
- [G36] Source references are grouped under `_source-manifest.json > Introduction`.
- [G37] The manifest includes these top-level fields:
  - `counts`
  - `provider`
  - `warnings`
  - `projectKey`
  - `truncation`
  - `extractedAt`
  - `projectName`
  - `artifactKeys`
  - `documentPaths`

### Project Metadata Conventions

- [G38] Set `provider` to `jira`.
- [G39] Set `projectKey` to `QAS310826`.
- [G40] Set `projectName` to lowercase `qas310826`.

### Identifier Naming Pattern

- [G41] Artifact/item/document identifiers use the pattern `QAS310826-<number>`.
- [G42] Use the project key prefix plus numeric suffix consistently throughout: `QAS310826-<number>`.
- [G43] Represent identifiers as quoted string values.
- [G44] Manifest entries use repeated arrays/lists of quoted IDs in this format.

### Collection and Field Patterns

- [G45] Represent artifact collections as `artifactKeys` arrays.
- [G46] The manifest contains a top-level `documentPaths` collection.
- [G47] `documentPaths` is an array field and may be empty; in this excerpt it is `[]`.
- [G48] `warnings` is an array and is empty in this excerpt: `[]`.
- [G49] `truncation` is an array and is empty in this excerpt: `[]`.

### Counts and Artifact Categories

- [G50] The `counts` object includes:
  - `releases`
  - `documents`

- [G51] Artifact types present:
  - `epic`
  - `task`
  - `story`
  - `feature`

- [G52] Artifact categories referenced:
  - `epic`
  - `task`
  - `story`
  - `feature`
  - `releases`
  - `documents`

## Error Handling & Resilience

### Manifest Error and Truncation Fields

- [G53] The manifest includes both of the following arrays:
  - `warnings`
  - `truncation`

- [G54] In the provided manifest excerpt:
  - `warnings` is an empty array.
  - `truncation` is an empty array.

## Performance & Observability

### Extraction Metadata

- [G55] Record an extraction timestamp in `extractedAt`.
  - Example value: `2026-09-07T12:13:04.094Z`.
- [G56] Include extraction metadata collections:
  - `warnings`
  - `truncation`
- [G57] `warnings` is an empty array in this excerpt.
- [G58] `truncation` appears as an empty array in this excerpt, but source metadata also shows a specific truncation marker example: `QAS310826-1427`.

### Aggregate Counts

- [G59] Track aggregate counts by artifact type in the manifest/repository metadata.
- [G60] The excerpt records these counts:
  - `epic: 15`
  - `task: 1520`
  - `story: 98`
  - `feature: 48`
  - `releases: 0`
  - `documents: 0`

### Source/Provider Metadata

- [G61] Record the source provider as `jira`.
- [G62] Include `documentPaths`; this excerpt shows `documentPaths: []`.