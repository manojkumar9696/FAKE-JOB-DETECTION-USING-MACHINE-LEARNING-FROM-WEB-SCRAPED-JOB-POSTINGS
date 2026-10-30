# Development Instructions

> Last updated: 2026-10-30
> This file is regenerated during specs generation. Keep inventory updates in version control and restore any manual edits after regeneration.

## Active Configuration

This project is configured as **Monolithic** architecture.

## Architecture Mode Rules

- **Monolithic**: Existing specs-to-code flow remains unchanged. All feature specs stay under `specs/<feature-slug>/` in the implementation repo.
- **Microservices / Multi-repo**: All generated specs still live in one selected specs home repo. Other repos should reference these specs; do not fork spec sources.

## Specs Home by Delivery Order

- Specs and implementation stay in the same repository.
- Use existing spec-to-code flow without multi-repo routing rules.

## Repository Inventory

| Repository | Purpose | URL | Default Branch | Notes |
| --- | --- | --- | --- | --- |
| _Populate by running discover-workspace.sh in the target workspace_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Existing APIs

| API / Route | Method | Owning Service/Repo | Notes |
| --- | --- | --- | --- |
| _Populate by running discover-workspace.sh in the target workspace_ | _TBD_ | _TBD_ | _TBD_ |

## Existing UI Screens

| Screen | Route/Path | Owning Repo | Notes |
| --- | --- | --- | --- |
| _Populate by running discover-workspace.sh in the target workspace_ | _TBD_ | _TBD_ | _TBD_ |

## New Feature Routing Decision

When a new feature is requested in microservices mode:

1. Check this inventory and current specs to identify the nearest existing microservice.
2. Decide whether to:
   - Extend an existing microservice with additional APIs/contracts, or
   - Create a new microservice when ownership, deployability, and domain boundaries require it.
3. Record the decision in the feature `specs.md` and linked work item before implementation starts.
