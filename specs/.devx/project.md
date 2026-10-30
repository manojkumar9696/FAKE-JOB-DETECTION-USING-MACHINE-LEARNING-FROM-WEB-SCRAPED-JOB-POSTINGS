# DemoTest

> Auto-generated project context for AI-assisted development.
> Last updated: 2026-10-30

**Organization:** https://jiratest26.atlassian.net

## Development Methodology

This project follows **Spec-Driven Development (SDD)**.

Every feature has:
- `specs.md` — Full technical specification
- `requirements.md` — Implementation acceptance checklist
- `prompt.md` — Ready-to-use implementation prompt

## Features (1)

- **[Planning] Define AADSTS code inventory, JSON schema, and severity classification; [Backend] Implement AADSTS error mapping service with JSON file loader and cache** (2 user stories)

## Getting Started

1. Read this file for project context
2. Check `specs/.devx/workflow.md` for the development workflow
3. Review `specs/.devx/instruction.md` for architecture and multi-repo rules
4. Pick a feature from `specs/.devx/features.json`
5. Open the feature's `prompt.md` and use it with your AI assistant
6. Follow the spec and requirements to implement

## Project Structure

```
specs/
  .devx/
    project.md          ← You are here
    workflow.md          ← Development workflow
    features.json        ← Feature index (machine-readable)
    tracker.json         ← Code-generation execution status
    generation.json      ← Last generation metadata
    architecture.md      ← System architecture
    init.sh              ← Setup AI tool configs
  <feature-slug>/
    specs.md             ← Technical specification
    requirements.md      ← Implementation acceptance checklist
    prompt.md            ← Implementation prompt
```

## AI Tool Setup

Run the init script to configure your AI tools automatically:

```bash
bash ./specs/.devx/init.sh
```

If you want execute permissions as well:

```bash
  chmod +x ./specs/.devx/init.sh && ./specs/.devx/init.sh
```

The script lists supported AI tools, lets you choose one, and creates only that tool's config files.
