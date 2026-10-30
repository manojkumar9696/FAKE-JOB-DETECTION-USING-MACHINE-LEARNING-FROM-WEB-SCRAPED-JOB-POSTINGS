# Astra MCP Server

A lightweight MCP (Model Context Protocol) server that exposes your project specs to AI tools.

## Setup

\`\`\`bash
# Run the init script which installs MCP dependencies automatically
bash specs/.devx/init.sh

# Or install MCP dependencies manually
cd specs/.devx/mcp
npm install
\`\`\`

> If the cloned repo does not preserve execute permissions, keep using \`bash specs/.devx/init.sh\` and \`bash specs/.devx/discover-workspace.sh\`.
> Local MCP dependencies under \`specs/.devx/mcp/node_modules\` are ignored by the generated \`.gitignore\` files.

## Usage

The server runs over stdio and is designed to be configured in your AI tool:

### Claude Code

Add to \`.claude/settings.json\`:
\`\`\`json
{
  "mcpServers": {
    "devx-specs": {
      "command": "node",
      "args": ["specs/.devx/mcp/server.js"]
    }
  }
}
\`\`\`

### Codex

Codex-compatible agents can read repo-level guidance from \`AGENTS.md\`.

> The \`init.sh\` script creates \`AGENTS.md\` automatically from \`project.md\` and \`workflow.md\`.

### Cursor

Add to \`.cursor/mcp.json\`:
\`\`\`json
{
  "mcpServers": {
    "devx-specs": {
      "command": "node",
      "args": ["specs/.devx/mcp/server.js"]
    }
  }
}
\`\`\`

### VS Code

Add to \`.vscode/settings.json\`:
\`\`\`json
{
  "mcp": {
    "servers": {
      "devx-specs": {
        "command": "node",
        "args": ["specs/.devx/mcp/server.js"]
      }
    }
  }
}
\`\`\`

### Kiro

Add to \`.kiro/settings/mcp.json\`:
\`\`\`json
{
  "mcpServers": {
    "devx-specs": {
      "command": "node",
      "args": ["specs/.devx/mcp/server.js"]
    }
  }
}
\`\`\`

> The \`init.sh\` script creates this file automatically. Kiro's steering context is also written to \`.kiro/steering/devx-context.md\`.

## Available Tools

| Tool | Description |
|------|-------------|
| \`list_features\` | List all features with status from features.json |
| \`get_feature_specs\` | Get specs.md content for a feature by slug or title |
| \`get_requirements\` | Get requirements.md checklist for a feature |
| \`get_tdd_tests\` | Get tdd-tests.md for a feature (if TDD enabled) |
| \`get_next_feature\` | Suggest next feature to implement (first "not-started") |
| \`validate_implementation\` | Return implementation acceptance checklist for validation |
| \`mark_feature_done\` | Update feature status in features.json and tracker.json |
| \`get_project_context\` | Return project.md + workflow.md content |
