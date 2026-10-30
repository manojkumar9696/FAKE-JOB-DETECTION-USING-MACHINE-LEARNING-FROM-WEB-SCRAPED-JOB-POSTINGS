#!/usr/bin/env node
/**
 * Astra MCP Server — Exposes project specs to AI tools via the Model Context Protocol.
 *
 * This is a self-contained stdio MCP server. Run it with:
 *   node server.js
 *
 * It reads from the specs/ folder relative to the repo root and exposes
 * tools for listing features, reading specs, managing status, etc.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { resolve } from "path";
import { getRepoRoot } from "../../../utils/module-paths";

// Resolve paths relative to repo root (specs/.devx/mcp/ -> repo root)
const REPO_ROOT = getRepoRoot();
const SPECS_DIR = resolve(REPO_ROOT, "specs");
const DEVX_DIR = resolve(SPECS_DIR, ".devx");
const FEATURES_JSON = resolve(DEVX_DIR, "features.json");
const TRACKER_JSON = resolve(DEVX_DIR, "tracker.json");

// ── Helpers ──

function readJson(filePath) {
  if (!existsSync(filePath)) return null;
  return JSON.parse(readFileSync(filePath, "utf-8"));
}

function readText(filePath) {
  if (!existsSync(filePath)) return null;
  return readFileSync(filePath, "utf-8");
}

function getFeaturesData() {
  const data = readJson(FEATURES_JSON);
  if (!data || !data.features) {
    throw new Error("features.json not found or invalid. Run specs generation first.");
  }
  return data;
}

function getTrackerData() {
  const data = readJson(TRACKER_JSON);
  if (data && data.tasks) return data;
  return { trackerVersion: 1, updatedAt: new Date().toISOString(), tasks: {} };
}

function findFeature(query) {
  const data = getFeaturesData();
  const q = query.toLowerCase().trim();
  return data.features.find(
    (f) =>
      f.slug === q ||
      f.title.toLowerCase() === q ||
      f.slug === q.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") ||
      f.id.toString() === q
  );
}

// ── Server Setup ──

const server = new Server(
  { name: "devx-specs", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// ── Tool Definitions ──

const TOOLS = [
  {
    name: "list_features",
    description: "List all features with their status, slug, and story count from features.json.",
    inputSchema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "get_feature_specs",
    description: "Get the full specs.md content for a feature by slug, title, or ID.",
    inputSchema: {
      type: "object",
      properties: {
        feature: { type: "string", description: "Feature slug, title, or ID" },
      },
      required: ["feature"],
    },
  },
  {
    name: "get_requirements",
    description: "Get the requirements.md implementation acceptance checklist for a feature.",
    inputSchema: {
      type: "object",
      properties: {
        feature: { type: "string", description: "Feature slug, title, or ID" },
      },
      required: ["feature"],
    },
  },
  {
    name: "get_tdd_tests",
    description: "Get the tdd-tests.md content for a feature (if TDD is enabled).",
    inputSchema: {
      type: "object",
      properties: {
        feature: { type: "string", description: "Feature slug, title, or ID" },
      },
      required: ["feature"],
    },
  },
  {
    name: "get_next_feature",
    description: "Suggest the next feature to implement (first one with status 'not-started').",
    inputSchema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "validate_implementation",
    description: "Return the implementation acceptance checklist for a feature so you can validate your implementation against it.",
    inputSchema: {
      type: "object",
      properties: {
        feature: { type: "string", description: "Feature slug, title, or ID" },
      },
      required: ["feature"],
    },
  },
  {
    name: "mark_feature_done",
    description: "Update a feature's status in features.json to 'done'.",
    inputSchema: {
      type: "object",
      properties: {
        feature: { type: "string", description: "Feature slug, title, or ID" },
      },
      required: ["feature"],
    },
  },
  {
    name: "get_project_context",
    description: "Return the full project.md and workflow.md content for project-level context.",
    inputSchema: { type: "object", properties: {}, required: [] },
  },
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS,
}));

// ── Tool Handlers ──

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "list_features": {
        const data = getFeaturesData();
        const summary = data.features.map((f) => ({
          id: f.id,
          title: f.title,
          slug: f.slug,
          status: f.status,
          storyCount: f.storyCount,
          totalStoryPoints: f.totalStoryPoints,
        }));
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                { totalFeatures: data.totalFeatures, enableTdd: data.enableTdd, features: summary },
                null,
                2
              ),
            },
          ],
        };
      }

      case "get_feature_specs": {
        const feature = findFeature(args.feature);
        if (!feature) {
          return { content: [{ type: "text", text: `Feature not found: ${args.feature}` }], isError: true };
        }
        const specsPath = resolve(REPO_ROOT, feature.files.specs);
        const content = readText(specsPath);
        if (!content) {
          return { content: [{ type: "text", text: `specs.md not found at ${feature.files.specs}` }], isError: true };
        }
        return { content: [{ type: "text", text: content }] };
      }

      case "get_requirements": {
        const feature = findFeature(args.feature);
        if (!feature) {
          return { content: [{ type: "text", text: `Feature not found: ${args.feature}` }], isError: true };
        }
        const reqPath = resolve(REPO_ROOT, feature.files.requirements);
        const content = readText(reqPath);
        if (!content) {
          return { content: [{ type: "text", text: `requirements.md not found at ${feature.files.requirements}` }], isError: true };
        }
        return { content: [{ type: "text", text: content }] };
      }

      case "get_tdd_tests": {
        const feature = findFeature(args.feature);
        if (!feature) {
          return { content: [{ type: "text", text: `Feature not found: ${args.feature}` }], isError: true };
        }
        if (!feature.files.tddTests) {
          return { content: [{ type: "text", text: `TDD is not enabled for feature: ${feature.title}` }], isError: true };
        }
        const tddPath = resolve(REPO_ROOT, feature.files.tddTests);
        const content = readText(tddPath);
        if (!content) {
          return { content: [{ type: "text", text: `tdd-tests.md not found at ${feature.files.tddTests}` }], isError: true };
        }
        return { content: [{ type: "text", text: content }] };
      }

      case "get_next_feature": {
        const data = getFeaturesData();
        const next = data.features.find((f) => f.status === "not-started");
        if (!next) {
          return { content: [{ type: "text", text: "All features are done! No remaining not-started features." }] };
        }
        return {
          content: [
            {
              type: "text",
              text: [
                `## Next Feature: ${next.title}`,
                ``,
                `- **Slug:** ${next.slug}`,
                `- **Stories:** ${next.storyCount}`,
                `- **Story Points:** ${next.totalStoryPoints}`,
                ``,
                `### Files`,
                `- Specs: ${next.files.specs}`,
                `- Requirements: ${next.files.requirements}`,
                next.files.tddTests ? `- TDD Tests: ${next.files.tddTests}` : null,
                `- Prompt: ${next.files.prompt}`,
                ``,
                `Use get_feature_specs or get_requirements to read the details. requirements.md must be an implementation acceptance checklist, not a spec-quality PASS/FAIL report.`,
              ]
                .filter(Boolean)
                .join("\n"),
            },
          ],
        };
      }

      case "validate_implementation": {
        const feature = findFeature(args.feature);
        if (!feature) {
          return { content: [{ type: "text", text: `Feature not found: ${args.feature}` }], isError: true };
        }
        const reqPath = resolve(REPO_ROOT, feature.files.requirements);
        const content = readText(reqPath);
        if (!content) {
          return { content: [{ type: "text", text: `requirements.md not found at ${feature.files.requirements}` }], isError: true };
        }
        return {
          content: [
            {
              type: "text",
              text: [
                `# Validation Checklist — ${feature.title}`,
                ``,
                `Go through each implementation acceptance item below and verify it is satisfied in the implementation.`,
                `If this content is a spec-quality PASS/FAIL report, stop and regenerate or repair the spec bundle before validating.`,
                ``,
                content,
              ].join("\n"),
            },
          ],
        };
      }

      case "mark_feature_done": {
        const data = getFeaturesData();
        const tracker = getTrackerData();
        const featureIndex = data.features.findIndex(
          (f) =>
            f.slug === (args.feature || "").toLowerCase().trim() ||
            f.title.toLowerCase() === (args.feature || "").toLowerCase().trim() ||
            f.id.toString() === (args.feature || "").trim()
        );
        if (featureIndex === -1) {
          return { content: [{ type: "text", text: `Feature not found: ${args.feature}` }], isError: true };
        }
        const now = new Date().toISOString();
        const feature = data.features[featureIndex];
        const taskId = feature.taskId || `FEATURE-${feature.id}`;
        data.features[featureIndex].status = "done";
        tracker.tasks[taskId] = {
          ...(tracker.tasks[taskId] || {}),
          status: "COMPLETED",
          title: feature.title,
          featureId: feature.id,
          slug: feature.slug,
          updatedAt: now,
        };
        tracker.trackerVersion = Number(tracker.trackerVersion || 1) + 1;
        tracker.updatedAt = now;
        writeFileSync(FEATURES_JSON, JSON.stringify(data, null, 2), "utf-8");
        writeFileSync(TRACKER_JSON, JSON.stringify(tracker, null, 2), "utf-8");
        return {
          content: [
            {
              type: "text",
              text: `Marked "${data.features[featureIndex].title}" as done in features.json and COMPLETED in tracker.json.`,
            },
          ],
        };
      }

      case "get_project_context": {
        const projectMd = readText(resolve(DEVX_DIR, "project.md")) || "project.md not found.";
        const workflowMd = readText(resolve(DEVX_DIR, "workflow.md")) || "workflow.md not found.";
        return {
          content: [
            {
              type: "text",
              text: projectMd + "\n\n---\n\n" + workflowMd,
            },
          ],
        };
      }

      default:
        return { content: [{ type: "text", text: `Unknown tool: ${name}` }], isError: true };
    }
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error: ${error.message || error}` }],
      isError: true,
    };
  }
});

// ── Start ──

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error("MCP server failed to start:", error);
  process.exit(1);
});
