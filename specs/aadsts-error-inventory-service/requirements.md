# Implementation Requirements Checklist

**Purpose**: Provide an implementation acceptance checklist that agents can execute one item at a time.  
**Feature**: [Planning] Define AADSTS code inventory, JSON schema, and severity classification; [Backend] Implement AADSTS error mapping service with JSON file loader and cache

## Functional Acceptance Criteria

- [ ] [R1] Define and provide an approved JSON schema for the externalized AADSTS mapping file with required fields `aadstsCode`, `severity`, `userMessage`, `recommendedAction`, `actionType`, and `localeKey`
- [ ] [R2] Enforce schema constraints so `aadstsCode` is a string, `severity` is an enum of `info|warning|error|critical`, `actionType` is an enum of `retry|resetPassword|completeMFA|contactIT`, `userMessage` is a string with max length 200, `recommendedAction` is a string with max length 100, and `localeKey` is a string
- [ ] [R3] Compile and deliver an AADSTS code inventory using Azure AD documentation plus the last 90 days of production authentication error logs with at least 15 distinct codes classified by severity and actionType
- [ ] [R4] Define and provide the structured error response contract containing `code`, `userMessage`, `recommendedAction`, `severityLevel`, `correlationId`, and `rawErrorDescription`
- [ ] [R5] Record and verify review/approval of the structured error response contract by frontend and backend leads where source-supported
- [ ] [R6] Implement an `ErrorMappingService` that loads the externalized mapping JSON at application startup into an in-memory dictionary keyed by `aadstsCode`
- [ ] [R7] Implement `resolve(errorPayload)` to extract the primary AADSTS code from either an `error_codes` array or an `error_description` string using the pattern `AADSTS[0-9]+`
- [ ] [R8] Return the mapped structured error object for known codes with the resolved `code`, mapped `userMessage`, mapped `recommendedAction`, mapped `severityLevel`, and source `correlationId`
- [ ] [R9] Implement and document fallback behavior for unmapped codes so the returned object includes the raw AADSTS code, the source Azure `correlation_id`, and fallback `userMessage` exactly `Authentication failed. Please contact IT Support.`
- [ ] [R10] Ensure all inventoried codes are resolvable through `resolve()` with the expected mapped outputs
- [ ] [R11] Cover primary path, alternate parsing path, unknown-code fallback path, malformed-entry skip path, and hot-reload path in implementation and verification

## UI Acceptance Criteria

- [ ] [R12] Ensure the structured error response contract exposed for frontend consumption contains all source-supported fields required for user-facing auth error presentation, including `rawErrorDescription` when included by the contract
- [ ] [R13] Keep user-facing strings within the specified schema limits so mapped `userMessage` does not exceed 200 characters and `recommendedAction` does not exceed 100 characters
- [ ] [R14] Preserve `localeKey` in the mapping schema for future internationalization without requiring current localized rendering behavior not specified by source
- [ ] [R15] Do not implement new screens, UI flows, or presentation behavior beyond the reviewed response contract because no source-supported UI changes are specified

## API and Integration Acceptance Criteria

- [ ] [R16] Load mappings from an external JSON file rather than hard-coding AADSTS mappings in application code
- [ ] [R17] Validate the JSON mapping file against the approved schema on initial load and on each reload attempt
- [ ] [R18] If a single mapping entry is malformed, log a WARNING with entry details, skip only that entry, and continue serving valid entries from the same file
- [ ] [R19] Implement a configurable hot-reload mechanism for mapping updates without application restart
- [ ] [R20] Support a polling-based hot-reload interval with default 60 seconds where configured by source
- [ ] [R21] Do not implement an unapproved config endpoint for hot-reload unless an explicit product decision resolves the hot-reload mechanism choice
- [ ] [R22] Log successful mapping reload events at INFO level
- [ ] [R23] Preserve backward compatibility of the response object fields named in source unless an explicit approved change is made to the reviewed contract

## Business Logic and Data Acceptance Criteria

- [ ] [R24] Classify each inventoried AADSTS code with one of the allowed severities `info|warning|error|critical`
- [ ] [R25] Assign each inventoried AADSTS code one of the allowed action types `retry|resetPassword|completeMFA|contactIT`
- [ ] [R26] Store mappings in an in-memory hash/dictionary structure keyed by `aadstsCode` to satisfy constant-time lookup expectations
- [ ] [R27] Extract the primary AADSTS code by first checking structured source data when present and also parsing free-text descriptions when needed
- [ ] [R28] Include the Azure `correlation_id` from the original error payload in the resolved structured response
- [ ] [R29] Sanitize `error_description` using HTML entity encoding before placing it into `rawErrorDescription`
- [ ] [R30] Ensure fallback behavior is applied when no mapped code is found after all supported extraction attempts
- [ ] [R31] Ensure malformed mapping entries do not poison the full cache and that valid cached mappings remain functional after partial load failures
- [ ] [R32] Define how `severity` from the mapping file maps to response `severityLevel` and verify the implementation uses that mapping consistently
- [ ] [R33] Do not assume undocumented precedence rules when multiple AADSTS codes are present in `error_codes` and `error_description`; if unresolved, hold completion pending clarification rather than implementing silently

## Non-Functional Acceptance Criteria

- [ ] [R34] Meet the source-supported lookup performance expectation so resolved mapping lookup completes in under 5 ms by using an in-memory hash map
- [ ] [R35] Prevent XSS exposure by HTML-encoding raw error description content before returning it in structured responses
- [ ] [R36] Ensure startup and reload behavior are reliable so invalid entries are isolated and valid mappings continue to function
- [ ] [R37] Emit INFO and WARNING logs for reload and malformed-entry scenarios respectively, with sufficient detail for diagnosis
- [ ] [R38] Keep the implementation within the selected monolith architecture style
- [ ] [R39] Do not introduce unrelated domains, infrastructure, endpoints, or persistence stores not required by this feature
- [ ] [R40] Provide unit tests or equivalent verification for known mapped codes, unknown unmapped codes, malformed entries, regex extraction behavior, startup load, and hot-reload scenarios
- [ ] [R41] Verify hot-reload updates cached mappings within the configured polling interval without restart

## Traceability

- [ ] [R42] Every implemented change maps back to the planning story requirements for code inventory, schema, severity classification, response contract, fallback behavior, and hot-reload specification
- [ ] [R43] Every implemented change maps back to the backend story requirements for startup file load, cached lookup, code extraction, fallback response, sanitization, schema validation, malformed-entry handling, logging, hot-reload, and performance
- [ ] [R44] Every non-blocking Open Question that was implemented has a recorded decision + one-line rationale in assumptions documentation (no Open Question is silently assumed)
- [ ] [R45] No BLOCKING Open Question was implemented as an assumption; unresolved blocking choices such as the definitive hot-reload mechanism or primary-code precedence must hold the feature at needs-clarification if not source-resolved

## Assumptions

### ASM-1
- [R46] Statement: The externalized AADSTS mapping file is a JSON collection of mapping entries rather than another JSON shape.
- [R47] Source: User stories require a JSON mapping file schema with per-entry fields and a loader that validates entries and skips malformed single entries.
- [R48] Rationale: A collection-of-entries structure is the minimal shape consistent with per-entry validation, inventory loading, and keyed dictionary caching.
- [R49] Potential impact: If a different top-level JSON structure is later required, loader and schema definitions may need refactoring.
- [R50] Confidence: medium
- [R51] Recommended resolution: Confirm and document the exact top-level JSON structure in the approved schema artifact.

### ASM-2
- [R52] Statement: The fallback structured response includes `rawErrorDescription` in addition to the explicitly listed fallback fields because the contract defines that field and sanitization is required before inclusion.
- [R53] Source: Planning story defines response contract with `rawErrorDescription`; backend story requires sanitizing `error_description` before including it in `rawErrorDescription`.
- [R54] Rationale: The sanitization requirement implies the fallback/resolve output may carry `rawErrorDescription` whenever source error description data exists.
- [R55] Potential impact: If consumers do not expect this field in some paths, response handling or contract tests may need adjustment.
- [R56] Confidence: high
- [R57] Recommended resolution: Confirm whether `rawErrorDescription` is mandatory on all resolve responses or only when source data is present.

### ASM-3
- [R58] Statement: The response field `severityLevel` is populated directly from the mapping entry `severity` value.
- [R59] Source: Planning story defines schema field `severity`; response contract defines field `severityLevel`; backend story returns structured error object containing `severityLevel`.
- [R60] Rationale: No alternate transformation is specified, so direct value propagation is the narrowest implementation consistent with source.
- [R61] Potential impact: If a different display or API normalization is required, mapping logic and tests will need revision.
- [R62] Confidence: medium
- [R63] Recommended resolution: Document the exact mapping between schema `severity` and response `severityLevel` in the approved contract.

## Notes

- [R64] Never resolve an Open Question silently. In an unattended run, record the chosen assumption + rationale in project assumptions documentation; blocking questions must instead hold the feature at needs-clarification.
- [R65] Mark an item complete only after verifying actual implementation code and behavior.