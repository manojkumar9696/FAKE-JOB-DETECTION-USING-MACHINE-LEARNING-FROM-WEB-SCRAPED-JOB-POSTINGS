# Feature: [Planning] Define AADSTS code inventory, JSON schema, and severity classification; [Backend] Implement AADSTS error mapping service with JSON file loader and cache
Status: NEW
Owner: Astra
Last Updated: 2026-10-30

## Summary
This feature defines and implements a centralized AADSTS error mapping capability so Azure AD authentication failures can be translated into structured, user-facing error responses.

The feature has two coordinated outcomes:

1. Planning deliverables:
   - Compile a production-informed inventory of AADSTS codes by auditing Azure AD documentation and the last 90 days of production authentication error logs.
   - Classify each inventoried code by severity and action type.
   - Define the JSON mapping file schema and the structured error response contract.
   - Document fallback behavior for unmapped codes and the hot-reload approach.

2. Backend deliverables:
   - Build an `ErrorMappingService` that loads the externalized JSON mapping file into memory at application startup.
   - Resolve Azure AD error payloads to structured error objects using cached lookups.
   - Support unmapped-code fallback behavior, malformed-entry handling, schema validation, and hot-reload of mapping changes without application restart.

The expected outcome is a backend service that consistently returns mapped, sanitized, structured authentication error information with low-latency lookups and operational support for updating mappings externally.

## Scope
### In Scope
- Auditing Azure AD documentation and the last 90 days of production authentication error logs to compile an AADSTS code inventory.
- Defining severity classifications for inventoried codes using:
  - `info`
  - `warning`
  - `error`
  - `critical`
- Assigning an `actionType` to each inventoried code using:
  - `retry`
  - `resetPassword`
  - `completeMFA`
  - `contactIT`
- Designing the JSON mapping file schema with the following fields:
  - `aadstsCode` (string)
  - `severity` (enum)
  - `userMessage` (string, max 200 chars)
  - `recommendedAction` (string, max 100 chars)
  - `actionType` (enum)
  - `localeKey` (string for future i18n)
- Defining the structured error response object contract.
- Documenting fallback behavior when an AADSTS code is not found in the mapping.
- Specifying a hot-reload mechanism using either file-watcher or polling/config endpoint as supported by the source.
- Implementing `ErrorMappingService`.
- Loading the externalized mapping JSON file at application startup.
- Maintaining an in-memory dictionary keyed by `aadstsCode`.
- Implementing `resolve(errorPayload)` parsing logic using:
  - Azure AD `error_codes` array
  - Azure AD `error_description` string
  - regex pattern `AADSTS[0-9]+`
- Returning mapped structured error objects for known codes.
- Returning fallback structured error objects for unknown codes.
- Sanitizing `error_description` using HTML entity encoding before inclusion in `rawErrorDescription`.
- Validating JSON against schema on load.
- Skipping malformed entries while continuing to use valid entries.
- Logging reload events at INFO level.
- Meeting lookup performance target of under 5 ms using a hash map.
- Unit test coverage for known codes, unknown codes, malformed entries, and hot-reload scenarios.

### Out of Scope
- Any UI screen implementation.
- Definition of API endpoints, transport methods, or external service hosts.
- Frontend rendering behavior beyond review/approval of the structured response contract.
- Internationalization implementation beyond inclusion of `localeKey` for future use.
- Analytics, telemetry schemas, dashboards, or alerting behavior not stated in source.
- Persistence design beyond the externalized JSON mapping file and in-memory cache.
- Authorization model changes not stated in source.

## Application Type & Platform Context
The feature targets a mixed application context.

### Source Evidence
- Derived Source Signals: `Application Type: mixed`
- Planning deliverables require response contract review by frontend and backend leads.
- Backend story explicitly requires implementation of a backend service.
- Hot-reload is described as `file-watcher or config endpoint`, indicating backend/service behavior with possible integration touchpoints.

### Platform Context
- Backend/service logic is explicitly in scope.
- Frontend dependency exists only as contract consumer/reviewer of the structured response object.

### Open Question
- Validate whether this service is consumed only within the monolith process or also exposed through an application API boundary.

## Actors and Permissions
### Actors
- Backend developer/maintainer: implements and maintains `ErrorMappingService`.
- Frontend lead: reviews and approves the structured error response contract.
- Backend lead: reviews the response object contract.
- Operations/supporting maintainer: updates the externalized JSON mapping file and relies on hot-reload behavior.
- Authentication flow/system component: invokes `resolve(errorPayload)` with Azure AD error payloads.

### Permissions and Access Constraints
Source-supported access expectations:
- The application must be able to read the externalized JSON mapping file at startup.
- The application must detect mapping file changes and hot-reload without restart.
- Logging must occur for reload events and malformed entries.

### Open Questions
- Who is permitted to modify the mapping file in production.
- Whether config-endpoint-based reload is allowed in addition to file-based reload in the target environment.
- Whether any administrative approval workflow is required before new mappings are deployed.

## Feature Development Intent
This is feature-development work because both planning artifacts and executable backend behavior must be created.

The planning work must produce the authoritative classification and contract inputs needed by development:
- A complete inventoried set of encountered AADSTS codes with at least 15 classified entries.
- An approved JSON schema document.
- A reviewed structured error response contract.
- Documented fallback and reload behavior.

The backend work must implement the runtime capability that uses those definitions:
- Load the externalized mapping file into cache at startup.
- Parse Azure AD error payloads to identify the primary AADSTS code.
- Resolve known codes into structured user-facing responses.
- Return a safe fallback for unknown codes.
- Support reload of mapping changes without restart.
- Maintain resilience when some mapping entries are malformed.

The outcome to be delivered is a low-latency, schema-validated, externally configurable AADSTS mapping service suitable for authentication error handling within the monolith.

## UI Design & Interaction Contract
No end-user screen, layout, navigation, or interaction design is defined in the source.

### Source-Supported UI Contract
- The structured error response contract must be reviewed and approved by the frontend lead.
- `userMessage` and `recommendedAction` are intended for user-facing presentation.
- `localeKey` must be included in the mapping schema for future internationalization support.

### Copy Contract
- For unmapped codes, the fallback `userMessage` must be:
  - `Authentication failed. Please contact IT Support.`

### Open Questions
- Whether frontend requires `actionType` in the runtime structured response object, since it is required in mapping and fallback behavior but omitted from one backend return-object statement.
- Whether `severityLevel` values shown to frontend must match mapping `severity` values exactly or require transformation.
- Whether `rawErrorDescription` should be returned to frontend in all cases, known-code cases only, unknown-code cases only, or server-side only.
- Whether frontend has any display rules tied to severity or action type.

## API Contract
No external API endpoint contract is defined in the source. The source defines a service-level method contract and a structured response object.

### Service Operation
#### `resolve(errorPayload)`
Resolves an Azure AD error payload into a structured error object.

### Inputs
Source-supported input content includes Azure AD error payload data containing:
- `error_codes` array
- `error_description` string
- `correlation_id`

### Parsing Contract
The service must extract the primary AADSTS code by parsing:
- the Azure AD `error_codes` array, and/or
- the Azure AD `error_description` string using regex pattern `AADSTS[0-9]+`

### Outputs
#### Planned structured error response object contract
```json
{
  "code": "string",
  "userMessage": "string",
  "recommendedAction": "string",
  "severityLevel": "string",
  "correlationId": "string",
  "rawErrorDescription": "string"
}
```

#### Backend mapped return object
Source states mapped return object:
```json
{
  "code": "string",
  "userMessage": "string",
  "recommendedAction": "string",
  "severityLevel": "string",
  "correlationId": "string"
}
```

#### Unknown-code fallback return behavior
For unmapped codes, the service must:
- return fallback `userMessage` = `Authentication failed. Please contact IT Support.`
- return `actionType` = `contactIT`
- include the raw AADSTS code
- include Azure `correlation_id` from the original response
- sanitize `error_description` using HTML entity encoding before including in `rawErrorDescription`

### Errors and Invalid Data Handling
- Mapping JSON must be validated against schema on load.
- If a single entry is malformed:
  - log a WARNING
  - include entry details in the log
  - skip that entry
  - continue loading/using valid entries

### Idempotency
- `resolve(errorPayload)` is expected to be read-only and deterministic for a given payload and current cache state.

### Integration Behavior
- Mapping file must be loaded at application startup.
- Hot-reload must occur through file-watcher or polling mechanism.
- Polling interval must be configurable with a default of 60 seconds.
- Cache reload must occur without application restart.

### Open Questions
- Confirm the authoritative runtime response contract: whether `rawErrorDescription` and `actionType` are always included, conditionally included, or omitted for known mappings.
- Confirm precedence rules when both `error_codes` array and `error_description` contain different AADSTS codes.
- Confirm behavior when no AADSTS code can be extracted at all.
- Confirm whether schema validation failure for the entire file should block startup or allow startup with empty/previous cache.

## Business Logic & Rules
- The AADSTS inventory must be compiled from:
  - Azure AD documentation
  - the last 90 days of production authentication error logs
- The inventory must contain at least 15 AADSTS codes.
- Each inventoried code must be assigned:
  - one severity value from `info | warning | error | critical`
  - one action type from `retry | resetPassword | completeMFA | contactIT`
- The mapping file is externalized and loaded at application startup.
- The runtime cache must be an in-memory dictionary keyed by `aadstsCode`.
- The service must use a hash map to support lookup completion in under 5 ms.
- Primary AADSTS code extraction must parse both:
  - `error_codes` array
  - `error_description` via regex `AADSTS[0-9]+`
- For mapped codes, the service must return the corresponding structured values from the cached mapping plus correlation information.
- For unmapped codes, the service must return fallback behavior including:
  - fallback user message
  - `contactIT` action type
  - raw AADSTS code
  - original Azure correlation ID
- `error_description` must be HTML-entity encoded before inclusion in `rawErrorDescription` to prevent XSS.
- Hot-reload must update cached mappings without application restart.
- Hot-reload detection mechanism may be file-watcher or polling; polling default is 60 seconds and interval is configurable.
- Reload events must be logged at INFO level.
- Malformed JSON entries must not invalidate valid entries; malformed entries are skipped with WARNING logging.

## Data Model & Validation
### Mapping Entry Schema
Each JSON mapping entry must include:
- `aadstsCode`: string
- `severity`: enum `info | warning | error | critical`
- `userMessage`: string, maximum 200 characters
- `recommendedAction`: string, maximum 100 characters
- `actionType`: enum `retry | resetPassword | completeMFA | contactIT`
- `localeKey`: string for future i18n

### Structured Error Response Object
Planned contract fields:
- `code`: string
- `userMessage`: string
- `recommendedAction`: string
- `severityLevel`: string
- `correlationId`: string
- `rawErrorDescription`: string

### Validation Rules
- Mapping JSON must validate against the approved schema at load time.
- Each malformed entry must be logged and skipped individually.
- Valid entries must remain available even when malformed entries exist in the same file.
- `userMessage` must not exceed 200 characters.
- `recommendedAction` must not exceed 100 characters.
- `severity` must be one of the approved enum values.
- `actionType` must be one of the approved enum values.
- `rawErrorDescription`, when included, must be HTML entity encoded.

### Reference Data
#### Severity enum
- `info`
- `warning`
- `error`
- `critical`

#### ActionType enum
- `retry`
- `resetPassword`
- `completeMFA`
- `contactIT`

### Open Questions
- Validate whether the JSON file root structure is an array of entries or an object containing entries.
- Validate whether `aadstsCode` must include the `AADSTS` prefix or may be a numeric code string only.
- Validate whether duplicate `aadstsCode` entries are allowed and, if so, which entry wins.
- Validate whether `localeKey` is required for every entry or may be empty.
- Validate whether `severityLevel` in the response must use the same enum literals as mapping `severity`.

## Functional Requirements
FR-1. The feature shall produce an approved JSON schema document for the AADSTS mapping file including the fields `aadstsCode`, `severity`, `userMessage`, `recommendedAction`, `actionType`, and `localeKey` with the source-defined types and constraints.

FR-2. The feature shall produce an AADSTS code inventory compiled from Azure AD documentation and the last 90 days of production authentication error logs.

FR-3. The AADSTS inventory shall contain at least 15 codes.

FR-4. Each inventoried AADSTS code shall be assigned a severity classification from `info`, `warning`, `error`, or `critical`.

FR-5. Each inventoried AADSTS code shall be assigned an action type from `retry`, `resetPassword`, `completeMFA`, or `contactIT`.

FR-6. The feature shall define and document the structured error response object contract containing `code`, `userMessage`, `recommendedAction`, `severityLevel`, `correlationId`, and `rawErrorDescription`.

FR-7. The structured error response contract shall be reviewed and approved by the frontend lead.

FR-8. The response object contract shall be reviewed by backend lead.

FR-9. The feature shall document fallback behavior for cases where an AADSTS code is not found in the mapping.

FR-10. The backend shall implement an `ErrorMappingService` class.

FR-11. `ErrorMappingService` shall load the externalized AADSTS mapping JSON file at application startup.

FR-12. `ErrorMappingService` shall store valid mapping entries in an in-memory dictionary keyed by `aadstsCode`.

FR-13. `ErrorMappingService` shall expose a `resolve(errorPayload)` method.

FR-14. `resolve(errorPayload)` shall extract the primary AADSTS code from Azure AD error payloads by parsing both the `error_codes` array and the `error_description` string using regex `AADSTS[0-9]+`.

FR-15. When the extracted AADSTS code exists in the cached mapping, `resolve(errorPayload)` shall return a structured error object populated with the mapped `userMessage`, `recommendedAction`, and `severityLevel`, and the Azure correlation identifier from the original payload.

FR-16. `resolve(errorPayload)` shall return correct mapped messages for all 15 or more inventoried AADSTS codes represented in the mapping file.

FR-17. When the extracted AADSTS code is not found in the cached mapping, `resolve(errorPayload)` shall return a fallback object with `userMessage` set to `Authentication failed. Please contact IT Support.`

FR-18. For unmapped AADSTS codes, the fallback response shall include the raw extracted AADSTS code and the original Azure `correlation_id`.

FR-19. For unmapped AADSTS codes, fallback behavior shall assign `actionType` value `contactIT`.

FR-20. The service shall HTML-entity encode `error_description` before including it in `rawErrorDescription`.

FR-21. The service shall validate the mapping JSON against the approved schema during load and reload operations.

FR-22. If a single mapping entry is malformed, the service shall log a WARNING containing entry details, skip that entry, and continue loading remaining valid entries.

FR-23. Valid mapping entries shall remain resolvable even when malformed entries are present in the same mapping file.

FR-24. The service shall support hot-reload of mapping file changes without application restart.

FR-25. The hot-reload mechanism shall use either file-watcher or polling.

FR-26. If polling is used, the interval shall be configurable and default to 60 seconds.

FR-27. Reload events shall be logged at INFO level.

FR-28. Cached lookup performance for resolving a code from the in-memory dictionary shall complete in under 5 ms.

FR-29. The backend implementation shall pass unit tests covering known codes, unknown codes, malformed entries, and hot-reload scenarios.

FR-30. The feature shall be implemented within the selected monolith architecture style.

## Non-Functional Requirements
- Performance:
  - Cached lookup must complete in under 5 ms.
  - Hash map/in-memory dictionary must be used to support lookup performance.
- Reliability:
  - Valid mappings must continue functioning when malformed entries are present.
  - Mapping changes must be applied without application restart.
- Security:
  - `error_description` must be sanitized using HTML entity encoding before inclusion in `rawErrorDescription` to prevent XSS.
- Maintainability:
  - Mapping definitions must be externalized to a JSON file.
  - Hot-reload must allow operational updates without redeployment/restart.
- Observability:
  - Reload events must be logged at INFO level.
  - Malformed mapping entries must be logged at WARNING level with entry details.
- Testability:
  - Unit tests must cover known-code resolution, unknown-code fallback, malformed entries, and hot-reload scenarios.
- Architectural context:
  - Implementation must fit within the selected monolith style.

## Acceptance Scenarios
### Scenario 1: Planning produces approved schema and inventory
**Given** the planning work for AADSTS mapping is underway  
**When** Azure AD documentation and the last 90 days of production authentication error logs are audited  
**Then** an AADSTS inventory with at least 15 codes shall be produced  
**And** each code shall have an assigned severity and action type  
**And** the JSON schema document shall include all required fields, types, and constraints

### Scenario 2: Structured response contract review
**Given** the structured error response contract has been defined  
**When** it is reviewed by the frontend lead and backend lead  
**Then** the contract review shall be completed  
**And** frontend lead approval shall be recorded as part of done criteria

### Scenario 3: Startup load of valid mapping file
**Given** the application starts with a valid externalized mapping JSON file  
**When** `ErrorMappingService` initializes  
**Then** it shall load the mapping file into an in-memory dictionary keyed by `aadstsCode`  
**And** valid entries shall be available for lookup via `resolve(errorPayload)`

### Scenario 4: Resolve known code from error_codes array
**Given** the cache contains a mapping for an inventoried AADSTS code  
**And** an Azure AD error payload contains that code in `error_codes`  
**When** `resolve(errorPayload)` is called  
**Then** the service shall return the mapped `userMessage`, `recommendedAction`, and `severityLevel`  
**And** it shall include the original Azure correlation identifier

### Scenario 5: Resolve known code from error_description
**Given** the cache contains a mapping for an inventoried AADSTS code  
**And** an Azure AD error payload contains the code in `error_description` matching regex `AADSTS[0-9]+`  
**When** `resolve(errorPayload)` is called  
**Then** the service shall extract the code from `error_description`  
**And** return the mapped structured error object

### Scenario 6: Unknown code fallback
**Given** an Azure AD error payload contains an AADSTS code not present in the cache  
**When** `resolve(errorPayload)` is called  
**Then** the service shall return `userMessage` equal to `Authentication failed. Please contact IT Support.`  
**And** the response shall include the raw AADSTS code  
**And** the response shall include the original Azure `correlation_id`  
**And** fallback behavior shall assign `actionType` `contactIT`

### Scenario 7: Sanitized raw error description
**Given** an Azure AD error payload contains HTML or script-like content in `error_description`  
**When** `resolve(errorPayload)` includes `rawErrorDescription` in the response  
**Then** the value shall be HTML entity encoded before inclusion

### Scenario 8: Malformed mapping entry during load
**Given** the mapping file contains both valid entries and one malformed entry  
**When** the service loads or reloads the file  
**Then** the malformed entry shall be skipped  
**And** a WARNING log shall be written with the malformed entry details  
**And** valid entries shall remain functional

### Scenario 9: Hot-reload after mapping file update
**Given** the service is running with cached mappings  
**And** the externalized mapping file is updated  
**When** the hot-reload mechanism detects the change  
**Then** the cache shall be refreshed without application restart  
**And** reload events shall be logged at INFO level  
**And** updated mappings shall become available within the configured polling interval when polling is used

### Scenario 10: Lookup performance
**Given** a valid cached mapping exists  
**When** `resolve(errorPayload)` performs a code lookup  
**Then** the lookup shall complete in under 5 ms using the in-memory hash map

## Traceability Matrix
| Source ID | Requirement | Acceptance Criteria | Test Coverage |
|---|---|---|---|
| Feature 1087793368 / US DEMO26-2417 | FR-1 | JSON schema document includes all required fields with types and constraints | Schema document review |
| Feature 1087793368 / US DEMO26-2417 | FR-2, FR-3 | At least 15 AADSTS codes inventoried with severity and actionType assigned | Inventory review against source inputs |
| Feature 1087793368 / US DEMO26-2417 | FR-4, FR-5 | At least 15 AADSTS codes inventoried with severity and actionType assigned | Classification review |
| Feature 1087793368 / US DEMO26-2417 | FR-6, FR-7 | Structured error response contract reviewed and approved by frontend lead | Contract review signoff |
| Feature 1087793368 / US DEMO26-2417 | FR-8, FR-9 | Definition of done includes response object contract reviewed by frontend and backend leads and fallback behavior documented | Review and documentation check |
| Feature 1087793368 / US DEMO26-2418 | FR-10, FR-11, FR-12, FR-13 | Service implemented with JSON file loader and cache | Unit/integration initialization tests |
| Feature 1087793368 / US DEMO26-2418 | FR-14, FR-15, FR-16 | `resolve()` returns correct mapped message for all 15+ inventoried AADSTS codes | Known-code unit tests from `error_codes` and `error_description` |
| Feature 1087793368 / US DEMO26-2418 | FR-17, FR-18, FR-19 | Fallback message returned for unmapped codes includes raw code and correlation_id | Unknown-code unit tests |
| Feature 1087793368 / US DEMO26-2418 | FR-20 | Sanitize `error_description` using HTML entity encoding before including in `rawErrorDescription` | Security-focused unit tests |
| Feature 1087793368 / US DEMO26-2418 | FR-21, FR-22, FR-23 | Malformed JSON entries are skipped with WARNING log while valid entries remain functional | Malformed-entry load tests |
| Feature 1087793368 / US DEMO26-2418 | FR-24, FR-25, FR-26, FR-27 | Hot-reload updates cached mappings within configured polling interval without restart | Hot-reload tests; log verification |
| Feature 1087793368 / US DEMO26-2418 | FR-28 | Ensure lookup completes in < 5 ms by using a hash map | Performance unit/benchmark test |
| Feature 1087793368 / US DEMO26-2418 | FR-29 | Definition of done: service passes unit tests for known codes, unknown codes, malformed entries, and hot-reload scenarios | Unit test suite |
| Feature 1087793368 | FR-30 | User-selected architecture style: monolith | Architecture conformance review |

## Open Questions
1. Which hot-reload mechanism is authoritative for implementation in this feature: file-watcher, polling, or both?
2. If both file-watcher and polling are supported, what is the selection/configuration rule?
3. What is the exact JSON file root structure: array of mapping entries or object wrapper?
4. Is `aadstsCode` stored with the `AADSTS` prefix (for example `AADSTS50076`) or as digits only?
5. If both `error_codes` array and `error_description` contain different AADSTS codes, which source takes precedence for the “primary” code?
6. If multiple AADSTS codes appear in `error_description`, which one is considered primary?
7. What response should be returned when no AADSTS code can be extracted from either `error_codes` or `error_description`?
8. Is `actionType` required in the runtime structured error response object for all cases, since planning defines it in mapping and backend fallback includes it, but one backend response example omits it?
9. Is `rawErrorDescription` required in all runtime responses or only fallback responses?
10. Should `correlationId` be populated only from Azure `correlation_id`, or is there any fallback source if absent?
11. What is the startup behavior if the overall JSON file is unreadable or fails schema validation broadly: fail startup, start with empty cache, or retain previous cache on reload?
12. Are duplicate `aadstsCode` entries permitted in the mapping file, and if so, which entry wins?
13. Is `localeKey` mandatory and non-empty for all entries, or optional until i18n is implemented?
14. What level of detail is required in malformed-entry logs to satisfy “entry details” while avoiding sensitive data exposure?
15. Does frontend require specific rendering behavior based on `severityLevel` or `actionType`?
16. Is a config endpoint for reload allowed in the monolith deployment model for this feature, or should implementation remain file-based only?
17. Who owns and approves ongoing production mapping file changes after initial delivery?

## Source References
- Feature ID: 1087793368
- Feature Reference: 1087793368
- Feature Title: [Planning] Define AADSTS code inventory, JSON schema, and severity classification; [Backend] Implement AADSTS error mapping service with JSON file loader and cache
- User Story: US DEMO26-2417 — [Planning] Define AADSTS code inventory, JSON schema, and severity classification
- User Story: US DEMO26-2418 — [Backend] Implement AADSTS error mapping service with JSON file loader and cache
- Source acceptance criteria/checklists embedded in:
  - US DEMO26-2417 description and acceptance checklist
  - US DEMO26-2418 description and acceptance checklist
- Golden Repo references used:
  - `_source-manifest.json`