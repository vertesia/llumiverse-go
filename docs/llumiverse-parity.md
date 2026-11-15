# Llumiverse parity

This file tracks how llumiverse-go follows the TypeScript
[llumiverse](https://github.com/vertesia/llumiverse). It lists every llumiverse commit that touched
`drivers/src`, `core/src` or `common/src` since the Go port, with what llumiverse-go did about it.
The commit llumiverse-go now matches is in [`LLUMIVERSE_PARITY`](../LLUMIVERSE_PARITY).

## Baseline

llumiverse-go's initial commit (2026-06-01 14:36 +0900) is a port of llumiverse `main` as of
`51e6007` (feat: support per-call timeouts and retry signals, #422, 2026-05-30) or slightly later:

- Present in Go: the per-call `HTTPTimeoutOptions` (`ExecutionOptions.HTTPTimeout`,
  `DriverOptions.HTTPTimeout`) that #422 introduced.
- Absent from Go: `keepRecentMessages` turn-aware truncation (`1cea46c`, 2026-05-31), the debug
  prompt (#438, 2026-06-03), and the Gemini URL-fetcher throttling retry that llumiverse had since
  #415 (2026-05-24) and #424 (2026-05-28). The last shows the port was not exhaustive, so the triage
  starts from the earliest plausible point, `51e6007`, and also lists the two older commits that
  reached `main` through later merges (`f140853`, `b413edb`).

The range is `git log --no-merges 51e6007..73dfc90 -- drivers/src core/src common/src`: 140 commits.

## Decisions

- **port**: affects behaviour llumiverse-go has; reproduced in Go with an offline test.
- **ported**: done; the Go commit is named in the reason.
- **covered**: already true in Go (often through the generated wire types).
- **skip**: no Go counterpart (TS tooling, dependency bumps, providers llumiverse-go does not have,
  catalogue metadata llumiverse-go does not carry). The reason says which.

A commit that spans several areas appears in each area's table with that area's part.


## Core and common

`core/`, `common/`, root re-exports: lifecycle, streams, errors, conversation utilities, token usage, finish reasons.

| Commit | Date | Subject | Decision | Reason |
| --- | --- | --- | --- | --- |
| `1cea46c` | 2026-05-31 | feat: turn-aware large-text truncation in conversations | port |  |
| `d74f811` | 2026-06-03 | feat: Debug prompt, binary truncation (#438) | port | core hook ported in the foundation commit; drivers implement FormatDebugPrompt. |
| `21be90e` | 2026-06-10 | fix: invalid_grant is non-retryable (#459) | port |  |
| `e53b936` | 2026-06-11 | fix: retry transport aborts and deadline-exceeded errors (#458) | port |  |
| `83cf1fa` | 2026-06-13 | fix(core): don't split surrogate pairs when truncating conversation text | port |  |
| `d3aa462` | 2026-06-16 | fix(stream): reassemble Gemini thought_signature across streamed tool-call chunks (#470) | port |  |
| `82d587e` | 2026-06-16 | fix: bring main OpenAI changes to preview, fix openai parsing bugs (#468) | port |  |
| `335bec0` | 2026-07-03 | feat: add Vertex MaaS and shared Chat Completions support (#506) | port | Vertex MaaS + shared Chat Completions; the MaaS routing file in drivers/vertexai is owned by the OpenAI batch. |
| `2a38024` | 2026-07-14 | feat: improve implicit prompt caching (#531) | port |  |
| `94adce2` | 2026-07-17 | [Backport main] fix: improve Claude reasoning and truncation reliability (#542) | port |  |
| `c71fb9b` | 2026-08-06 | feat: preserve native thinking output across providers (#532) | port |  |
| `105b57f` | 2026-08-12 | fix: harden model catalog resolution (#579) | port | Catalogue/option metadata parts skip (no Go counterpart); review driver routing. |
| `a096147` | 2026-08-14 | fix: make driver cleanup in-flight safe (#587) | port |  |
| `a7c34be` | 2026-08-18 | fix: carry tool-result attachments to OpenAI Responses and Gemini (#593) | port |  |
| `ddb1416` | 2026-08-20 | feat(vertexai): add Gemini Omni video generation (#600) | port |  |
| `4627464` | 2026-08-25 | feat: capture provider service tiers (#607) | port |  |
| `ce176ba` | 2026-08-26 | fix: coordinate Gemini context caches (#615) | port |  |
| `ee01d4d` | 2026-08-28 | [Backport main] fix: reject malformed streamed tool calls (#626) | port |  |
| `98f5eee` | 2026-09-01 | feat: named tool choice across drivers with typed contract failures (#629) | port |  |
| `2e6580b` | 2026-09-07 | [Backport main] fix: honor Gemini HTTP transport timeouts (#643) | port |  |
| `311c5b9` | 2026-09-18 | [Backport main] feat: add embedding batch provider adapter with Vertex support (#671) | port |  |
| `c1df2ee` | 2026-09-22 | feat: add bounded file audio across primary providers (#672) | port |  |
| `cf780e4` | 2026-09-28 | feat: report one-hour cache writes and provider-reported cost in token usage (#697) | port |  |
| `08bc093` | 2026-10-02 | [Backport main] fix: normalize provider errors before logging (#732) | port |  |
| `6003a78` | 2026-10-09 | fix: drop empty streamed tool arguments on a length stop (#756) | port |  |
| `e449750` | 2026-10-13 | [Backport main] fix: cap total inline Gemini media per request, log result validation failures at info (#769) | port |  |
| `657ff94` | 2026-10-14 | feat(claude): append-only tool changes, Bedrock Claude on the Messages API (#779) | port |  |
| `3e94ace` | 2026-10-26 | fix: classify errors raised while building a prompt (#822) | port |  |
| `77a9744` | 2026-10-29 | fix: report the provider's stop when it cuts a response off mid tool call (#835) | port |  |
| `b3cdb45` | 2026-11-13 | feat(openai): update openai to 7.56.0 and map new usage, tool usage and residency (#893) | port |  |

## Claude (Anthropic, Claude on Vertex AI, Claude on Bedrock)

`drivers/internal/claude`, `drivers/anthropic`, `drivers/vertexai/claude.go`, and the Claude-specific Bedrock path.

| Commit | Date | Subject | Decision | Reason |
| --- | --- | --- | --- | --- |
| `1cea46c` | 2026-05-31 | feat: turn-aware large-text truncation in conversations | port |  |
| `d74f811` | 2026-06-03 | feat: Debug prompt, binary truncation (#438) | port | core hook ported in the foundation commit; drivers implement FormatDebugPrompt. |
| `0349028` | 2026-06-10 | feat: remove restrictedTypes for file handling for Claude tool results, and log video attachment. (#457) | port |  |
| `6c39d20` | 2026-06-14 | fix(drivers): drop orphaned tool_result blocks before sending (Claude, Bedrock, OpenAI, Gemini) (#463) | port |  |
| `2a38024` | 2026-07-14 | feat: improve implicit prompt caching (#531) | port |  |
| `94adce2` | 2026-07-17 | [Backport main] fix: improve Claude reasoning and truncation reliability (#542) | port |  |
| `fb86a38` | 2026-08-01 | fix(bedrock): drop Claude JSON prefill and honor the exclusive 128K output bound (#562) | port |  |
| `c71fb9b` | 2026-08-06 | feat: preserve native thinking output across providers (#532) | port |  |
| `01b3149` | 2026-08-11 | fix: derive Bedrock Claude limits from model version and strip region prefixes (#582) | port |  |
| `105b57f` | 2026-08-12 | fix: harden model catalog resolution (#579) | port | Catalogue/option metadata parts skip (no Go counterpart); review driver routing. |
| `307cb5e` | 2026-08-24 | fix: keep Claude agent prompt prefixes byte-stable across turns (#606) | port |  |
| `98f5eee` | 2026-09-01 | feat: named tool choice across drivers with typed contract failures (#629) | port |  |
| `067f0dd` | 2026-09-18 | [Backport main] fix: make model option family IDs optional (#676) | port | Optional family IDs: covered by wire types; review Claude option reads. |
| `c1df2ee` | 2026-09-22 | feat: add bounded file audio across primary providers (#672) | port |  |
| `1d8ded7` | 2026-09-23 | [Backport main] fix: remove cross-model temperature defaults (#687) | port | Option metadata defaults; check no driver sends a default temperature. |
| `bef5ef3` | 2026-09-24 | [Backport main] fix: correct Bedrock Claude output token limits (#694) | port |  |
| `8e6e227` | 2026-09-24 | [Backport main] feat: update current OpenAI and Claude model support (#695) | port |  |
| `cf780e4` | 2026-09-28 | feat: report one-hour cache writes and provider-reported cost in token usage (#697) | port |  |
| `4448f13` | 2026-09-29 | [Backport main] feat: support Sonnet 5.5 between-tools thinking (#713) | port |  |
| `1d2a91e` | 2026-10-08 | [Backport main] fix: honor Haiku 5.5 default thinking and effort (#753) | port |  |
| `f61c36b` | 2026-10-12 | [Backport main] fix: align provider options, limits, and input handling (#764) | port |  |
| `28b35a1` | 2026-10-13 | feat: log provider refusals and content-filter stops (#772) | port |  |
| `1eed259` | 2026-10-14 | [Backport main] fix(claude): drop thinking blocks whose conversation prefix changed (#778) | port |  |
| `657ff94` | 2026-10-14 | feat(claude): append-only tool changes, Bedrock Claude on the Messages API (#779) | port |  |
| `70c413c` | 2026-10-22 | fix(drivers): refuse a non-array conversation in OpenAI and Gemini as a 400 (#810) | port |  |
| `84bf7ed` | 2026-10-29 | chore(deps): update model provider client libraries to v1 (main) (major) (#837) | port | SDK major bump; port the Claude API version change. |
| `77a9744` | 2026-10-29 | fix: report the provider's stop when it cuts a response off mid tool call (#835) | port |  |
| `b9c8505` | 2026-11-10 | feat(bedrock): use native structured outputs for Claude 4.5+ (#867) | port |  |
| `d53f9c0` | 2026-11-10 | feat(claude): support Claude Fable 5.2 fast mode and Mythos thinking binding (#873) | port |  |
| `bf4ddae` | 2026-11-11 | fix(claude): keep agent prompt caches growing across tool loops and turns (#880) | port |  |
| `9b49f2e` | 2026-11-13 | feat: server-side conversation compaction for Claude and OpenAI Responses (#890) | port |  |
| `e1cb233` | 2026-11-13 | fix(claude): end Opus 4.8 fast mode and offer fast mode on Vertex AI Fable/Mythos 5.2+ (#892) | port |  |
| `56274b2` | 2026-11-14 | fix(vertexai): route a bare Claude model id to the Claude definition (#898) | port |  |
| `73dfc90` | 2026-11-14 | fix(drivers): send Claude result schemas natively on Vertex AI (#899) | port |  |

## OpenAI and OpenAI-compatible

`drivers/openai` (Responses and Chat Completions), plus the Vertex AI MaaS models that ride on Chat Completions.

| Commit | Date | Subject | Decision | Reason |
| --- | --- | --- | --- | --- |
| `d74f811` | 2026-06-03 | feat: Debug prompt, binary truncation (#438) | port | core hook ported in the foundation commit; drivers implement FormatDebugPrompt. |
| `6c39d20` | 2026-06-14 | fix(drivers): drop orphaned tool_result blocks before sending (Claude, Bedrock, OpenAI, Gemini) (#463) | port |  |
| `0ea356e` | 2026-06-16 | Refactor TogetherAI driver onto OpenAI base (#471) | port | TogetherAI parts skip (no TogetherAI driver); review the OpenAI base changes. |
| `82d587e` | 2026-06-16 | fix: bring main OpenAI changes to preview, fix openai parsing bugs (#468) | port |  |
| `19f6e02` | 2026-06-24 | feat: support Grok models on Google Gemini Enterprise Agent Platform (Vertex AI) (#388) | port | Grok on Vertex rides on the Vertex MaaS Chat Completions path (owned with #506). |
| `f7e2931` | 2026-07-02 | fix(togetherai): use Chat Completions for vision (Together lacks /v1/responses) (#507) | port | TogetherAI/Groq parts skip; review the shared OpenAI changes. |
| `335bec0` | 2026-07-03 | feat: add Vertex MaaS and shared Chat Completions support (#506) | port | Vertex MaaS + shared Chat Completions; the MaaS routing file in drivers/vertexai is owned by the OpenAI batch. |
| `4c470e3` | 2026-07-10 | feat: bedrock mantle models (#522) | port | Bedrock Mantle (OpenAI-compatible on Bedrock): port only if it rides on the OpenAI-compatible driver. |
| `0b5adb7` | 2026-07-13 | feat: expand Bedrock model support (#526) | port |  |
| `eb77310` | 2026-07-13 | feat: unify OpenAI-compatible drivers (#528) | port | Groq/Mistral/Together/xAI parts skip; review the OpenAI-compatible unification. |
| `2a38024` | 2026-07-14 | feat: improve implicit prompt caching (#531) | port |  |
| `2df2ace` | 2026-07-14 | feat: add explicit OpenAI cache boundaries (#535) | port |  |
| `c8f5863` | 2026-07-16 | feat: preserve prompt caches across result schemas (#541) | port |  |
| `94adce2` | 2026-07-17 | [Backport main] fix: improve Claude reasoning and truncation reliability (#542) | port |  |
| `c71fb9b` | 2026-08-06 | feat: preserve native thinking output across providers (#532) | port |  |
| `0290d95` | 2026-08-06 | fix: pass effort to OpenAI-compatible endpoints (#576) | port |  |
| `105b57f` | 2026-08-12 | fix: harden model catalog resolution (#579) | port | Catalogue/option metadata parts skip (no Go counterpart); review driver routing. |
| `d7c3752` | 2026-08-12 | feat: support provider service tiers (#583) | port |  |
| `a7c34be` | 2026-08-18 | fix: carry tool-result attachments to OpenAI Responses and Gemini (#593) | port |  |
| `b47bee1` | 2026-08-19 | fix(openai): handle schemas outside strict subset (#599) | port |  |
| `4627464` | 2026-08-25 | feat: capture provider service tiers (#607) | port |  |
| `98f5eee` | 2026-09-01 | feat: named tool choice across drivers with typed contract failures (#629) | port |  |
| `a2741de` | 2026-09-14 | [Backport main] fix: preserve Chat Completions tool images (#658) | port |  |
| `c1df2ee` | 2026-09-22 | feat: add bounded file audio across primary providers (#672) | port |  |
| `1d8ded7` | 2026-09-23 | [Backport main] fix: remove cross-model temperature defaults (#687) | port | Option metadata defaults; check no driver sends a default temperature. |
| `8e6e227` | 2026-09-24 | [Backport main] feat: update current OpenAI and Claude model support (#695) | port |  |
| `cf780e4` | 2026-09-28 | feat: report one-hour cache writes and provider-reported cost in token usage (#697) | port |  |
| `adf8f86` | 2026-10-01 | [Backport main] feat: support OpenAI image generation and editing (#725) | port |  |
| `45e0cda` | 2026-10-09 | [Backport main] fix: complete Bedrock GPT reasoning options and metadata (#759) | port |  |
| `f61c36b` | 2026-10-12 | [Backport main] fix: align provider options, limits, and input handling (#764) | port |  |
| `28b35a1` | 2026-10-13 | feat: log provider refusals and content-filter stops (#772) | port |  |
| `71d2a55` | 2026-10-15 | [Backport main] fix: model maintenance for Gemini 4, Claude 5.5, DeepSeek V4.1 Flash, GLM 5.3 and Bedrock listings (#787) | port |  |
| `2c7fc5f` | 2026-10-15 | [Backport main] fix: MiniMax structured output and Gemini explicit-cache bugs (#790) | port |  |
| `70c413c` | 2026-10-22 | fix(drivers): refuse a non-array conversation in OpenAI and Gemini as a 400 (#810) | port |  |
| `c51e8ba` | 2026-10-29 | [Backport main] fix(openai): harden GPT-6 and GPT-6.1 support (#840) | port |  |
| `77a9744` | 2026-10-29 | fix: report the provider's stop when it cuts a response off mid tool call (#835) | port |  |
| `97f7380` | 2026-11-06 | feat: support GPT Image 2.6 options and GIF/SVG output (#860) | port |  |
| `080f6a1` | 2026-11-10 | feat(openai): add a per-model prompt-cache lifetime option (#866) | port |  |
| `939bbd3` | 2026-11-12 | fix(openai): keep function call arguments sent without deltas (#884) | port |  |
| `9b49f2e` | 2026-11-13 | feat: server-side conversation compaction for Claude and OpenAI Responses (#890) | port |  |
| `b3cdb45` | 2026-11-13 | feat(openai): update openai to 7.56.0 and map new usage, tool usage and residency (#893) | port |  |

## Bedrock

`drivers/bedrock` for non-Claude models: Converse, InvokeModel, embeddings, images, listing, errors.

| Commit | Date | Subject | Decision | Reason |
| --- | --- | --- | --- | --- |
| `d74f811` | 2026-06-03 | feat: Debug prompt, binary truncation (#438) | port | core hook ported in the foundation commit; drivers implement FormatDebugPrompt. |
| `6c39d20` | 2026-06-14 | fix(drivers): drop orphaned tool_result blocks before sending (Claude, Bedrock, OpenAI, Gemini) (#463) | port |  |
| `c4183e6` | 2026-07-07 | [Backport main] fix: Bedrock execute timeout for long responses (#515) | port |  |
| `0b5adb7` | 2026-07-13 | feat: expand Bedrock model support (#526) | port |  |
| `2a38024` | 2026-07-14 | feat: improve implicit prompt caching (#531) | port |  |
| `94adce2` | 2026-07-17 | [Backport main] fix: improve Claude reasoning and truncation reliability (#542) | port |  |
| `c71fb9b` | 2026-08-06 | feat: preserve native thinking output across providers (#532) | port |  |
| `105b57f` | 2026-08-12 | fix: harden model catalog resolution (#579) | port | Catalogue/option metadata parts skip (no Go counterpart); review driver routing. |
| `d7c3752` | 2026-08-12 | feat: support provider service tiers (#583) | port |  |
| `307cb5e` | 2026-08-24 | fix: keep Claude agent prompt prefixes byte-stable across turns (#606) | port |  |
| `4627464` | 2026-08-25 | feat: capture provider service tiers (#607) | port |  |
| `ee01d4d` | 2026-08-28 | [Backport main] fix: reject malformed streamed tool calls (#626) | port |  |
| `98f5eee` | 2026-09-01 | feat: named tool choice across drivers with typed contract failures (#629) | port |  |
| `fead893` | 2026-09-14 | [Backport main] fix: preserve Bedrock tool images for vision models (#656) | port |  |
| `b5b6581` | 2026-09-18 | [Backport main] fix: align model option schemas, metadata and routing (#674) | port | Option schema parts covered by wire types. |
| `9ab9acb` | 2026-09-21 | [Backport main] refactor: use lru-cache for Bedrock cache (#680) | port | lru-cache swap is a TS implementation detail; review whether Go caches the same data. |
| `c1df2ee` | 2026-09-22 | feat: add bounded file audio across primary providers (#672) | port |  |
| `1d8ded7` | 2026-09-23 | [Backport main] fix: remove cross-model temperature defaults (#687) | port | Option metadata defaults; check no driver sends a default temperature. |
| `cf780e4` | 2026-09-28 | feat: report one-hour cache writes and provider-reported cost in token usage (#697) | port |  |
| `f157a07` | 2026-10-05 | fix: preserve generated-image metadata and reject empty output (#735) | port |  |
| `c01c096` | 2026-10-08 | fix(bedrock): never replay an empty assistant turn (#747) | port |  |
| `45e0cda` | 2026-10-09 | [Backport main] fix: complete Bedrock GPT reasoning options and metadata (#759) | port |  |
| `fa54a95` | 2026-10-10 | fix(bedrock): resolve regionless streaming discovery references (#761) | port |  |
| `f61c36b` | 2026-10-12 | [Backport main] fix: align provider options, limits, and input handling (#764) | port |  |
| `f7a78a3` | 2026-10-13 | [Backport main] fix(bedrock): surface the provider error when foundation models cannot be listed (#768) | port |  |
| `7187f56` | 2026-10-13 | [Backport main] fix: keep unknown schema formats and the Bedrock canStream check off stderr (#771) | port |  |
| `28b35a1` | 2026-10-13 | feat: log provider refusals and content-filter stops (#772) | port |  |
| `71d2a55` | 2026-10-15 | [Backport main] fix: model maintenance for Gemini 4, Claude 5.5, DeepSeek V4.1 Flash, GLM 5.3 and Bedrock listings (#787) | port |  |
| `c51e8ba` | 2026-10-29 | [Backport main] fix(openai): harden GPT-6 and GPT-6.1 support (#840) | port |  |
| `d7399b8` | 2026-11-13 | fix(bedrock): recognise Daybreak GPT model IDs as their GPT generation (#891) | port |  |

## Vertex AI (Gemini, Imagen, embeddings)

`drivers/vertexai` except the Claude path.

| Commit | Date | Subject | Decision | Reason |
| --- | --- | --- | --- | --- |
| `b413edb` | 2026-05-28 | fix: vertexai URL retry - cherry pick 2a6af93384f8bb3f5f572c951b47f51a355cade2 (#424) | port |  |
| `e1beebb` | 2026-06-03 | feat: Support GS URIs for Gemini (#436) | port |  |
| `d74f811` | 2026-06-03 | feat: Debug prompt, binary truncation (#438) | port | core hook ported in the foundation commit; drivers implement FormatDebugPrompt. |
| `21be90e` | 2026-06-10 | fix: invalid_grant is non-retryable (#459) | port |  |
| `0349028` | 2026-06-10 | feat: remove restrictedTypes for file handling for Claude tool results, and log video attachment. (#457) | port |  |
| `e53b936` | 2026-06-11 | fix: retry transport aborts and deadline-exceeded errors (#458) | port |  |
| `6c39d20` | 2026-06-14 | fix(drivers): drop orphaned tool_result blocks before sending (Claude, Bedrock, OpenAI, Gemini) (#463) | port |  |
| `ebbe339` | 2026-06-14 | fix(gemini): keep split parallel tool responses (#464) | port |  |
| `b1c7c28` | 2026-06-15 | fix(vertexai): retry Gemini 499 cancellations (#467) | port |  |
| `420fd65` | 2026-06-23 | fix: filter embedding models from global model in vertexai listing (#485) | port |  |
| `8207b0f` | 2026-07-09 | fix(vertexai): retry Gemini URL-fetcher throttle 400s (#520) | port |  |
| `2a38024` | 2026-07-14 | feat: improve implicit prompt caching (#531) | port |  |
| `94adce2` | 2026-07-17 | [Backport main] fix: improve Claude reasoning and truncation reliability (#542) | port |  |
| `7b6f5b0` | 2026-07-20 | [Backport main] fix: extend default Gemini request timeout (#545) | port |  |
| `c71fb9b` | 2026-08-06 | feat: preserve native thinking output across providers (#532) | port |  |
| `105b57f` | 2026-08-12 | fix: harden model catalog resolution (#579) | port | Catalogue/option metadata parts skip (no Go counterpart); review driver routing. |
| `d7c3752` | 2026-08-12 | feat: support provider service tiers (#583) | port |  |
| `a7c34be` | 2026-08-18 | fix: carry tool-result attachments to OpenAI Responses and Gemini (#593) | port |  |
| `b47bee1` | 2026-08-19 | fix(openai): handle schemas outside strict subset (#599) | port |  |
| `ddb1416` | 2026-08-20 | feat(vertexai): add Gemini Omni video generation (#600) | port |  |
| `2b3748e` | 2026-08-21 | feat(vertexai): cache the Gemini prompt prefix with Vertex cachedContents (#604) | port |  |
| `ce4eb06` | 2026-08-21 | fix: classify Gemini finish reason errors (#605) | port |  |
| `307cb5e` | 2026-08-24 | fix: keep Claude agent prompt prefixes byte-stable across turns (#606) | port |  |
| `4627464` | 2026-08-25 | feat: capture provider service tiers (#607) | port |  |
| `ce176ba` | 2026-08-26 | fix: coordinate Gemini context caches (#615) | port |  |
| `03c62c7` | 2026-08-27 | fix(vertexai): recombine split parallel tool results into one Gemini turn (#620) | port |  |
| `98f5eee` | 2026-09-01 | feat: named tool choice across drivers with typed contract failures (#629) | port |  |
| `0f1bf9d` | 2026-09-04 | [Backport main] feat(vertexai): support Gemini Omni Flash 1.1 (#640) | port |  |
| `2e6580b` | 2026-09-07 | [Backport main] fix: honor Gemini HTTP transport timeouts (#643) | port |  |
| `de5d7f1` | 2026-09-16 | fix(vertexai): keep parallel Gemini calls to one tool distinct when streaming (#663) | port |  |
| `311c5b9` | 2026-09-18 | [Backport main] feat: add embedding batch provider adapter with Vertex support (#671) | port |  |
| `b5b6581` | 2026-09-18 | [Backport main] fix: align model option schemas, metadata and routing (#674) | port | Option schema parts covered by wire types. |
| `c1df2ee` | 2026-09-22 | feat: add bounded file audio across primary providers (#672) | port |  |
| `1d8ded7` | 2026-09-23 | [Backport main] fix: remove cross-model temperature defaults (#687) | port | Option metadata defaults; check no driver sends a default temperature. |
| `e81a976` | 2026-09-25 | fix(vertexai): normalize Gemini audio finish reasons (#691) | port |  |
| `cf780e4` | 2026-09-28 | feat: report one-hour cache writes and provider-reported cost in token usage (#697) | port |  |
| `a3ac772` | 2026-09-30 | fix: clarify Vertex AI storage permission denials (#709) | port |  |
| `f157a07` | 2026-10-05 | fix: preserve generated-image metadata and reject empty output (#735) | port |  |
| `98f733a` | 2026-10-07 | [Backport main] fix(vertexai): support Nano Banana IDs and exclude Live models (#745) | port |  |
| `124d400` | 2026-10-08 | fix: cap inline Gemini media at 100 MB and quiet AJV unknown formats (#750) | port | AJV unknown-format part skip (no Go validation). |
| `f61c36b` | 2026-10-12 | [Backport main] fix: align provider options, limits, and input handling (#764) | port |  |
| `e449750` | 2026-10-13 | [Backport main] fix: cap total inline Gemini media per request, log result validation failures at info (#769) | port |  |
| `28b35a1` | 2026-10-13 | feat: log provider refusals and content-filter stops (#772) | port |  |
| `71d2a55` | 2026-10-15 | [Backport main] fix: model maintenance for Gemini 4, Claude 5.5, DeepSeek V4.1 Flash, GLM 5.3 and Bedrock listings (#787) | port |  |
| `2c7fc5f` | 2026-10-15 | [Backport main] fix: MiniMax structured output and Gemini explicit-cache bugs (#790) | port |  |
| `5c06658` | 2026-10-16 | feat: Gemini 4 Argon long decode on Vertex (#788) | port |  |
| `74be150` | 2026-10-19 | [Backport main] fix(vertexai): give every Gemini tool call a unique id (#802) | port |  |
| `70c413c` | 2026-10-22 | fix(drivers): refuse a non-array conversation in OpenAI and Gemini as a 400 (#810) | port |  |
| `3e94ace` | 2026-10-26 | fix: classify errors raised while building a prompt (#822) | port |  |
| `eefdaed` | 2026-10-28 | fix(vertexai): retry URL fetcher overload and timeout rejections (#832) | port |  |
| `151a031` | 2026-10-29 | fix(vertexai): report Gemini inline media over its size limit as a 413 (#838) | port |  |
| `27b5529` | 2026-11-13 | feat: @google/genai 3.3, Gemini 4 Hydrogen XHIGH and Gemini Omni 1.2 Flash (#894) | port |  |
| `b3cdb45` | 2026-11-13 | feat(openai): update openai to 7.56.0 and map new usage, tool usage and residency (#893) | port |  |

## Azure (Microsoft Foundry, Azure OpenAI)

New `drivers/azurefoundry` driver.

| Commit | Date | Subject | Decision | Reason |
| --- | --- | --- | --- | --- |
| `adf8f86` | 2026-10-01 | [Backport main] feat: support OpenAI image generation and editing (#725) | port |  |
| `59221b6` | 2026-10-02 | [Backport main] feat: support Anthropic Messages on Microsoft Foundry (#733) | port |  |
| `f61c36b` | 2026-10-12 | [Backport main] fix: align provider options, limits, and input handling (#764) | port |  |
| `1eed259` | 2026-10-14 | [Backport main] fix(claude): drop thinking blocks whose conversation prefix changed (#778) | port |  |
| `9da2fdc` | 2026-10-28 | feat(azure): request the Entra scope of the endpoint's cloud (#831) | port |  |
| `84bf7ed` | 2026-10-29 | chore(deps): update model provider client libraries to v1 (main) (major) (#837) | port | SDK major bump; port the Claude API version change. |
| `97f7380` | 2026-11-06 | feat: support GPT Image 2.6 options and GIF/SVG output (#860) | port |  |
| `80e7b80` | 2026-11-13 | feat(azure): bring the Microsoft Foundry and Azure OpenAI drivers up to date (#896) | port |  |

## Skipped and covered outside the areas

| Commit | Date | Subject | Decision | Reason |
| --- | --- | --- | --- | --- |
| `f140853` | 2026-05-26 | chore: update provider names (#406) | skip | Provider display names in the TS provider list; Go has no provider catalogue. |
| `062b21f` | 2026-06-01 | chore: switch Biome checks to the check command and apply fixes (#425) | skip | Biome formatting only. |
| `bf5b6ec` | 2026-06-01 | sync: merge preview to main (b413edb) (#426) | skip | Provider display names and the preview-to-main sync workflow. |
| `cc956cc` | 2026-06-04 | chore: apply main merge result | skip | Merge bookkeeping (preview to main); its content arrives through the individual commits triaged here. |
| `e76805a` | 2026-06-05 | chore(deps): update model provider client libraries (#446) | skip | Dependency update. |
| `f33f316` | 2026-06-09 | chore(deps): update dependency @biomejs/biome to v2.4.11 (#455) | skip | Biome update. |
| `fb000fc` | 2026-06-15 | feat(drivers): per-provider subpath exports for lazy loading (#466) | skip | TS package subpath exports (bundling); Go packages are already separate imports. |
| `9883db2` | 2026-06-16 | remove type assertion | skip | Lint-only. |
| `c416d25` | 2026-06-16 | linting fixes | skip | Lint-only. |
| `977cc81` | 2026-07-06 | chore: clean up biome suppressions (#511) | skip | Biome suppression cleanup. |
| `79a981d` | 2026-07-14 | fix: default streaming tool support to tool_support (#534) | skip | Capability catalogue metadata; Go has no capability catalogue. |
| `7e691c6` | 2026-07-24 | chore(deps): update dependency @biomejs/biome to v2.5.5 (#552) | skip | Biome update. |
| `e8b5b95` | 2026-08-04 | feat(common): define the API contracts as runtime Zod schemas (#565) | covered | API contracts as Zod schemas: covered by the generated wire types (#11); driver edits are TS type refactors. |
| `d797068` | 2026-08-14 | feat: support xAI image generation (#591) | skip | xAI image generation; no xAI driver in Go. |
| `87c2454` | 2026-08-15 | fix(core): stop rejecting valid completions after a schema's first use (#592) | skip | AJV validator caching bug; Go does not validate result schemas. |
| `c6d8566` | 2026-08-28 | feat: add native OpenRouter support (#624) | skip | Native OpenRouter driver; not a provider llumiverse-go has. |
| `ea82cc9` | 2026-09-08 | chore(deps): update node dependencies (main) (#649) | skip | Dependency update. |
| `51a9f91` | 2026-09-17 | chore(deps): update dependency openai to v7 (main) (#652) | skip | openai SDK v7 bump (TS SDK typing changes). |
| `2e4de8d` | 2026-09-22 | [Backport main] chore(deps): align Zod catalog and schema tests (#685) | skip | Zod catalogue test alignment. |
| `f618ce7` | 2026-09-28 | feat(drivers): add a driver factory with SDK-free declarations (#698) | skip | TS driver factory for lazy SDK loading; Go constructors are explicit and static. |
| `f9c602e` | 2026-10-08 | chore(deps): update model provider client libraries (main) (#746) | skip | Dependency update. |
| `3799add` | 2026-10-14 | [Backport main] test: update live provider smoke tests and remove redundant paths (#781) | skip | Live smoke tests and catalogue tweaks. |
| `d74c98a` | 2026-10-15 | fix: Watsonx embedding default, Groq structured outputs and smoke test follow-ups (#784) | skip | Watsonx and Groq only. |
| `2edad30` | 2026-10-15 | feat: describe Together Qwen reasoning controls, report Together streamed usage, and fix Groq Qwen version parsing (#792) | skip | Together and Groq only. |
| `1db4375` | 2026-10-20 | chore(deps): update model provider client libraries (main) (#785) | skip | Dependency update (Groq SDK). |
| `85934e5` | 2026-10-30 | chore: enable Biome 2.6 rules for YAML, Markdown and TypeScript (#841) | skip | Biome rules. |
| `5719e42` | 2026-11-03 | chore(deps): update dependency together-ai to v1 (main) (#856) | skip | TogetherAI SDK. |
| `63a13aa` | 2026-11-06 | fix(common): emit ModelOptionsSchema through a named type (#861) | covered | Named ModelOptions schema: covered by wire types. |
| `21c06da` | 2026-11-06 | fix(common): export the OpenAiImageSvgOptions type (#863) | skip | TS type export. |
| `b485719` | 2026-11-06 | fix(core): cache compiled result-schema validators by content (#864) | skip | AJV validator cache; Go does not validate result schemas. |
| `d01fb11` | 2026-11-10 | chore(deps): update dependency @biomejs/biome to v2.6.4 (main) (#878) | skip | Biome update. |
| `f7d2155` | 2026-11-13 | feat(common): publish a JSON Schema document of the common schemas (#895) | covered | JSON Schema document: the source of the generated wire types (#11). |
| `98ea7a4` | 2026-11-13 | docs: align JSON Schema openness notes with the API contract (#897) | covered | Schema documentation only. |
