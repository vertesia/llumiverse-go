# CLAUDE.md

This project's guidance for AI agents lives in [AGENTS.md](AGENTS.md). Read it
before making changes — it covers the architecture, build/test/lint commands,
coding conventions, and how to add a driver.

Quick reminders:

- Run `gofmt -w` on every file you touch; CI fails on any unformatted file.
- `go build ./...`, `go vet ./...`, and `go test ./...` must pass (tests run
  offline with mocks; live-credential tests are opt-in via `.env`).
- Provider-specific knobs go in `ExecutionOptions.ModelOptions`, not new struct
  fields. Keep public types provider-neutral and isolate quirks in the driver.
