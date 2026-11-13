// Package wire holds the JSON shapes llumiverse publishes, generated from the llumiverse JSON Schema
// document (schema/llumiverse.schema.json), which the TypeScript llumiverse emits from its Zod schemas.
//
// Do not edit types.gen.go: run `go generate ./wire` after updating the schema with
// scripts/sync-schema.sh.
package wire

//go:generate go -C ../gen run ./cmd/wiregen -in ../schema/llumiverse.schema.json -config ../wire/oapi-codegen.yaml
