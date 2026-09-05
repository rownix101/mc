# Repository Guidelines

## Project Structure & Module Organization

This Rust 2024 crate has its library in `src/lib.rs` and desktop entry point in `src/main.rs`. Application/UI code is in `src/app.rs`; GPU setup and frame rendering are in `src/render/gpu.rs`. World generation is under `src/world/`: blocks, texture atlasing, columns, continent/plains heightmaps, and trees. Diagnostic binaries are in `src/bin/` (`blocks`, `preview`, `continent`). Textures and attribution are in `assets/textures/`; project direction is in `docs/TECH_ROADMAP.md`. Build output belongs in generated `target/`.

## Build, Test, and Development Commands

Run these from the repository root:

- `cargo fmt --all -- --check` — verify formatting.
- `cargo check --all-targets --all-features` — type-check the app and helper binaries.
- `cargo test --all-targets --all-features` — run unit tests, including world-generation invariants.
- `cargo clippy --all-targets --all-features -- -D warnings` — run lint checks as errors.
- `cargo run` — launch the debug app (`debug-ui` is enabled by default).
- `cargo run --release --bin blocks` — generate atlas/column previews under `target/`.
- `cargo run --release --bin preview -- [seed] [size_px] [step] [out]` — render a plains preview.
- `cargo run --release --bin continent -- [seed]` — render continent and coast diagnostics.

## Coding Style & Naming Conventions

Use rustfmt defaults, four-space indentation, and idiomatic Rust naming: `snake_case` for functions/modules, `UpperCamelCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Keep world generation deterministic and pure where possible; preserve explicit block IDs and append new IDs. Keep comments and docs accurate when changing coordinate, terrain, or rendering assumptions.

## Testing Guidelines

Tests are colocated with implementation modules under `src/world/` and use Rust’s built-in `#[test]` framework. Name tests after the behavior they protect, such as `deterministic`, `layers_land`, or `land_and_sea_both_exist`. For terrain changes, cover determinism, land/sea boundaries, continuity, layer invariants, and representative integration counts. Run the test command before submitting changes.

## Commit & Pull Request Guidelines

No Git history is available in this checkout, so no existing convention can be inferred. Use short, imperative subjects (for example, `Add column ore distribution`) and keep unrelated changes separate. Pull requests should explain behavior and affected modules, mention validation commands, link an issue when applicable, and include preview images or a short capture for visual changes.

## Configuration & Asset Notes

The default `debug-ui` feature enables egui diagnostics; use `--no-default-features` when checking the non-UI build. Texture filenames referenced by `src/world/block.rs` must exist in `assets/textures/`, remain 16×16 where expected, and retain required attribution.
