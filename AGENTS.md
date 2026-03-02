# Repository Guidelines

## Project Structure & Module Organization
NeuralReflex is a C++23 application built with CMake and Ninja.

- `src/`: implementation files grouped by domain (`core/`, `gfx/`, `utils/`).
- `include/nrx/`: public headers, organized to mirror source domains.
- `CMakeLists.txt`: single build entry point (targets, dependencies, compile settings).
- `CMakePresets.json`: standard local configure presets (`debug`, `relwithdebinfo`).
- `build/`: out-of-source build artifacts (do not commit generated files).

Keep new modules consistent with the existing layout, for example:
`include/nrx/gfx/new_feature.hpp` and `src/gfx/new_feature.cpp`.

## Build, Test, and Development Commands
Use CMake presets for reproducible local builds.

- `cmake --preset debug`: configure Debug build in `build/debug`.
- `cmake --build --preset debug`: build Debug preset.
- `cmake --preset relwithdebinfo`: configure optimized build with symbols.
- `cmake --build build/debug -j`: build with parallel jobs (manual form).

`compile_commands.json` is exported and copied to `build/compile_commands.json` for tooling (clangd, static analysis).

## Coding Style & Naming Conventions
- Language: C++23 (`CMAKE_CXX_STANDARD 23`).
- Formatting: use `.clang-format` before committing.
- Linting/static checks: follow `.clang-tidy` guidance when running analysis locally.
- Indentation: 4 spaces, no tabs.
- Naming: follow the naming rules configured in `.clang-tidy` for symbols (types, functions, variables).
- File naming: use `snake_case` for source and header files (for example `dx_context.cpp`, `dx_context.hpp`).

## Testing Guidelines
There is currently no dedicated automated test target in this repository.
Until tests are added:

- Build both `debug` and `relwithdebinfo` presets before opening a PR.
- Validate startup/shutdown paths and DirectX initialization changes manually.
- When adding non-trivial logic, include a brief manual test procedure in the PR description.

## Commit & Pull Request Guidelines
Commit messages MUST follow Conventional Commits exactly.

- Required header format: `<type>[optional scope][optional !]: <description>`
- `type` MUST be lowercase and SHOULD be one of:
  `feat`, `fix`, `build`, `chore`, `ci`, `docs`, `style`, `refactor`, `perf`, `test`, `revert`.
- `feat` MUST be used for new features.
- `fix` MUST be used for bug fixes.
- `scope` (if used) MUST be a noun in parentheses, e.g. `fix(renderer): ...`.
- Description MUST start immediately after `: ` and be a short summary.
- Body is OPTIONAL, but if present MUST start after one blank line.
- Footer(s) are OPTIONAL, but if present MUST start after one blank line.
- Footer token format MUST be `Token: value` or `Token #value`.
- Breaking changes MUST be indicated by either:
  `!` before `:`, or a footer `BREAKING CHANGE: <description>`.
- `BREAKING CHANGE` token MUST be uppercase (`BREAKING-CHANGE` is also allowed).

Examples:
- `feat(gfx): add swapchain resize handling`
- `fix(core): avoid null dereference on shutdown`
- `refactor(dx)!: replace device init sequence`
- `chore!: drop legacy adapter path` with footer `BREAKING CHANGE: removed old adapter fallback`

PRs should include:
- clear summary of behavior changes,
- linked issue/task (if available),
- build/test evidence (commands run and results),
- screenshots or logs when UI/runtime behavior changes.
