---
name: Incremental builds invalidate bisection
description: CUDA incremental builds may not recompile all dependents when headers change — bisection across header-modifying commits requires clean builds
type: feedback
originSessionId: e6381887-6047-47bb-a3a3-a7bfce8e6af4
---
Never trust bisection results from incremental builds when the commit range includes header changes (especially CUDA `.cuh` files). Ninja/cmake dependency tracking for nvcc is not reliable across checkout switches — stale `.cu.o` files can persist, making "good" commits appear to pass when they're actually running old code.

**Why:** During the MTP acceptance regression bisection (2026-05-06), all intermediate commits tested GOOD with incremental builds, but the identified "first bad commit" was pure instrumentation that couldn't logically cause the regression. The entire bisection was invalid.

**How to apply:** When bisecting across commits that touch headers (`.h`, `.cuh`, `.hpp`), use `rm -rf build && cmake -B build ... && cmake --build build` for EVERY test point. Budget the extra build time rather than getting a fast but wrong answer.
