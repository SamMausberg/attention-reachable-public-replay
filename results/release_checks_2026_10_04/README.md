# Release checks: October 4, 2026

The current source was checked with these commands:

```sh
make all test check output-check
cd formal
lake exe cache get
lake build
lake env lean CheckAxioms.lean
```

The C++ build, arithmetic tests, compact replay, and retained output checks
passed with Python 3.12.3, NumPy 2.3.5, SciPy 1.17.0, and mpmath 1.3.0.
The numerical records here match the retained records apart from elapsed time.
This rerun does not repeat model-checkpoint extraction or GPU experiments.

The Lake build passed with Lean 4.34.1 and the pinned mathlib revision
`d13f23b723b8a846827a245b89c10fc7d3f11612`. An identical source copy was built
on the local Linux filesystem. The six declarations have no `sorryAx`
dependencies; their precise axiom lists are in `lean_axioms.log`. This checks
the stated finite kernel, not the complete manuscript or numerical executor.

`bash paper/build.sh` also passed in a separate validation copy with pdfTeX
1.40.25 (TeX Live 2023/Debian), without final-pass warnings or unresolved
references. The released manuscript PDF is preserved.
