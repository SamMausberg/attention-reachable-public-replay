# A reachable-state separation between attention compression and its certificates

This repository contains the [paper](paper/paper.pdf), [LaTeX source](paper/paper.tex),
and replay bundle by Samuel Mausberg, including the retained witnesses and
reproduction instructions below. The checkpoint is fetched separately and checked
by SHA-256 before loading.

The experiment changes only the final attention read of frozen
SmolLM2-135M-Instruct Q8_0, dequantized to binary32. The earlier twenty-nine blocks
and every KV write remain dense. The refusal concerns a specified scalar-envelope
class: it passes incoming error through the final feed-forward residual with
unit coefficient and adds a nonnegative branch allowance. This convention is
not a derivative lower bound for the composite local map.

## Check the retained witnesses

Python, NumPy, a C++17 compiler and Boost headers suffice for the model-free checks.

```sh
make all
make test
make check
make output-check
```

The compact check verifies the local readout lower gain, the output-projection
Rayleigh witness, all nine clique witnesses, their attention-mass bounds, the
joint allocations at 6,451, 18,632 and 18,633 groups, and the rational sequence
budget. It does not re-execute the transformer. The readout upper gain of 0.746
also requires the full checkpoint; the compact witness establishes the lower
gain of 0.28 needed for the refusal.

The independent output check reconstructs eight first-intervention categorical
laws, compares their finite samplers and ideal-softmax enclosures, and checks
all 22 retained rollout snapshots. It compares 21 second-moment enclosures;
one Alice snapshot has a conservative precision-guard rejection, for which it
checks the served reference count law instead. The uniform sequence-TV bound
is below 0.000924 under the paper's stated numerical and sampling assumptions.
A sampled-path sum is not substituted for this guarantee.

## Fetch or verify the checkpoint

```sh
python code/fetch_checkpoint.py /absolute/path/SmolLM2-135M-Instruct-Q8_0.gguf
export MODEL_PATH=/absolute/path/SmolLM2-135M-Instruct-Q8_0.gguf
```

The downloader and model loader require this digest:

```
5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83
```

The downloader leaves an existing mismatched file unchanged and rejects a
mismatched download. `--verify-only` checks an existing file without using the
network. The checkpoint's published location is recorded in the script and
paper. No full model checkpoint or font file is bundled.

## Re-execute the sampled runs

Both prompts contain exactly 7,168 tokens. `data/prompt.txt` is the technical
research note. `data/alice_prompt.txt` is the public-domain opening of Lewis
Carroll's *Alice's Adventures in Wonderland*, from Project Gutenberg ebook 11.
The exact retained UTF-8 text fixes whitespace. `data/alice_provenance.json`
records the source, text and token digests, and selection time before execution.
The sampler draws and all emitted tokens are retained for deterministic replay.

```sh
python -m pip install -r requirements.txt
python code/rollout.py --model "$MODEL_PATH" --tokens 1024 \
  --prompt-ids data/technical_replay_prompt_ids.npy \
  --replay results/technical_replay.json --tag reproduce_technical
python code/rollout.py --model "$MODEL_PATH" --tokens 1024 \
  --prompt-ids data/alice_prompt_ids.npy \
  --replay results/alice.json --tag reproduce_alice
python code/baseline.py --model "$MODEL_PATH" --trace reproduce_technical
python code/baseline.py --model "$MODEL_PATH" --trace reproduce_alice
```

The replay option asserts agreement with every recorded token. Omit it for a
fresh sample. A different numerical environment can change logits and replayed
tokens; saved finite-array witnesses remain checkable independently of such a
new execution. The dense control uses the checked run's random words, but
follows its own generated history. Shared-word token agreement is not a TV test.

The technical record is unchanged from the preceding revision. The Alice
execution and its dense control were rerun sequentially for this revision at
concurrency one, with four CPU compute threads. No inference or numerical-audit
job overlapped the pair. `results/environment_alice.json` records the environment;
`results/unchanged_execution_sources.json` records hashes of the unchanged
inference and sampling sources. The new sampled acceptance count happens to
match the former second-prompt count; the prompt, words, tokens and curves are
new.

The technical gate accepts 1,019 candidates and uses five fallbacks. The Alice
gate accepts 961 candidates and uses 63 fallbacks: four KL rejections and 59
precision-guard rejections. These are two sampled continuations, not population
acceptance estimates. The enforced conditional cap is `1/600000000` in both.
Complete timing records include loading, prompt preparation, prefill, cache
allocation and updates, score scans, sorting, centroid construction, both
terminal paths, integer checks, sampling and streamed trace output. The paper
reports the recorded totals. The compatible dense control is faster in both
comparisons; no inference acceleration is claimed.

## Reconstruct the first-intervention witnesses

The following commands overwrite reconstructed data and results. Preserve the
release before comparing a fresh run with it.

```sh
export OPENBLAS_NUM_THREADS=1
python code/probe.py fp32
python code/probe.py bf16_attention
python code/probe.py fp64_attention
python code/interventions.py
python code/reachable.py
make output-check
```

The probes use `MODEL_PATH`. They reconstruct `data/tail_weights.npz`, which is
not included as a duplicate model-weight archive. `reachable.py` also checks the
all-vocabulary upper gain. The old first-intervention and packing arrays are
unchanged in this release. The exact one-read sequence TV remains
`211699215287897/18446744073709551616`, and the necessary group floor remains
18,633 of 64,512 interactions for the stated envelope class.

## Build the paper

```sh
python code/make_paper_data.py
make paper
```

pdfLaTeX, BibTeX, newtx, natbib, TikZ and pgfplots are required. The build script
rejects final-pass warnings and overfull or underfull boxes. Every figure is
native TeX using retained data. The LaTeX sources are self-contained.

`formal/FiniteKernel.lean` remains a partial development. For this public release,
all six theorems compile with Lean 4.34.1 and the pinned Mathlib revision, without
`sorry` or `sorryAx` dependencies. Reproduce the check with:

```sh
cd formal
lake exe cache get
lake build
lake env lean CheckAxioms.lean
```

`formal/check_status.json` records the check and its scope. The paper describes
these checked declarations and the results that remain outside the formalization.
The checker implementation remains trusted numerical code; `AUDIT.md` states what
the exact witnesses and tests establish.

## Citation

If you use this paper, code, or replay data, please cite Samuel Mausberg,
*A reachable-state separation between attention compression and its certificates*
(2026). [CITATION.cff](CITATION.cff) supplies the preferred paper citation for
GitHub's **Cite this repository** feature.

The October 4 rerun passed the replay, arithmetic, retained output, paper build,
and six-declaration Lean checks. See [the validation record](results/release_checks_2026_10_04/README.md).

## License

The manuscript and original research material use [CC BY 4.0](LICENSES/CC-BY-4.0.txt).
Original code uses [MIT](LICENSES/MIT.txt). Third-party notices remain in effect.
See [LICENSE](LICENSE) for the scope and [CITATION.cff](CITATION.cff) for citation metadata.
