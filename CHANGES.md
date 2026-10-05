# Final revision: changes and checks

## 1. Public-domain replacement and new executions

The second prompt is now the opening of Lewis Carroll's *Alice's Adventures in Wonderland*, from Project Gutenberg ebook 11. It contains exactly 7,168 tokens under the frozen checkpoint tokenizer, with no added beginning-of-sequence token or chat wrapper. The retained UTF-8 text round-trips to the saved token array. The excerpt, token array and their SHA-256 digests were fixed before the new execution. The source and whitespace convention are recorded in `data/alice_provenance.json`.

The checked 1,024-output execution and its compatible dense control were rerun sequentially on the four-thread AMD EPYC 9V74 CPU at concurrency one. The checkpoint, inference code, centroid algorithm, KL cap and sampler are unchanged. The dense control uses the checked execution's recorded random words and follows its own emitted history.

The new checked execution took **267.006 seconds**, replacing the former second execution's 214.832 seconds in the paper. The new dense control took **184.962 seconds**, replacing 175.120 seconds. The new checked execution has 961 accepted, nonidentical candidate logit vectors and 63 fallbacks. Those totals happen to equal the former second execution's totals, but come from a new prompt and a new sample. Four fallbacks exceed the KL allowance; 59 trigger precision guards. The paper now gives this breakdown.

All second-execution records have been replaced: input IDs, emitted tokens, random words, paired-logit snapshots, exact certificate records, diagnostic TV values, phase timings, logs and dense-control output. The new checked prefill took 15.345032318 seconds and decode took 251.194053496 seconds; the corresponding dense prefill and decode took 15.242769370 and 169.250754880 seconds. Full-precision timing values and diagnostic statistics remain in the JSON records. None of these sampled diagnostics is used to infer full-sequence TV.

## 2. Figures and numerical consistency

Figure 4 now plots the new Alice trace and labels it “Alice prompt”. Its dense-fallback markers were regenerated from the technical and Alice executions. Precision-guard failures remain omitted where no numerical KL bound was computed. `figures/alice.dat` replaces the former second-prompt curve.

The technical curve, packing frontier and first-intervention TV figure are byte-for-byte unchanged. Apart from the newly measured second-prompt timing macros, every existing numerical macro is unchanged. The acceptance macros retain their values because the rerun produced the same counts. All 30 displayed mathematical expressions are unchanged. The first-intervention, packing and technical-execution arrays and records were preserved, with 56 file digests checked against the preceding replay bundle.

## 3. Corrected scope of Section 3.3

Section 3.3 now defines the refused class to pass incoming error through the final feed-forward residual with coefficient one and add a nonnegative allowance for the feed-forward branch. It expressly distinguishes this bookkeeping rule from a lower bound on the block's local derivative or on the block-readout composite. Local enclosures that exploit contraction across the block are outside this refusal.

The claim that assigning unit gain is a generous relaxation has been removed. Only setting the nonnegative branch allowance and inherited KV uncertainty to zero is used to lower-bound the declared certificate within the stated class. The abstract, introduction, Figure 1 caption and conclusion now refer to the specified scalar-envelope certificate where needed. No composite derivative lower bound was computed or asserted. The refusal equation, gains, optimal allocation and 18,633-group floor are unchanged under the explicit class restriction.

## 4. Theorem 3 notation

Theorem 3 defines `P_64` and `Q_64` at their first occurrence as the complete-sequence laws using the sampler's 64-bit integer probability grid and uniform random word. The network arithmetic and all mathematical statements and constants are unchanged.

## 5. Prior-art attribution

Section 7 explicitly cites Wei and Liu, arXiv:2608.15810v1, Section 7, as precedent for a gap between a certified witness and served output. The text describes their reported 1,064-fold ratio between the deployed witness threshold and the threshold required by their sound law for a 5% **per-read attention-TV** target, and the factor of 1.50 attributable to their query envelope. It does not describe this threshold ratio as a ratio of full-sequence distances. Their earlier paired-logit comparison remains credited.

The relevant statements were checked in the original arXiv text. The existing comparison with Kang et al., arXiv:2609.27981v1, was rechecked and retained. A Gutenberg citation was added for the replacement prompt, and the publisher's checkpoint digest was rechecked.

## 6. Public replay contents and provenance

The historical attribution note has been removed from the manuscript. The public replay contains neither the former copyrighted prompt nor its token encoding, continuation, paired snapshots, random-word trace or dense control. The revised source archive contains only the current figure data, rather than retaining the obsolete curve as an unused file.

The PDF and replay bundle are still described as being linked from Samuel Mausberg's GitHub profile. A checkpoint-fetch helper verifies the required SHA-256 before loading or installing a downloaded file. The delivered replay excludes the full checkpoint and font files. Its downloader's existing-file verification and mismatch rejection were tested; the network-download path was not executed in this environment because the hash-matching checkpoint was already available.

The replay documentation and figure-generation script now use the Alice files. The independent output check was extended to the new rollout. The source and public replay build instructions use an explicit shell invocation so extraction need not preserve executable permissions. The AI-assistance acknowledgment is unchanged.

## 7. Validation performed

The replacement prompt's tokenizer round-trip and checkpoint digest were checked. The new executions both completed all 1,024 outputs. Every recorded accepted step satisfies the KL cap by exact integer comparison. Saved snapshot hashes and emitted-token selections agree with the recorded count laws.

The arithmetic tests passed 35 sampler cases, 33 independent moment comparisons, two expected precision-guard cases, and their edge controls. The compact replay rechecked all nine cliques, the three decisive allocation budgets and the rational sequence budget. The independent output replay checked eight first-intervention categorical laws and all 22 retained paired snapshots: 21 second-moment comparisons and one precision-guard fallback law. Its original first-intervention comparisons and eleven technical moment checks are unchanged.

The release audit passed 1,101 assertions, including the per-admitted-step integer comparisons, numerical macros, plotted data, unchanged mathematical displays, unchanged execution sources and preserved witnesses. The 13-page paper compiled without final-pass warnings, unresolved references, or overfull/underfull boxes. All pages were rendered and inspected. The complete LaTeX archive was then extracted into a clean directory and rebuilt; its page text agrees with the delivered PDF.

No new Lean verification is claimed. The stated numerical implementation and independent-uniform-word assumptions remain the trust boundary.

## 8. Source formatting

The Python sources are formatted with `ruff format` and the C++ sources with `clang-format`; `ruff.toml` and `.clang-format` record the settings. The six Python execution sources hashed in `results/unchanged_execution_sources.json` were formatted only: their Python syntax trees are identical to the hashed bytes, which remain at commit `d3f5acd`. Both C++ sources compile to identical machine code with the Makefile flags. In the remaining Python files, unused standard-library imports were removed and import blocks sorted; their other syntax is unchanged. `formal/FiniteKernel.lean` received layout changes only, and `formal/check_status.json` records its new digest.

After formatting, `make all test check output-check` reproduced the retained arithmetic, compact-replay and output-verification records apart from elapsed time, and `lake build` and `lake env lean CheckAxioms.lean` reproduced the six axiom lists. Check formatting with:

```sh
ruff check code && ruff format --check code
clang-format --dry-run --Werror code/*.cpp
```
