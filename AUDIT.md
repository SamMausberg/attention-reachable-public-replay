# Numerical and interpretive audit

## Certificate class

The local readout inequalities remain 0.28 <= C_1*(x) <= 0.746. They concern the
real readout map on a Euclidean ball around a reached binary32 vector. They do
not lower-bound the derivative of the final feed-forward block followed by the
readout. Section 3.3 now explicitly restricts the refusal to scalar envelopes
that pass the feed-forward residual's incoming error with coefficient one and
add a nonnegative branch allowance. A local proof that exploits contraction
across that block is outside this claim. No composite derivative bound was
computed or asserted for this revision.

The weighted packing inequality, nine clique witnesses, exact allocation and
readout/projection witnesses are unchanged. The necessary group floor is 18,633
of 64,512 final-layer interactions. The universal tenfold budget is 6,451; the
concrete candidate uses 6,444 groups. Exact optimality is claimed for allocation
of the verified lower-bound curves, not for finding maximum cliques or for
minimizing the true output error over all partitions.

## Public prompt replacement

The second prompt is now the opening of Lewis Carroll's public-domain *Alice's
Adventures in Wonderland*. It has exactly 7,168 tokens under the frozen checkpoint
vocabulary and tokenizer. The retained UTF-8 prompt round-trips to those tokens.
The source and digests were fixed before running the replacement execution.
The former second prompt, its token array, generated continuation, paired
snapshots, random-word traces and dense control are excluded from this bundle.
The manuscript has no historical attribution note.

The new checked execution and dense control use unchanged inference and sampler
sources, the same checkpoint digest, four CPU threads and concurrency one.
They ran sequentially. The dense control receives the checked run's words but
follows its own emitted history. Differences under this coupling do not measure
TV. The original technical record is retained, rather than replacing its timing
with a measurement from this revision.

## Arithmetic and sequence law

The output checks regard stored binary32 logits as exact dyadic inputs. They
therefore compare the actual saved output laws without a numerical enclosure of
every upstream BLAS operation. The per-step gate uses a 96-bit fixed-point
exponential enclosure, 256-bit products and checked range guards. Its integer
KL cap is 1/600000000. Independent Python calculations use 256-bit exponential
enclosures. The sampler's 64-bit probability grid is now defined when Theorem 3
introduces P_64 and Q_64.

The complete-sequence theorem assumes independent uniform sampler words and
correct execution of the specified checker. It is uniform over token histories,
not estimated from the two traces. The unchanged one-read identity relies on
all KV writes preceding the changed read. Earlier-layer compression requires
another state invariant. The BF16/FP32 controls are arithmetic variants of this
executor, not complete production-engine comparisons.

The package retains the original arithmetic and compact replay records, with
fresh revision checks in results/revision_checks. The independent output replay
now checks all 22 retained rollout snapshots: 21 second-moment comparisons and
one precision-guard fallback count law. This extends the audit, without changing
the first-intervention laws or refusal arrays. No full 1,024-token sampled trace
is an empirical estimate of sequence TV.

## Prior work

Wei and Liu, arXiv:2608.15810v1, Section 7, explicitly report a gap between their
certified witness and served-output law. Their 1,064-fold value is a ratio of
deployed and required witness thresholds for a 5% per-read attention-TV target;
it is not a ratio of full-sequence distances. The query envelope accounts for
only a factor of 1.50. The revised paper cites this as precedent, while preserving
the distinction between their per-read and cumulative-loss guarantees and the
present complete-token-sequence guarantee. Their paired-logit comparison is also
credited. Kang et al., arXiv:2609.27981v1, instead certify task-degradation risk
under a specified calibration population.

## Remaining trust boundary

The Python/C++ checkers and the transformer executor are trusted implementations,
not verified compiler outputs. The partial Lean sources remain unchecked. The
passing diagnostic pays for dense reference logits and all score/value reads;
its value here is the separation between a scalar proof representation and an
output-aware check, not an acceleration result. Runtime and acceptance records
are measurements on the specified machine and two sampled histories.
