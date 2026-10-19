# Findings

**No empirical result yet.** No model has been trained or evaluated here. This
file records analytic results and their claim boundary. Each is pinned by
`tests/test_theory.py`, and several were corrected or added after Codex's
adversarial review on 2026-10-19.

## Analytic results (2026-10-19)

These are expected gradients for the stylized game, where a trajectory reduces
to one action from a two-action policy. Notation:

- g = defection premium
- b = benefit my cooperating gives the other player
- n = group size

1. **The post's round robin is CDT pressure.**
   - With ρ = 0, defectors out-earn cooperators in every group (the post's
     footnote 7).
   - The exact advantage gap is −((n−1)g + b)/n, which is −17/16 at T = 2 and
     n = 16.
2. **Coupled grading matches EDT's gap with correlation ρ.** In the limit, its
   advantage gap equals EDT's expected-utility gap for a twin regressing on the
   agent with coefficient ρ.
   - **Finite groups.** The trainer sees an extra −(1−ρ)b/n. At n = 16 the
     cooperation threshold rises from 0.5 to 0.531 at T = 2, and from 0.5 to
     0.625 at T = 10 near p = 0.
   - **Interpretation.** At ρ = 1 the problem is a bandit whose rewarded action
     agrees with EDT/FDT. Whether it teaches either is untested.
3. **Policy-level credit is policy selection.** Crediting every copy's tokens is
   the total derivative of the selfish symmetric value.
   - In symmetric games, it equals a welfare reward.
   - Group centering scales it by (n−2)/n, so it vanishes at n = 2.
   - A fixed focal role in asymmetric games separates selfish policy selection
     from welfare.
4. **Donation-game degeneracy.** When R + P = S + T, as at the post's T = 2,
   welfare and ρ = 1 advantages are exactly proportional, 14/15 at n = 16. The
   two can only be contrasted in non-additive games.
5. **Shared vs private coin.**
   - ρ = 1 always converges to cooperation.
   - Policy-level credit converges to argmax V: 3/4 at T = 4, 9/16 at T = 10.
   - At T = 10 above p = 9/16, the two push in opposite directions.
6. **Zero-sum self-play is not neutral.** V is flat, so EDT and policy selection
   are indifferent, but focal credit can strictly prefer an action. Zero-sum
   self-play is pure focal-credit pressure.
7. **Stranger controls need specified rates.**
   - Selfish defection strength depends on the stranger's cooperation rate q,
     except in donation games.
   - Stranger-welfare at T = 10 trains cooperation only while q < 9/16.

## Boundary

- These describe expected gradients in a stylized game. They are not claims
  about what a language model learns. Reasoning, framing, priors and parameter
  sharing across prompts sit between the gradient and any attitude change.
- Claims 1–7 assume the trainer is plain group-centered REINFORCE. The hosted
  trainer's std normalization, clipping, KL penalty, number of epochs and loss
  reduction are unverified (design.md, Step 0).
- vLLM's seeded sampler couples token noise across prompts. That is
  source-verified, not execution-qualified, and token coupling is not action
  coupling.
