# Findings

**No empirical result yet.** No model has been trained or evaluated here. This
file records only analytic results and the claim boundary around them.

## Analytic results (2026-10-19)

These hold in expectation for the stylized game, where each trajectory reduces
to one action from a two-action policy. `tests/test_theory.py` verifies each
one.

1. **The post's round robin is CDT.**
   - With ρ = 0, a defector earns at least as much as a cooperator in every
     group, for every T ≥ R ≥ P ≥ S (the post's footnote 7).
   - The expected advantage gap is −g(p).
2. **Coupled grading is EDT with correlation ρ.**
   - Scoring each trajectory against its own action with weight ρ gives exactly
     EDT's expected-utility gap for a twin whose action regresses on the
     agent's with coefficient ρ.
   - Training favors cooperating iff ρ > g / (R − P + g).
   - That threshold is 1/2 at every population state for the post's +2 game,
     and rises from 1/2 to 0.9 for the +10 game.
3. **Policy-level credit is welfare, in symmetric games.**
   - Crediting through both copies gives the total derivative of the selfish
     symmetric policy value.
   - In a symmetric round robin, that equals a utilitarian welfare reward.
     Training cannot separate "selfish policy selection" from "altruism"; only
     evaluation can.
4. **Shared coin vs private coin.** At high temptation the two non-CDT schemes
   disagree.
   - ρ = 1 always converges to cooperation.
   - Policy-level credit converges to the mixed optimum argmax V: 3/4 at T = 4,
     9/16 at T = 10.
   - Appendix C's randomization problem arises only for private coins.
5. **Small groups add a little defection pressure.** Others' rewards in the
   baseline depend on one's own action, an O(1/n) effect.

## Boundary

- These are statements about expected gradients on the stylized game, not about
  what a language model learns. A model's reasoning, framing and prior sit
  between the gradient and its decision-theoretic attitudes. That gap is what
  the experiment exists to measure.
- The claim that ρ can be made physical through shared sampling seeds is a
  hypothesis about the inference stack. It has not been checked.
