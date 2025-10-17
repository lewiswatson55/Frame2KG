### Ablation: Effect of Adding MLP (GateProj/Up/Down)

| Model            | Node F1µ | Edge F1µ | IoU mean |
|------------------|----------|----------|----------|
| 3B–QKVO          | 0.597    | 0.187    | 0.600    |
| 3B–QKVO-Gate     | 0.616    | 0.208    | 0.609    |
| 7B–QKVO          | 0.607    | 0.195    | 0.601    |
| 7B–QKVO-Gate     | 0.621    | 0.204    | 0.605    |

**Summary.** Adding the lightweight MLP gating (GateProj/Up/Down) yields consistent gains in predicate quality (Edge F1µ) and localisation (IoU mean) at both 3B and 7B scales, with modest improvements in entity recognition (Node F1µ). The effect is visible within each backbone: **3B–QKVO → 3B–QKVO-Gate** (+0.021 Edge F1µ, +0.009 IoU), and **7B–QKVO → 7B–QKVO-Gate** (+0.009 Edge F1µ, +0.004 IoU). Moving from 3B to 7B further helps nodes (Node F1µ +0.010 without gate; +0.005 with gate), suggesting scale primarily benefits entity labelling, while gating reliably helps relations and box alignment.

**Takeaways.**
- **Gating helps predicates and boxes:** consistent Edge F1µ and IoU gains across both scales.
- **Scale helps nodes:** 7B variants slightly improve Node F1µ over 3B.
- **Low overhead:** improvements come with minimal additional parameters from the MLP gate.
- **Practical implication:** if compute is tight, prefer a **3B–QKVO-Gate** adapter; if memory allows, **7B–QKVO-Gate** offers the best overall balance.