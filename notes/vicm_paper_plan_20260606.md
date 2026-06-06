# VICM Paper Plan 2026-06-06

## Core Thesis

This paper should not claim that VICM universally improves ordinary humanoid
walking. The defensible thesis is:

> SRBM remains very strong when WBC, foot placement, and feedback are present.
> VICM is useful in regimes with large limb inertia, significant angular
> momentum variation, and operation near the hybrid stability boundary. In
> those regimes, configuration-dependent centroidal inertia and the
> `-I_G^{-1} dot(I_G) omega` term improve angular-dynamics consistency and can
> enlarge closed-loop stability margins.

The paper's center of gravity should be model fidelity and boundary robustness,
not perfect velocity/yaw tracking.

## Proposed Contributions

1. A variable-inertia centroidal MPC formulation for humanoid locomotion that
   uses the whole-body centroidal inertia `I_G(q)` instead of a fixed SRBM
   inertia.
2. A practical frozen-inertia MPC implementation where `-I_G^{-1} dot(I_G)
   omega` is injected as a linear angular-rate dynamics term in the continuous
   `A_c` matrix.
3. An experimental comparison showing that the model advantage appears mainly
   under high limb-inertia turning and recovery-boundary conditions, while
   nominal walking remains well handled by SRBM plus WBC.
4. Ablation evidence separating the effects of configuration-dependent inertia,
   the `dot(I_G) omega` dynamics term, and numerical filtering.

## Theory Narrative

### SRBM Baseline

SRBM assumes the robot behaves as a single rigid body with fixed inertia:

```text
dot(omega) = I_0^{-1} tau
```

This is computationally efficient and surprisingly robust in closed-loop
walking because WBC, contact switching, swing-foot planning, and feedback absorb
many modeling errors.

### VICM Model

VICM replaces fixed inertia with the centroidal inertia around the whole-body
CoM:

```text
h_G ~= I_G(q) omega
dot(h_G) = tau_ext
I_G(q) dot(omega) + dot(I_G) omega = tau_ext
dot(omega) = I_G^{-1} tau_ext - I_G^{-1} dot(I_G) omega
```

In the current MPC implementation, `I_G` is frozen over the local prediction
linearization, while the term

```text
-I_G^{-1} dot(I_G) omega
```

is included in `A_c` as a linear angular-rate dynamics term. This is more
theoretically consistent than treating it only as a constant feedforward torque,
because it changes the local angular-rate evolution rather than only biasing
one instantaneous prediction.

### Numerical Regularization

`dot(I_G)` is obtained from finite differences, so filtering is not arbitrary
tuning. It should be presented as numerical regularization for a differentiated
quantity entering the MPC dynamics. The no-filter ablation confirms that
removing the filter degrades boundary robustness.

Avoid saying:

```text
The scale/clamp is necessary for the theory.
```

Say instead:

```text
The model term is theoretically motivated; finite-difference estimation of
dot(I_G) requires numerical regularization. The ablation isolates this
implementation effect.
```

## Experiment Structure

### Experiment 1: High-Inertia Turning Stability Boundary

Purpose:

Show that the advantage is not in ordinary straight walking but appears near
the turning stability boundary under increased leg inertia.

Final condition:

- Lambda-form leg inertia scaling.
- `vx command = 1.5 m/s`.
- `tSwing = 0.45 s`.
- WBC PosRot attitude scale `0.35`.
- MPC weights:

```text
L_diag = 50 50 80 1 200 1 1 1 10 100 10 1
```

- Sinusoidal yaw-rate:

```text
wz_amp = 0.25 rad/s
period = 4 s
start = 4 s
```

Key result:

At high lambda, VICM-Ac expands the stable region relative to SRBM. In the full
sweep, `lambda=1.7` is especially useful because SRBM falls while VICM-Ac
survives.

Important caveat:

Do not write that the robot actually tracks `1.5 m/s` forward speed. The command
is `1.5 m/s`, but measured body-forward speed in representative turning trials
is much lower, around `0.7-0.8 m/s`.

Useful files:

- Sweep summary:
  `record/lambda_filter_turn_exp1_20260604_220713/summary.csv`
- Paired summary:
  `record/lambda_filter_turn_exp1_20260604_220713/paired_summary.csv`
- Representative tracking:
  `record/lambda_filter_turn_exp1_20260604_220713/exp1_lam1p5_rep1_body_forward_tracking.png`
- Main figure:
  `figures/lambda_filter_exp1_summary_high_lambda_ge1p0_no_fall_count_20260605.png`

Suggested paper wording:

> Experiment 1 evaluates the effect of increasing leg inertia under a
> sinusoidal turning command. The commanded forward velocity is held fixed, but
> the evaluation focuses on survival and angular response rather than exact
> velocity tracking, because the gait and WBC stack limit the realized forward
> speed in this high-demand turning condition.

### Experiment 2: Nominal Velocity Response Demonstration

Purpose:

Show that under nominal `lambda=1` conditions, both SRBM and VICM-Ac produce
reasonable forward/lateral velocity responses. This is not meant to prove VICM
tracking superiority.

Final condition:

- `lambda = 1`.
- `tSwing = 0.25 s`.
- Forward ramp: `vx = 0.6 -> 1.2 m/s`.
- Lateral ramp: `vy = 0.15 -> 0.30 m/s`.
- No yaw-rate command.

Key result:

Both controllers run stably and have similar tracking quality.

Metrics from representative run:

```text
forward SRBM: vx_ref=1.200, vx_mean=1.112, platform_rms=0.116
forward VICM: vx_ref=1.200, vx_mean=1.124, platform_rms=0.106
lateral SRBM: vy_ref=0.300, vy_mean=0.298, platform_rms=0.113
lateral VICM: vy_ref=0.300, vy_mean=0.301, platform_rms=0.107
```

Useful files:

- `record/exp2_velocity_ramp_lam1_20260606_165735/exp2_velocity_ramp_tracking_lam1.png`
- `record/exp2_velocity_ramp_lam1_20260606_165735/summary.csv`

Suggested paper wording:

> Under nominal speed-ramp commands, both SRBM and VICM-Ac exhibit comparable
> tracking behavior. This supports the interpretation that the proposed model is
> not required for ordinary velocity tracking when the rest of the locomotion
> stack is effective.

### Experiment 3: Model Component Ablation

Purpose:

Isolate the role of each VICM component around the informative boundary.

Final condition:

- `lambda = 1.7`.
- `vx command = 1.5 m/s`.
- `wz_amp = 0.25 rad/s`.
- `tSwing = 0.45 s`.
- 3 repeats.

Compared variants:

1. SRBM.
2. VICM-Ig: configuration-dependent inertia only.
3. VICM-Ac: `I_G(q)` plus Ac-injected `dot(I_G) omega` term.
4. VICM-Ac no filter.
5. VICM affine tau.

Key result:

```text
SRBM:              16.70 s, 3/3 falls
VICM-Ig:           23.02 s, 3/3 falls
VICM-Ac:           30.00 s, 0/3 falls
VICM-Ac no filter: 27.03 s, 2/3 falls
VICM affine tau:   30.00 s, 0/3 falls
```

Interpretation:

- `I_G(q)` alone helps, but does not fully stabilize the boundary case.
- The Ac-injected `dot(I_G) omega` term is the most important part of the
  current VICM implementation.
- Filtering improves robustness of the finite-difference `dot(I_G)` estimate.
- The affine tau variant also performs well, but VICM-Ac is more aligned with
  the local linear dynamics interpretation.

Useful files:

- `record/exp3_model_ablation_lam1p7_20260606_175527/summary.csv`
- `record/exp3_model_ablation_lam1p7_20260606_175527/trials.csv`
- `record/exp3_model_ablation_lam1p7_20260606_175527/exp3_ablation_summary_clean.png`
- `record/exp3_model_ablation_lam1p7_20260606_175527/exp3_ablation_tracking.png`

Suggested paper wording:

> The ablation shows a monotonic improvement in the boundary turning condition:
> fixed-inertia SRBM fails earliest, configuration-dependent inertia alone
> extends survival, and the Ac-injected inertia-rate dynamics enables full
> recovery across all repeats.

### Experiment 4: Push Recovery Map

Purpose:

Show that VICM-Ac can enlarge recovery regions under external disturbances,
especially lateral pushes.

Final condition:

- `lambda = 1.7`.
- `vx command = 1.2 m/s`.
- No yaw-rate command.
- Push duration `0.15 s`.
- Push start `8.0 s`.
- Directions: `0, 90, 180, 270 deg`.
- Forces: `0, 100, 200, 300, 400 N`.

Key result:

The strongest difference appears for lateral `90 deg` pushes:

```text
90 deg, 200 N: SRBM falls, VICM-Ac recovers
90 deg, 300 N: SRBM falls, VICM-Ac recovers
90 deg, 400 N: both fall
```

Recovery boundary summary:

```text
SRBM:
0 deg   -> 300 N
90 deg  -> 100 N
180 deg -> 400 N
270 deg -> 200 N

VICM-Ac:
0 deg   -> 300 N
90 deg  -> 300 N
180 deg -> 400 N
270 deg -> 200 N
```

Important caveat:

The push map is not perfectly monotonic because contact timing, gait phase, and
hybrid switching interact with the disturbance. Do not claim a strict scalar
"maximum recoverable force" unless the grid is repeated across phases and
trials. Present it as a direction-force recovery map.

Useful files:

- `record/exp4_push_recovery_lam1p7_20260606_180914/summary.csv`
- `record/exp4_push_recovery_lam1p7_20260606_180914/recovery_boundary.csv`
- `record/exp4_push_recovery_lam1p7_20260606_180914/exp4_recovery_heatmap_clean.png`
- `record/exp4_push_recovery_lam1p7_20260606_180914/exp4_dir90_F200_response.png`

Suggested paper wording:

> The recovery map shows that the benefit is direction dependent. The most
> visible improvement appears for lateral pushes, where VICM-Ac recovers at
> disturbance levels that cause SRBM to fall. This is consistent with the
> proposed model being most useful when angular momentum and whole-body inertia
> variations interact strongly with contact transitions.

## Recommended Paper Outline

### 1. Introduction

Start with the tension:

- SRBM is widely used because it is simple and robust.
- Humanoid robots have non-negligible limb masses, so fixed inertia can be
  inaccurate during dynamic turning or recovery.
- However, a better model does not automatically mean better closed-loop
  performance because WBC and feedback compensate.

End with the narrowed claim:

> This paper studies when variable-inertia centroidal dynamics matter in
> humanoid MPC, and shows that the benefit appears near angular-momentum-rich
> stability boundaries rather than in nominal walking.

### 2. Related Work

Suggested categories:

1. SRBM / centroidal MPC for legged robots.
2. Whole-body MPC and centroidal dynamics.
3. Variable inertia / angular momentum compensation.
4. Comparative model studies in legged locomotion.

Be careful:

Do not oversell that prior papers prove huge tracking differences. Many also
show modest differences except in recovery/boundary cases.

### 3. Model and MPC Formulation

Subsections:

1. SRBM dynamics.
2. Variable-inertia centroidal dynamics.
3. Frozen-inertia local MPC model.
4. Ac injection of `-I_G^{-1} dot(I_G) omega`.
5. Numerical filtering and implementation details.

### 4. Controller Stack

Explain honestly:

- MPC optimizes contact forces.
- WBC tracks base/foot/contact tasks and can modify MPC forces.
- Foot placement and contact switching are active.
- Therefore, closed-loop differences are expected to be smaller than open-loop
  model differences.

This section helps reviewers understand why SRBM is strong and why VICM does
not dominate every condition.

### 5. Experimental Setup

Include:

- Robot model and mass/inertia scaling.
- SRBM/VICM variants.
- WBC PosRot scale.
- MPC weights.
- Metrics:
  - survival/final time,
  - yaw/wz RMS,
  - one-step omega prediction error,
  - torso angle,
  - push recovery map.

### 6. Results

Suggested order:

1. Experiment 1: lambda turning sweep.
2. Experiment 2: nominal velocity response.
3. Experiment 3: ablation.
4. Experiment 4: push recovery.

This order tells a clean story:

```text
Where does VICM help? -> It helps near turning boundary.
Does it ruin nominal behavior? -> No, nominal response is comparable.
Which component matters? -> Ac-injected inertia-rate dynamics.
Does it generalize to disturbances? -> Yes, especially lateral recovery.
```

### 7. Discussion

Key points:

- SRBM is not weak; it is strong in a full locomotion stack.
- VICM helps when model mismatch aligns with angular momentum/contact boundary.
- Filtering is a numerical implementation requirement for `dot(I_G)`, not the
  theoretical contribution.
- The realized forward speed in high-demand turning is below the commanded
  speed, so velocity tracking should not be the central claim.
- Push recovery is phase/hybrid dependent, so recovery maps should be treated
  as empirical regions rather than strict monotonic thresholds.

### 8. Conclusion

Close with:

- VICM is not a universal replacement for SRBM.
- It is a targeted improvement for angular dynamics consistency.
- The practical benefit is boundary robustness in high-inertia turning and
  lateral disturbance recovery.

## Claims to Avoid

Avoid:

```text
VICM always improves locomotion stability.
VICM achieves better tracking in all conditions.
The robot walks at 1.5 m/s in Experiment 1.
The push recovery boundary is a strict monotonic force threshold.
The filter is part of the physical model.
```

Use instead:

```text
VICM-Ac improves stability margin in selected boundary conditions.
Nominal tracking is comparable between SRBM and VICM-Ac.
Experiment 1 uses a commanded forward speed of 1.5 m/s.
Push recovery is summarized as a direction-force empirical recovery map.
Filtering regularizes finite-difference estimation of dot(I_G).
```

## Next Writing Tasks

1. Write the abstract around the narrowed claim.
2. Convert the theory derivation into a compact Section 3.
3. Insert Experiment 1-4 figures and write one paragraph per figure.
4. Add a Discussion section that explicitly explains why SRBM remains strong.
5. Prepare a limitation paragraph about velocity tracking and hybrid
   nonmonotonic recovery maps.

