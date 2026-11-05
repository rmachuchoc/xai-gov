# Worked examples (v2)

Computed by the package; see scripts/worked_examples_v2.py.

## 1. One period of the serial model, and the fill rate beta

Retailer with 3 units facing a demand of 2 per period; it orders 20 at period 0 from a
DC holding 12. The DC sees that order at period 1, ships 12 and owes 8; it orders 8
from the plant, which ships them at period 2; the DC forwards them at period 3. All
lead times are 1.

| t | customer demand | served on time (beta) | shipped to customer | retail backlog | retail stock | retail in transit | retail awaiting | DC stock | DC backlog |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.0 | 2.0 | 2.0 | 0.0 | 1.0 | 0.0 | 20.0 | 12.0 | 0.0 |
| 1 | 2.0 | 1.0 | 1.0 | 1.0 | 0.0 | 12.0 | 8.0 | 0.0 | 8.0 |
| 2 | 2.0 | 2.0 | 3.0 | 0.0 | 9.0 | 0.0 | 8.0 | 0.0 | 8.0 |
| 3 | 2.0 | 2.0 | 2.0 | 0.0 | 7.0 | 8.0 | 0.0 | 0.0 | 0.0 |
| 4 | 2.0 | 2.0 | 2.0 | 0.0 | 13.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | 2.0 | 2.0 | 2.0 | 0.0 | 11.0 | 0.0 | 0.0 | 0.0 | 0.0 |

beta over the six periods = 11 / 12 = 0.9167. A unit shipped late counts in the backlog, never in beta, so beta cannot exceed 1.

## 2. The admissible envelope and the three operators

Retail state: inventory 4, in transit (on order) 6, backlog 0, last demand 10, capacity 120. With one period of lookahead the projected position 10 - 10 = 0 reaches the critical level (5), so SAFE-2 and SAFE-3 engage and require an order of at least 5 + 10 - 4 - 6 = 5.

Envelope: [5, 120] (SAFE-2 requires ordering at least 5.0). F-backup's level is S = s + Q = 60, so its order is max(60 - 10, 0) = 50, clamped to the envelope.

| proposal | admissible | projection executes | F-null executes | F-null admissible | F-backup executes |
| --- | --- | --- | --- | --- | --- |
| 0.0 | False | 5.0 | 0.0 | False | 50.0 |
| 30.0 | True | 30.0 | 30.0 | True | 30.0 |
| 150.0 | False | 120.0 | 0.0 | False | 50.0 |

Projection moves an inadmissible proposal to the nearest bound; F-null replaces it with 0 even though 0 is itself inadmissible here; F-backup replaces it with an admissible base-stock order.

## 3. The aGRAPA bet, step by step

Five paired differences of beta (treatment minus control). The bet for replicate t
uses only replicates 1..t-1: lambda_t = clip(mu / (sigma2 + mu^2), 0, 1/2), with a
prior mean 0 and variance 1. The wealth is the e-value; the plan rejects when it
reaches 240.

| t | d_t | mu_{t-1} | sigma2_{t-1} | lambda_t | 1 + lambda_t d_t | wealth K_t |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| 2 | 0.06 | 0.05 | 0.5012 | 0.0993 | 1.006 | 1.006 |
| 3 | 0.08 | 0.0533 | 0.3342 | 0.1582 | 1.0127 | 1.0187 |
| 4 | -0.02 | 0.06 | 0.2507 | 0.2359 | 0.9953 | 1.0139 |
| 5 | 0.09 | 0.044 | 0.2014 | 0.2164 | 1.0195 | 1.0336 |

With lambda at most 1/2 and differences near 0.05, the log wealth grows by about 0.025 per replicate, which is why reaching 240 takes over 220 replicates.

## 4. The intervention threshold: myopic and optimal

c = 1, L = 10, gamma = 0.95. Myopic: b_m = c / L = 0.1: intervene once one period of expected loss b*L exceeds the cost.

Optimal: intervene at the first belief where the margin D(b) = b*L + gamma*(C(b) - V(0)) - c turns nonnegative; C(b) - V(0) is what waiting costs in later periods, so b* = 0.06 is earlier than the myopic 0.1.

| belief b | D(b) | decision |
| --- | --- | --- |
| 0.045 | -0.1343 | wait |
| 0.05 | -0.0731 | wait |
| 0.055 | -0.0118 | wait |
| 0.06 | 0.0495 | intervene |
| 0.065 | 0.1107 | intervene |
| 0.07 | 0.172 | intervene |

## 5. Coverage error against the fixed target

A node scored in 60 periods raises 7 flags: realized rate 7/60 = 0.1167; coverage error = 0.1167 - 0.10 = +0.0167. The adapted level eps_t is the controller's variable and is shown separately; it is not the reference.
