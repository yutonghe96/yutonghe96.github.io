# Estimator notes

Maths behind `estimator.html`. Both models step monthly: months are $m = 1, 2, \dots$, and year $y$ covers months $12(y-1)+1, \dots, 12y$.

## Plan rows (both sections)

- A row with start year $a$ and end year $b$ covers months $12(a-1)+1, \dots, 12b$. Where rows overlap, later rows override rates and avgift; savings additions from overlapping rows add up.
- An empty parameter cell leaves that parameter unchanged; values carry forward until another row sets them. In housing, months before a parameter is first set use that first value.
- Amount fields accept `3,000,000`, `2.7e6` and arithmetic such as `12*2000`.

## Housing payments

Inputs: mortgage $L$, down payment $D$; per row: yearly repayment rate $p$, yearly interest rate $r$, monthly avgift $A$, lump sum $S$.

With remaining loan $B_m$ at the start of month $m$ ($B_1 = L$):

$$
\begin{aligned}
R_m &= \min\!\left(\frac{L\,p_m}{12},\; B_m\right) && \text{repayment: a share of the original loan, capped at what's left} \\
I_m &= \frac{B_m\, r_m}{12} && \text{interest: nominal annual rate, compounded monthly} \\
T_m &= R_m + I_m + A_m && \text{monthly total} \\
B_{m+1} &= \max\!\left(B_m - R_m + S_m,\; 0\right) && \text{new balance}
\end{aligned}
$$

**Lump sum:** $S_m$ is nonzero only in month $m = 12a$, the end of the row's start year $a$. It is applied after that month's payment, so the next month's interest uses the new balance. $S < 0$ pays the loan down and $S > 0$ borrows more. The amount actually applied is $\tilde S_m = B_m - R_m - B_{m+1}$, which is capped by the remaining balance.

**Plotted (right axis):** every value is also shown as a percentage of $L + D$.

$$
\begin{aligned}
\text{paid}_m &= D + (L - B_{m+1}), \qquad \text{paid}_m + \text{remaining}_m = L + D \\
\text{remaining}_m &= B_{m+1} \\
\text{cumulative}_m &= D + \sum_{k \le m} \left(T_k + \tilde S_k\right) \\
\text{cumulative excl. avgift}_m &= \text{cumulative}_m - \sum_{k \le m} A_k
\end{aligned}
$$

**Horizon:** $N = 12 \max(b)$, the latest end year. If a row with no end year sets a rate or avgift, the model instead runs until $B = 0$, capped at $N = 1200$ (100 years).

## Cumulative savings

Inputs: base $P_0$; per row: yearly addition $c$, yearly change in addition $\delta$, yearly growth rate $\rho$.

- **Range row** (start $a$, end $b$): month $k = 0, 1, \dots$ since the row began adds $\displaystyle \frac{c}{12}\,(1+\delta)^{k/12}$. The row also sets $\rho$ for those months.
- **One-time row** (no end year): adds $c$ once in month $12a$, and sets $\rho$ from that month on.
- **Growth rate:** once set, $\rho$ carries forward until another row changes it. Before any row sets it, $\rho = 0$.

With $c_m$ the sum of additions in month $m$:

$$
P_m = P_{m-1}\,(1+\rho_m)^{1/12} + c_m
$$

This uses an effective annual rate: twelve months at $\rho$ multiply the balance by exactly $1+\rho$.

$$
\text{contributed}_m = P_0 + \sum_{k \le m} c_k, \qquad
\text{interest}_m = P_m - \text{contributed}_m, \qquad
\text{total}_m = P_m
$$

## Optimisation: prepay the mortgage or invest?

*Not implemented in `estimator.html`; this section outlines the model.*

Each month a surplus can either pay down the loan, which saves future interest, or be invested, which earns compound growth. Avgift is the same under every strategy, so it drops out of the comparison.

**Decision variables** (months $m = 0, \dots, N-1$): extra prepayment $u_m \ge 0$ and investment $v_m \ge 0$.

**Parameters:**

- $X_m$: disposable cash after living costs
- $r$: mortgage rate
- $g$: expected investment return
- $\tau$: interest deduction (ränteavdrag, $\tau = 30\%$ in Sweden). After-tax mortgage rate: $\tilde r = r(1-\tau)$.
- $\kappa$: yearly tax on investments as a share of capital (e.g. ISK schablonskatt). After-tax return: $\tilde g \approx g - \kappa$.

**Dynamics and budget:**

$$
\begin{aligned}
B_{m+1} &= B_m - R_m - u_m, \qquad R_m = \min\!\left(\frac{L\,p_m}{12},\; B_m\right), \qquad B_m \ge 0 \\
P_{m+1} &= P_m\,(1+\tilde g)^{1/12} + v_m \\
X_m &= R_m + \frac{B_m\,\tilde r}{12} + u_m + v_m
\end{aligned}
$$

Once the loan is gone, $R_m = I_m = 0$ and all of $X_m$ is investable.

**Amortisation rule (Sweden), optional:** the required rate depends on loan-to-value $B/V$, where $V$ is the property value:

$$
p(B) =
\begin{cases}
2\% & B/V > 70\% \\
1\% & 50\% < B/V \le 70\% \\
0 & B/V \le 50\%
\end{cases}
\;+\; 1\% \cdot \mathbf{1}\!\left[\,B > 4.5 \times \text{gross income}\,\right]
$$

This couples the decisions: prepaying lowers $B/V$, which lowers $p$, which frees cash flow.

**Objective:** maximise terminal net worth

$$
\max_{\{u_m,\,v_m\}} \; W = P_N - B_N \quad (+\,V_N,\ \text{which no decision changes})
$$

Under uncertainty, maximise $\mathbb{E}[U(W)]$ or a chosen percentile of $W$ instead.

**Finding the best strategy, from simplest to most general:**

1. **Constant rates, fixed $p$.** $W$ is linear in $(u, v)$, so the optimum is a corner solution. One krona prepaid in month $m$ earns $\tilde r$ (interest saved) for as long as the loan is outstanding; the cash flow it frees after payoff then earns $\tilde g$. One krona invested earns $\tilde g$ throughout. Put every surplus krona into whichever is larger:
   $$
   \text{prepay if } \; r(1-\tau) > g - \kappa, \qquad \text{otherwise invest.}
   $$
   When the two are equal, the choice doesn't matter.
2. **Rates that change over time (plan rows).** This is still a linear program in $(u_m, v_m)$, solvable with e.g. `scipy.optimize.linprog`. Equivalently, compare each month the growth factors from $m$ to $N$ under the time-varying rates,
   $$
   \prod_{k=m}^{N-1}\left(1+\frac{\tilde r_k}{12}\right) \quad \text{vs.} \quad \prod_{k=m}^{N-1}(1+\tilde g_k)^{1/12},
   $$
   and choose the larger. The result is a bang-bang policy that can switch when rates change.
3. **LTV-dependent amortisation.** The piecewise $p(B)$ breaks linearity. Use dynamic programming on the state $B$ (discretised), or search over threshold policies: "prepay until $B/V \le \theta$, then invest", with $\theta \in \{70\%, 50\%, 0\}$. Simulate each with the recursion above and keep the policy with the largest $W$.
4. **Uncertain returns.** $g$ is risky, while saved interest is a guaranteed return. Simulate return paths (Monte Carlo, e.g. lognormal monthly returns) and evaluate each policy from step 3 by the distribution of $W$. The risk-neutral break-even is $\tilde r = \mathbb{E}[\tilde g]$. A risk-averse investor instead compares $\tilde r$ with a certainty-equivalent return
   $$
   \tilde g_{\text{CE}} \approx \tilde g - \tfrac{1}{2}\gamma\sigma^2,
   $$
   where $\sigma$ is volatility and $\gamma$ is risk aversion. Also factor in liquidity: prepaid equity can't easily be withdrawn.
