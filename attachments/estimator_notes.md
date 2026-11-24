# Estimator notes

Maths behind `estimator.html`. Both models step monthly: months are $m = 1, 2, \dots$, and year $y$ covers months $12(y-1)+1, \dots, 12y$.

## Plan rows (both sections)

- A row with start year $a$ and end year $b$ covers months $12(a-1)+1, \dots, 12b$. Where rows overlap, later rows override rates and avgift; savings additions from overlapping rows add up.
- An empty parameter cell leaves that parameter unchanged; values carry forward until another row sets them. In housing, months before a parameter is first set use that first value.
- Amount fields accept `3,000,000`, `2.7e6` and arithmetic such as `12*2000`.

## Housing payments

Inputs: mortgage $L_0$, down payment $D$; per row: yearly repayment rate $p$, yearly interest rate $r$, monthly avgift $A$, lump sum $\ell$.

With remaining loan $L_m$ at the start of month $m$ ($L_1 = L_0$):

$$
\begin{aligned}
R_m &= \min\!\left(\frac{L_0\,p_m}{12},\; L_m\right) && \text{repayment: a share of the original loan, capped at what's left} \\
I_m &= \frac{L_m\, r_m}{12} && \text{interest: nominal annual rate, compounded monthly} \\
T_m &= R_m + I_m + A_m && \text{monthly total} \\
L_{m+1} &= \max\!\left(L_m - R_m + \ell_m,\; 0\right) && \text{new balance}
\end{aligned}
$$

**Lump sum:** $\ell_m$ is nonzero only in month $m = 12a$, the end of the row's start year $a$. It is applied after that month's payment, so the next month's interest uses the new balance. $\ell < 0$ pays the loan down and $\ell > 0$ borrows more. The amount actually applied is $\tilde\ell_m = L_m - R_m - L_{m+1}$, which is capped by the remaining balance.

**Plotted (right axis):** every value is also shown as a percentage of $L_0 + D$.

$$
\begin{aligned}
\text{paid}_m &= D + (L_0 - L_{m+1}), \qquad \text{paid}_m + \text{remaining}_m = L_0 + D \\
\text{remaining}_m &= L_{m+1} \\
\text{cumulative}_m &= D + \sum_{k \le m} \left(T_k + \tilde\ell_k\right) \\
\text{cumulative excl. avgift}_m &= \text{cumulative}_m - \sum_{k \le m} A_k
\end{aligned}
$$

**Horizon:** $N = 12 \max(b)$, the latest end year. If a row with no end year sets a rate or avgift, the model instead runs until $L_m = 0$, capped at $N = 1200$ (100 years).

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
\text{return}_m = P_m - \text{contributed}_m, \qquad
\text{total}_m = P_m
$$

## Optimisation: prepay the mortgage or invest?

This section outlines the general model. The **Auto** tab implements step 4 below; see the last subsection for exactly how.

Each month a surplus can either pay down the loan, which saves future interest, or be invested, which earns compound growth. Avgift is the same under every strategy, so it drops out of the comparison.

**Decision variables** (months $m = 0, \dots, N-1$): extra prepayment $0 \le u_m \le s_m$; the rest of the surplus, $s_m - u_m$, is invested.

**Parameters:**

- $X_m$: disposable cash after living costs
- $r$: mortgage rate
- $g$: expected investment return
- $\tau$: interest deduction (ränteavdrag, $\tau = 30\%$ in Sweden on the first 100k SEK of interest per year; $\tau_2 = 21\%$ above that). After-tax mortgage rate: $\tilde r = r(1-\tau)$, or $r(1-\tau_2)$ at the margin once yearly interest exceeds 100k SEK.
- $\kappa$: yearly tax on investments as a share of capital (e.g. ISK schablonskatt). After-tax return: $\tilde g \approx g - \kappa$.

**Dynamics and budget:**

$$
\begin{aligned}
L_{m+1} &= L_m - R_m - u_m, \qquad R_m = \min\!\left(\frac{L_0\,p_m}{12},\; L_m\right), \qquad L_m \ge 0 \\
P_{m+1} &= P_m\,(1+\tilde g)^{1/12} + (s_m - u_m) \\
s_m &= X_m - R_m - \frac{L_m\,\tilde r}{12} \qquad \text{(surplus after required costs)}
\end{aligned}
$$

Once the loan is gone, $R_m = 0$, there is no interest, and all of $X_m$ is investable.

**Amortisation rule (Sweden), optional:** the required rate depends on loan-to-value $L_m/V$, where $V$ is the property value:

$$
p(L_m) =
\begin{cases}
2\% & L_m/V > 70\% \\
1\% & 50\% < L_m/V \le 70\% \\
0 & L_m/V \le 50\%
\end{cases}
$$

(The extra 1% for loans above 4.5× gross income is not modelled.)

This couples the decisions: prepaying lowers $L_m/V$, which lowers $p$, which frees cash flow.

**Objective:** maximise terminal net worth

$$
\max_{\{u_m\}} \; W = P_N - L_N \quad (+\,V_N,\ \text{which no decision changes})
$$

Under uncertainty, maximise $\mathbb{E}[U(W)]$ or a chosen percentile of $W$ instead.

**Finding the best strategy, from simplest to most general:**

1. **Constant rates, fixed $p$.** $W$ is linear in $(u, v)$, so the optimum is a corner solution. One krona prepaid in month $m$ earns $\tilde r$ (interest saved) for as long as the loan is outstanding; the cash flow it frees after payoff then earns $\tilde g$. One krona invested earns $\tilde g$ throughout. Put every surplus krona into whichever is larger:
   $$
   \text{prepay if } \; r(1-\tau) > g - \kappa, \qquad \text{otherwise invest.}
   $$
   When the two are equal, the choice doesn't matter.
2. **Rates that change over time (plan rows).** This is still a linear program in $u_m$, solvable with e.g. `scipy.optimize.linprog`. Equivalently, compare each month the growth factors from $m$ to $N$ under the time-varying rates,
   $$
   \prod_{k=m}^{N-1}\left(1+\frac{\tilde r_k}{12}\right) \quad \text{vs.} \quad \prod_{k=m}^{N-1}(1+\tilde g_k)^{1/12},
   $$
   and choose the larger. The result is a bang-bang policy that can switch when rates change.
3. **LTV-dependent amortisation.** The piecewise $p(L_m)$ breaks linearity. Use dynamic programming on the state $L_m$ (discretised), or search over threshold policies: "prepay until $L_m/V \le \theta$, then invest", with $\theta \in \{70\%, 50\%, 0\}$. Simulate each with the recursion above and keep the policy with the largest $W$.
4. **Uncertain returns.** $g$ is risky, while saved interest is a guaranteed return. Simulate return paths (Monte Carlo, e.g. lognormal monthly returns) and evaluate each policy from step 3 by the distribution of $W$. The risk-neutral break-even is $\tilde r = \mathbb{E}[\tilde g]$. A risk-averse investor instead compares $\tilde r$ with a certainty-equivalent return
   $$
   \tilde g_{\text{CE}} \approx \tilde g - \tfrac{1}{2}\gamma\sigma^2,
   $$
   where $\sigma$ is volatility and $\gamma$ is risk aversion. Also factor in liquidity: prepaid equity can't easily be withdrawn.

### Auto tab implementation

The Auto tab is a Monte Carlo search over a grid of simple rule-based strategies (step 3 policies under step 4 uncertainty). Every strategy is simulated on the same $n$ random market paths, summarised by outcome statistics, and shown side by side for you to choose from.

**Inputs** (Auto tab label → symbol):

| Input | Symbol | Input | Symbol |
|---|---|---|---|
| Mortgage | $L_0$ | Expected return / yr | $g$ |
| Down payment | $D$, with $V = L_0 + D$ | Volatility / yr | $\sigma$ |
| Interest rate / yr | $r$ | Interest deduction | $\tau$ |
| | | Deduction above 100k | $\tau_2$ |
| Avgift / month | $A$ | Investment tax / yr | $\kappa$ |
| Monthly budget | $X_0$ | | |
| Budget change / yr | $\Delta X$ | Horizon (years) | $N/12$ |
| Starting savings | $P_0$ | Confidence | $c$ |
| Savings buffer | buffer | Simulated paths | $n$ |

The property value $V$ is held constant, so LTV $= L_m/V$ only falls as the loan is repaid.

**Market paths.** Path $i = 1, \dots, n$ gets independent standard normals $z_{i,m}$ from a fixed seed. Every strategy reuses the same draws (common random numbers), so differences between strategies are not sampling noise. Savings grow each month by

$$
G_{i,m} = \exp\!\left(\frac{\ln(1+g)}{12} - \frac{\sigma^2}{24} + \frac{\sigma}{\sqrt{12}}\, z_{i,m}\right)(1-\kappa)^{1/12}.
$$

Log returns are normal (geometric Brownian motion, monthly steps). The $-\sigma^2/24$ term is the lognormal mean correction, so that over a year

$$
\mathbb{E}\!\left[\prod_{m=1}^{12} G_{i,m}\right] = (1+g)(1-\kappa).
$$

**What $g$ and $\sigma$ mean.** $g$ is the *mean* yearly return before tax. $\sigma$ is the standard deviation of the yearly *log* return, $\operatorname{sd}\,\ln(P_{12}/P_0)$ without contributions. For $g = 9\%$, $\sigma = 0.15$ (the defaults):

$$
\text{median yearly return} = (1+g)\,e^{-\sigma^2/2} - 1 \approx 7.8\%, \qquad
\operatorname{sd}(\text{yearly return}) = (1+g)\sqrt{e^{\sigma^2} - 1} \approx 16\%.
$$

Over many years the median path compounds at roughly $g - \sigma^2/2$, so a higher $\sigma$ lowers the pessimistic quantiles and favours prepaying (a certain return of $r(1-\tau)$). Only savings are random: $r$, $X$ and $A$ are deterministic.

**Monthly step** for strategy $(\alpha, \theta, \lambda)$ on path $i$, months $m = 0, \dots, N-1$, starting from the original loan $L_0$ and savings $P_0$:

1. *Start of each year* ($m \equiv 0 \bmod 12$): set the amortisation rate $p = p(L_m)$ from the LTV rule above, and the budget $X_m = X_0\,(1+\Delta X)^{m/12}$. Both stay fixed for the next 12 months.
2. *Surplus* after required repayment, full interest, and avgift (the deduction comes back at year end, step 5):
   $$
   R_m = \min\!\left(\frac{L_0\,p}{12},\; L_m\right), \qquad
   s_m = X_m - R_m - \frac{L_m\, r}{12} - A.
   $$
3. *Split the surplus.* Prepay a share $\alpha$ while the LTV after the required repayment is above $\theta$, and invest the rest:
   $$
   L_m' = L_m - R_m, \qquad
   u_m = \begin{cases} \min(\alpha\, s_m,\; L_m') & s_m > 0 \text{ and } L_m'/V > \theta \\ 0 & \text{otherwise} \end{cases}.
   $$
   A negative surplus gives $u_m = 0$, so $s_m < 0$ is withdrawn from savings.
4. *Update balances:*
   $$
   L_{m+1} = L_m' - u_m, \qquad
   P_{m+1} = \begin{cases} P_m\, G_{i,m} + (s_m - u_m) & P_m \ge 0 \\ P_m\,(1 + r/12) + (s_m - u_m) & P_m < 0 \end{cases}
   $$
   Negative savings act as a debt at the mortgage rate (no deduction). Any month with $P_{m+1} < \min(\text{buffer}, P_0)$ marks the path as a *shortfall*: savings fell below the buffer (or below the starting savings, if those are already under the buffer).
5. *Year end* ($m + 1 \equiv 0 \bmod 12$, or the last month): refund the interest deduction on that year's interest $I_y = \sum_{m \in y} L_m r/12$ into savings,
   $$
   D_y = \tau \min(I_y,\ 100\text{k}) + \tau_2 \max(I_y - 100\text{k},\ 0), \qquad P_{m+1} \leftarrow P_{m+1} + D_y.
   $$
   Then, if $\lambda$ is set and $L_{m+1}/V > \lambda$: move savings above the buffer into the loan, down to LTV $\lambda$ at most:
   $$
   \ell = \max\!\big(\min(L_{m+1} - \lambda V,\; P_{m+1} - \text{buffer}),\; 0\big), \qquad L_{m+1} \leftarrow L_{m+1} - \ell, \quad P_{m+1} \leftarrow P_{m+1} - \ell.
   $$
6. *Record the debt-free time* (first month $m+1$ with no loan left; $\infty$ if never within the horizon):
   $$
   t_F = \min\{m : L_m = 0\}.
   $$

After the loan is paid off, $R_m = u_m = 0$ and the whole surplus $X_m - A$ is invested. Everything is net of the interest deduction and investment tax, so net worth is an after-tax amount.

**Strategy grid:** $\alpha \in \{0, 0.05, \dots, 1\}$, $\theta \in \{70\%, 50\%, 0\}$ (the amortisation-rule steps), $\lambda \in \{\text{none}, 70\%, 50\%, 0\}$. When $\alpha = 0$, $\theta$ has no effect and only one value is kept, giving $4 \times (1 + 20 \times 3) = 244$ strategies. The grid is fixed: the number of paths $n$ only sets how precisely each strategy's outcomes are estimated. For example, $(\alpha, \theta, \lambda) = (1, 0, \text{none})$ is "prepay everything until debt-free", and $(0, \cdot, \text{none})$ is "invest everything".

**Outcome statistics** per strategy over the $n$ paths. With $Q_q$ the empirical $q$-quantile (the $\lceil qn \rceil$-th smallest value) and $W_i = P_{i,N} - L_{i,N}$ the final net worth on path $i$ (excluding the property value, which no strategy changes):

$$
W_{\text{low}} = Q_{1-c}(W), \qquad W_{\text{med}} = Q_{0.5}(W), \qquad
\hat t_F = Q_c(t_F), \qquad
\text{risk} = \frac{1}{n}\sum_i \mathbf{1}\big[\min_m P_{i,m} < \min(\text{buffer}, P_0)\big].
$$

So $W_{\text{low}}$ is the net worth beaten in a share $c$ of paths, and $\hat t_F$ is the time by which the loan is paid off in a share $c$ of paths.

The confidence $c$ and the max shortfall risk $\rho_{\max}$ are not used in the simulation itself. Each strategy keeps its sorted per-path outcomes ($W_i$ and $t_F$), so changing either only re-takes these quantiles, re-checks the risk limit and redraws, with no re-run.

**Filtering and default selection:**

1. Strategies with risk above the max shortfall risk $\rho_{\max}$ (an input, default 1%) are hidden. Nothing else is filtered: a strategy not debt-free within the horizon just has no point on the debt-free axis.
2. The shown strategy with the largest $W_{\text{low}}$ is pre-selected; the choice among the rest is yours, from the corner plot.

**Charts.**

- *Corner plot:* the shown strategies (those within the risk limit; a line above the plot counts how many were simulated, are too risky and are shown). The pre-selected one is ringed; click any point to select another. Seven variables are shown: $\alpha$, $\theta$, $\lambda$ (none drawn at 100%), shortfall risk, $\hat t_F$, $W_{\text{med}}$ and $W_{\text{low}}$. The diagonal holds each variable's distribution across the shown strategies: a histogram of the share at each level for $\theta$ and $\lambda$ (titled max = the most common level), and a Gaussian kernel density estimate for the rest (Silverman's rule-of-thumb bandwidth, computed on the un-jittered values; titled max = the density's peak, clamped to the data range, and $\sigma$ = the standard deviation). Each cell below it plots one pair. The settings take only a few values, so their points are jittered slightly.
- *Band chart:* for the selected strategy, per month, the $1-c$, $0.5$ and $c$ quantiles across paths of savings $P_m$, loan $L_m$ and net worth $P_m - L_m$.

**Apply to manual** runs the selected strategy once on the expected path ($G = ((1+g)(1-\kappa))^{1/12}$, no randomness) and writes one row per year:

- housing repayment rate $= \sum_\text{year}(R + u)\,/\,L_0$
- savings yearly addition $= \sum_\text{year} (s - u)$, with growth rate $(1+g)(1-\kappa)-1$
- each lump sum $\ell$ as $-\ell$ in both tables

Consecutive years with identical values are merged.

**Not modelled:** the extra 1% amortisation for loans above 4.5× gross income; changing interest rates, house prices or avgift; fat-tailed or autocorrelated returns; and the illiquidity of prepaid equity beyond the buffer rule.
