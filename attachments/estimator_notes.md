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

**Plotted (balance chart, below the payments chart):** every value is also shown as a percentage of $L_0 + D$.

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
| Interest rate / yr | $r$ | Investment tax / yr | $\kappa$ |
| Rate volatility / yr | $\sigma_r$ | | |
| Avgift / month | $A$ | | |
| Monthly budget | $X_0$ | | |
| Budget change / yr | $\Delta X$ | Horizon (years) | $N/12$ |
| Starting savings | $P_0$ | Confidence | $c$ |
| Savings buffer | buffer | Simulated paths | $n$ |

$\tau = 30\%$ and $\tau_2 = 21\%$ are fixed by Swedish tax rules (like the 70/50 % amortisation thresholds), so they are not inputs.

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

Over many years the median path compounds at roughly $g - \sigma^2/2$, so a higher $\sigma$ lowers the pessimistic quantiles and favours prepaying (a certain return of $r(1-\tau)$). Savings returns and the mortgage rate are random; $X$ and $A$ are deterministic.

**Rate paths.** The mortgage rate is re-set once a year (like the amortisation rate), from separate standard normals $z'_{i,y}$ with their own fixed seed, again shared by all strategies. Year 1 uses the input rate $r_0$; later years follow a mean-reverting AR(1), floored at 0:

$$
x_{i,0} = r_0, \qquad
x_{i,y} = r_0 + \phi\,(x_{i,y-1} - r_0) + \sigma_r\, z'_{i,y}, \qquad
r_{i,y} = \max(x_{i,y},\ 0), \qquad \phi = 0.5.
$$

$\sigma_r$ is in rate units: 0.01 ≈ ±1 percentage point a year, so from $r_0 = 3\%$ the next year's rate is typically 2–4%, and in the long run about two years in three lie within $3\% \pm 1.15\% = 1.85$–$4.15\%$ (and 90% of paths within about 1.1–4.9% by year 10). With $\phi = 0.5$ a deviation halves each year, the year-to-year change has sd $\sigma_r\sqrt{2/(1+\phi)} \approx 1.15\,\sigma_r$, and the long-run spread of the rate around $r_0$ is $\sigma_r/\sqrt{1-\phi^2} \approx 1.15\,\sigma_r$.

*Default $\sigma_r = 0.01$.* Average variable rates on new Swedish mortgages (SCB, since 2005) ran from about 3% in 2005 to about 5.5% in late 2008, fell to about 2% in 2009, rose to about 4% in 2011, stayed near 1.4–1.6% from 2015 to 2021 (low 1.38% in November 2021), jumped to about 4.5–5% in 2023 and fell back to 3.45% by December 2024 and under 3% in mid-2025. Over these years both the yearly change and the spread around the mean are roughly 1.2 percentage points, which $\sigma_r = 0.01$ with $\phi = 0.5$ reproduces. The yearly path values are approximate, read from SCB releases, not a fitted series. Rates are assumed independent of market returns (in practice rate rises have often come with weak markets, which would make prepaying look somewhat better).

**Monthly step** for strategy $(\alpha, \theta, \lambda)$ on path $i$, months $m = 0, \dots, N-1$, starting from the original loan $L_0$ and savings $P_0$:

1. *Start of each year* ($m \equiv 0 \bmod 12$): set the amortisation rate $p = p(L_m)$ from the LTV rule above, the budget $X_m = X_0\,(1+\Delta X)^{m/12}$ and the mortgage rate $r_{i,y}$ from the rate path. All three stay fixed for the next 12 months.
2. *Surplus* after required repayment, full interest, and avgift (the deduction comes back at year end, step 5):
   $$
   R_m = \min\!\left(\frac{L_0\,p}{12},\; L_m\right), \qquad
   s_m = X_m - R_m - \frac{L_m\, r_{i,y}}{12} - A,
   $$
   where $y = \lfloor m/12 \rfloor$ is the year of month $m$ and $r_{i,y}$ that year's rate on path $i$ (the rate $r$ in step 4 below is also $r_{i,y}$).
3. *Split the surplus.* The minimum payments in step 2 always come first. If savings after this month's growth are below the buffer, the surplus first refills them up to it; of what is left, $f_m$, prepay a share $\alpha$ while the LTV after the required repayment is above $\theta$, and invest the rest:
   $$
   f_m = s_m - \min\!\big(\max(s_m, 0),\; \max(\text{buffer} - P_m G_{i,m},\, 0)\big), \qquad
   L_m' = L_m - R_m, \qquad
   u_m = \begin{cases} \min(\alpha\, f_m,\; L_m') & f_m > 0 \text{ and } L_m'/V > \theta \\ 0 & \text{otherwise} \end{cases}.
   $$
   ($G_{i,m}$ is replaced by $1 + r/12$ when $P_m < 0$, as in step 4.) A negative surplus gives $u_m = 0$, so $s_m < 0$ is withdrawn from savings.
4. *Update balances:*
   $$
   L_{m+1} = L_m' - u_m, \qquad
   P_{m+1} = \begin{cases} P_m\, G_{i,m} + (s_m - u_m) & P_m \ge 0 \\ P_m\,(1 + r/12) + (s_m - u_m) & P_m < 0 \end{cases}
   $$
   Negative savings act as a debt at the mortgage rate (no deduction). The lowest $P_{m+1}$ over the path is recorded as $P_{\min,i}$.
5. *Year end* ($m + 1 \equiv 0 \bmod 12$, or the last month): refund the interest deduction on that year's interest $I_y = \sum_{m \in y} L_m r_{i,y}/12$ into savings,
   $$
   D_y = \tau \min(I_y,\ 100\text{k}) + \tau_2 \max(I_y - 100\text{k},\ 0), \qquad P_{m+1} \leftarrow P_{m+1} + D_y.
   $$
   Then, if $\lambda$ is set and $L_{m+1}/V > \lambda$: move savings above the buffer into the loan, down to LTV $\lambda$ at most:
   $$
   \ell = \max\!\big(\min(L_{m+1} - \lambda V,\; P_{m+1} - \text{buffer}),\; 0\big), \qquad L_{m+1} \leftarrow L_{m+1} - \ell, \quad P_{m+1} \leftarrow P_{m+1} - \ell.
   $$
6. *Record the debt-free time* (first month $m+1$ with no loan left; $\infty$ if never within the horizon):
   $$
   t = \min\{m : L_m = 0\}.
   $$

After the loan is paid off, $R_m = u_m = 0$ and the whole surplus $X_m - A$ is invested. Everything is net of the interest deduction and investment tax, so net worth is an after-tax amount.

**Strategy grid:** $\alpha \in \{0, 0.05, \dots, 1\}$, $\theta \in \{70\%, 50\%, 0\}$ (the amortisation-rule steps), $\lambda \in \{\text{none}, 70\%, 50\%, 0\}$. When $\alpha = 0$, $\theta$ has no effect and only one value is kept, giving $4 \times (1 + 20 \times 3) = 244$ strategies. The grid is fixed: the number of paths $n$ only sets how precisely each strategy's outcomes are estimated. For example, $(\alpha, \theta, \lambda) = (1, 0, \text{none})$ is "prepay everything until debt-free", and $(0, \cdot, \text{none})$ is "invest everything".

**Outcome statistics** per strategy over the $n$ paths. With $Q_q$ the empirical $q$-quantile (the $\lceil qn \rceil$-th smallest value) and $W_i = P_{i,N} - L_{i,N}$ the final net worth on path $i$ (excluding the property value, which no strategy changes):

$$
W_{\text{low}} = Q_{1-c}(W), \qquad W_{\text{med}} = Q_{0.5}(W), \qquad
t_f = Q_c(t), \qquad
P_{\text{low}} = Q_{1-c}(P_{\min}), \qquad P_{\min,i} = \min_m P_{i,m}.
$$

So $W_{\text{low}}$ is the net worth beaten in a share $c$ of paths, $t_f$ is the time by which the loan is paid off in a share $c$ of paths, and $P_{\text{low}}$ is the lowest savings that a share $c$ of paths never go under.

*Why lowest savings rather than a shortfall probability.* Since the surplus refills savings up to the buffer before anything is prepaid (step 3), savings only stay below the buffer if a month's market loss exceeds that month's surplus or the budget can't cover housing costs. With typical inputs the probability of breaking the buffer is about 0 for every strategy, so it no longer tells strategies apart. What does differ is how much cash a strategy keeps: investing leaves savings well above the buffer, while prepaying and lump sums hold them near it. $P_{\text{low}}$ measures that.

The confidence $c$ is not used in the simulation itself. Each strategy keeps its sorted per-path outcomes ($W_i$, $t$ and $P_{\min,i}$), so changing it only re-takes these quantiles and redraws, with no re-run.

**Filtering and default selection:**

1. All strategies are shown; nothing is filtered. A strategy not debt-free within the horizon just has no point on the debt-free axis.
2. The shown strategy with the largest $W_{\text{low}}$ is pre-selected; the choice among the rest is yours, from the corner plot.

**Charts.**

- *Corner plot:* every strategy. The pre-selected one is ringed; click any point to select another. Seven variables are shown: $\alpha$, $\theta$, $\lambda$ (none drawn at 100%), lowest savings $P_{\text{low}}$, $t_f$, $W_{\text{med}}$ and $W_{\text{low}}$. The diagonal holds each variable's distribution across the shown strategies: a histogram of the share at each level for $\theta$ and $\lambda$ (titled Mo = mode, the most common level), and a Gaussian kernel density estimate for the rest (Silverman's rule-of-thumb bandwidth, computed on the un-jittered values; titled Mo = mode, the density's peak, clamped to the data range, and $\sigma$ = the standard deviation). Each cell below it plots one pair. The settings take only a few values, so their points are jittered slightly.
- *All strategies:* each shown strategy's median (across paths) savings $P$, loan $L$ and net worth $P - L$ at the start and at the end of every month, one faint line per strategy, with the selected strategy drawn bold. Medians use quickselect rather than a full sort, which keeps the extra cost of the grid run under a second. The medians are taken per month and per quantity, so the net-worth line is not the savings median minus the loan median. They are computed during the grid run. Click a line to select that strategy.
- *Summary:* for the selected strategy, $W_{\text{med}}$ and $W_{\text{low}}$; the debt-free time as one value if every path agrees, otherwise its median with the $1-c$ to $c$ band; and the lowest savings as median $Q_{0.5}(P_{\min})$ and $P_{\text{low}}$.
- *Band chart:* for the selected strategy, per month, the $1-c$, $0.5$ and $c$ quantiles across paths of savings $P_m$, loan $L_m$ and net worth $P_m - L_m$.

**Apply to manual** runs the selected strategy once on the expected path ($G = ((1+g)(1-\kappa))^{1/12}$ and the fixed rate $r_0$, no randomness) and writes one row per year:

- housing repayment rate $= \sum_\text{year}(R + u)\,/\,L_0$
- savings yearly addition $= \sum_\text{year} (s - u)$, with growth rate $(1+g)(1-\kappa)-1$
- each lump sum $\ell$ as $-\ell$ in both tables

Consecutive years with identical values are merged.

**Not modelled:** the extra 1% amortisation for loans above 4.5× gross income; a link between interest rates and market returns; changing house prices or avgift; fat-tailed or autocorrelated returns; and the illiquidity of prepaid equity beyond the buffer rule.
