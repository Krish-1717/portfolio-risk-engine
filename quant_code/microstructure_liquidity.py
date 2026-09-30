"""
microstructure_liquidity.py
Liquidity metrics: Amihud illiquidity ratio, Roll's implicit spread,
Corwin-Schultz high-low spread, Kyle's lambda (price impact),
Almgren-Chriss optimal execution, TWAP/VWAP slippage decomposition.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

def mean(xs): return sum(xs) / max(len(xs), 1)
def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs) / max(len(xs)-1, 1))

# ---------------------------------------------------------------------------
# 1. Amihud (2002) illiquidity ratio
# ---------------------------------------------------------------------------
def amihud_ratio(returns: list[float], volumes: list[float],
                  window: int = 21) -> list[float]:
    """
    Amihud illiquidity = |r_t| / volume_t (rolling average).
    Higher = less liquid.
    """
    n = len(returns)
    daily = [abs(r) / max(v, 1.0) for r, v in zip(returns, volumes)]
    rolling = []
    for t in range(n):
        w = daily[max(0, t - window + 1): t + 1]
        rolling.append(mean(w) * 1e6)  # scale to per-million-shares
    return rolling

# ---------------------------------------------------------------------------
# 2. Roll (1984) implicit bid-ask spread
# ---------------------------------------------------------------------------
def roll_spread(prices: list[float]) -> float:
    """
    Roll's model: Spread = 2 * sqrt(-Cov(delta_p_t, delta_p_{t-1}))
    where delta_p = price change.
    """
    dp = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    n = len(dp)
    if n < 2:
        return 0.0
    cov = sum(dp[i] * dp[i-1] for i in range(1, n)) / max(n - 1, 1)
    if cov >= 0:
        return 0.0  # No valid spread estimate
    return 2.0 * math.sqrt(-cov)

def roll_spread_rolling(prices: list[float], window: int = 21) -> list[float]:
    """Rolling Roll spread over a sliding window."""
    spreads = []
    for t in range(len(prices)):
        w_prices = prices[max(0, t - window + 1): t + 1]
        spreads.append(roll_spread(w_prices) if len(w_prices) >= 3 else 0.0)
    return spreads

# ---------------------------------------------------------------------------
# 3. Corwin-Schultz (2012) high-low spread estimator
# ---------------------------------------------------------------------------
def corwin_schultz_spread(highs: list[float], lows: list[float]) -> list[float]:
    """
    CS spread uses adjacent day high-low ranges.
    beta = [ln(H_t/L_t)]^2 + [ln(H_{t+1}/L_{t+1})]^2
    gamma = [ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1}))]^2
    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma/(3-2*sqrt(2)))
    spread = 2*(exp(alpha)-1)/(1+exp(alpha))
    """
    n = len(highs)
    spreads = [0.0]
    for t in range(1, n):
        try:
            beta = (math.log(highs[t-1]/lows[t-1]))**2 + (math.log(highs[t]/lows[t]))**2
            gamma = (math.log(max(highs[t-1], highs[t]) / min(lows[t-1], lows[t])))**2
            k = 3 - 2*math.sqrt(2)
            alpha = (math.sqrt(2*beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
            if alpha > 0:
                spread = 2*(math.exp(alpha) - 1) / (1 + math.exp(alpha))
            else:
                spread = 0.0
        except (ValueError, ZeroDivisionError):
            spread = 0.0
        spreads.append(spread)
    return spreads

# ---------------------------------------------------------------------------
# 4. Kyle's lambda (price impact)
# ---------------------------------------------------------------------------
class KyleLambda:
    """
    Linear price impact model: dp_t = lambda * x_t + epsilon_t
    where x_t is signed order flow (+ = net buy, - = net sell).
    lambda estimated via OLS regression of dp on x.
    """
    def __init__(self):
        self.lambda_: float = 0.0
        self.alpha: float = 0.0
        self.r_squared: float = 0.0

    def fit(self, prices: list[float], signed_volumes: list[float]) -> 'KyleLambda':
        dp = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        x = signed_volumes[1:]
        n = len(dp)
        if n < 2:
            return self
        mean_x = mean(x)
        mean_dp = mean(dp)
        cov_xy = sum((x[i] - mean_x) * (dp[i] - mean_dp) for i in range(n)) / (n - 1)
        var_x = sum((xi - mean_x)**2 for xi in x) / (n - 1)
        self.lambda_ = cov_xy / max(var_x, 1e-12)
        self.alpha = mean_dp - self.lambda_ * mean_x
        pred = [self.alpha + self.lambda_ * xi for xi in x]
        ss_res = sum((dp[i] - pred[i])**2 for i in range(n))
        ss_tot = sum((dp[i] - mean_dp)**2 for i in range(n))
        self.r_squared = 1 - ss_res / max(ss_tot, 1e-12)
        return self

    def impact(self, order_size: float) -> float:
        """Predicted price impact of a signed order."""
        return self.lambda_ * order_size

# ---------------------------------------------------------------------------
# 5. Square-root market impact model
# ---------------------------------------------------------------------------
def sqrt_market_impact(order_size: float, adv: float,
                        volatility: float, eta: float = 0.1) -> float:
    """
    Square-root law: MI = eta * sigma * sqrt(Q / ADV)
    order_size: shares to trade
    adv: average daily volume
    volatility: daily vol (annualized / sqrt(252))
    """
    return eta * volatility * math.sqrt(abs(order_size) / max(adv, 1.0))

# ---------------------------------------------------------------------------
# 6. Almgren-Chriss optimal execution
# ---------------------------------------------------------------------------
@dataclass
class AlmgrenChriss:
    """
    Almgren-Chriss (2001) model for optimal execution of X shares over T periods.
    Minimizes E[cost] + lambda * Var[cost].
    Permanent impact: g(v) = gamma * v
    Temporary impact: h(v) = eta * v
    """
    total_shares: float    # X: total shares to sell
    T: int                  # number of time periods
    gamma: float = 1e-7    # permanent impact coefficient
    eta: float = 1e-6      # temporary impact coefficient
    sigma: float = 0.02    # per-period price volatility
    risk_aversion: float = 1e-6  # lambda

    def optimal_schedule(self) -> list[float]:
        """
        Compute optimal trading trajectory x_j (shares remaining at step j).
        x_j = X * sinh(kappa*(T-j)) / sinh(kappa*T)
        where kappa = sqrt(lambda*sigma^2 / eta_tilde)
        """
        eta_tilde = self.eta - 0.5 * self.gamma
        if eta_tilde <= 0:
            # Degenerate: trade everything at start
            return [self.total_shares] + [0.0] * self.T

        kappa_sq = self.risk_aversion * self.sigma**2 / max(eta_tilde, 1e-15)
        kappa = math.sqrt(max(kappa_sq, 0))

        if kappa * self.T < 1e-8:
            # Linear schedule (no risk aversion)
            return [self.total_shares * (1 - j/self.T) for j in range(self.T + 1)]

        denom = math.sinh(kappa * self.T)
        x = [self.total_shares * math.sinh(kappa * (self.T - j)) / max(denom, 1e-10)
             for j in range(self.T + 1)]
        return x

    def trading_rates(self) -> list[float]:
        """Shares traded per period n_j = x_{j-1} - x_j."""
        x = self.optimal_schedule()
        return [x[j] - x[j+1] for j in range(self.T)]

    def expected_cost(self) -> dict:
        """Compute expected shortfall and variance."""
        rates = self.trading_rates()
        x = self.optimal_schedule()
        perm_cost = 0.5 * self.gamma * self.total_shares**2
        temp_cost = self.eta * sum(n**2 for n in rates)
        variance = self.sigma**2 * sum(x[j]**2 for j in range(1, self.T + 1))
        return {
            'permanent_cost': perm_cost,
            'temporary_cost': temp_cost,
            'total_expected_cost': perm_cost + temp_cost,
            'variance': variance,
            'std_shortfall': math.sqrt(max(variance, 0)),
        }

# ---------------------------------------------------------------------------
# 7. VWAP slippage decomposition
# ---------------------------------------------------------------------------
def vwap_slippage(execution_prices: list[float], execution_vols: list[float],
                   market_prices: list[float], market_vols: list[float]) -> dict:
    """
    Execution shortfall = (VWAP_execution - VWAP_market) / VWAP_market.
    Also decomposes into timing (IS), spread cost, and market impact.
    """
    def vwap(prices, vols):
        tv = sum(vols)
        if tv == 0:
            return 0.0
        return sum(p*v for p, v in zip(prices, vols)) / tv

    exec_vwap = vwap(execution_prices, execution_vols)
    mkt_vwap = vwap(market_prices, market_vols)
    slippage = (exec_vwap - mkt_vwap) / max(abs(mkt_vwap), 1e-10)
    arrival_price = market_prices[0] if market_prices else 0.0
    is_cost = (exec_vwap - arrival_price) / max(abs(arrival_price), 1e-10)
    return {
        'exec_vwap': exec_vwap,
        'market_vwap': mkt_vwap,
        'vwap_slippage': slippage,
        'implementation_shortfall': is_cost,
        'market_impact_estimate': slippage - is_cost,
    }

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import random
    rng = random.Random(42)

    print("=" * 60)
    print("Market Microstructure: Liquidity & Impact")
    print("=" * 60)

    T = 100
    prices = [100.0]
    for _ in range(T - 1):
        prices.append(prices[-1] * math.exp(rng.gauss(0, 0.01)))
    volumes = [rng.uniform(50000, 200000) for _ in range(T)]
    returns = [math.log(prices[i]/prices[i-1]) for i in range(1, T)]

    print("\n1. Amihud Illiquidity Ratio (rolling 21-day)")
    amihud = amihud_ratio(returns, volumes[1:], window=21)
    print(f"   Mean illiquidity : {mean(amihud):.6f}")
    print(f"   Max illiquidity  : {max(amihud):.6f}")
    print(f"   (Higher = less liquid)")

    print("\n2. Roll Implicit Bid-Ask Spread")
    roll = roll_spread(prices)
    print(f"   Roll spread: {roll:.5f}  (${roll:.5f} per share)")
    rolling_roll = roll_spread_rolling(prices, window=21)
    print(f"   Rolling mean: {mean(rolling_roll):.5f}")

    print("\n3. Corwin-Schultz High-Low Spread")
    highs = [p * (1 + abs(rng.gauss(0, 0.005))) for p in prices]
    lows  = [p * (1 - abs(rng.gauss(0, 0.005))) for p in prices]
    cs = corwin_schultz_spread(highs, lows)
    valid_cs = [s for s in cs if s > 0]
    print(f"   Mean CS spread: {mean(valid_cs):.5f}  ({mean(valid_cs)*100:.3f}%)")

    print("\n4. Kyle's Lambda (Price Impact)")
    signed_vols = [rng.gauss(0, 10000) for _ in range(T)]
    kyle = KyleLambda()
    kyle.fit(prices, signed_vols)
    print(f"   Lambda: {kyle.lambda_:.4e}  ($/share per share traded)")
    print(f"   R²:     {kyle.r_squared:.4f}")
    impact_1k = kyle.impact(1000)
    print(f"   Impact of 1,000 shares: ${impact_1k:.4f}")
    impact_10k = kyle.impact(10000)
    print(f"   Impact of 10,000 shares: ${impact_10k:.4f}")

    print("\n5. Square-Root Market Impact")
    adv = 1_000_000
    daily_vol = 0.015
    for size in [10_000, 50_000, 100_000, 500_000]:
        impact = sqrt_market_impact(size, adv, daily_vol)
        print(f"   Order {size:>8,} shrs | Impact: {impact*100:.4f}% of price")

    print("\n6. Almgren-Chriss Optimal Execution")
    ac = AlmgrenChriss(
        total_shares=100_000, T=10,
        gamma=1e-7, eta=1e-6,
        sigma=0.015, risk_aversion=1e-6
    )
    schedule = ac.optimal_schedule()
    rates = ac.trading_rates()
    costs = ac.expected_cost()
    print(f"   Selling 100,000 shares over 10 periods:")
    print(f"   {'Period':>8} | {'Shares left':>12} | {'Shares traded':>14}")
    print("   " + "-" * 38)
    for j, (x, n) in enumerate(zip(schedule[:-1], rates)):
        if j % 2 == 0 or j == len(rates)-1:
            print(f"   {j:>8} | {x:>12,.0f} | {n:>14,.0f}")
    print(f"\n   Permanent cost   : ${costs['permanent_cost']:,.2f}")
    print(f"   Temporary cost   : ${costs['temporary_cost']:,.2f}")
    print(f"   Total E[shortfall]: ${costs['total_expected_cost']:,.2f}")
    print(f"   Std[shortfall]   : ${costs['std_shortfall']:,.2f}")

    print("\n7. VWAP Slippage Decomposition")
    exec_p = [100.02, 100.05, 100.08, 100.03, 100.06]
    exec_v = [5000, 8000, 6000, 4000, 7000]
    mkt_p  = [100.00, 100.01, 100.04, 100.02, 100.03]
    mkt_v  = [20000, 30000, 25000, 18000, 22000]
    slip = vwap_slippage(exec_p, exec_v, mkt_p, mkt_v)
    print(f"   Exec VWAP:  {slip['exec_vwap']:.4f}")
    print(f"   Mkt VWAP:   {slip['market_vwap']:.4f}")
    print(f"   VWAP slippage: {slip['vwap_slippage']*100:+.4f}%")
    print(f"   Impl. shortfall: {slip['implementation_shortfall']*100:+.4f}%")
    print(f"   Market impact est: {slip['market_impact_estimate']*100:+.4f}%")

    print("\n[Done] Liquidity & market impact analysis.")
