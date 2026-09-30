"""
microstructure_vpin.py
VPIN (Volume-Synchronized Probability of Informed Trading) — Easley, López de Prado, O'Hara (2012).
Volume bucketing, bulk classification of buyer/seller-initiated trades,
rolling VPIN toxicity signal. Pure Python stdlib only.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    price: float
    volume: float
    side: str = ''  # 'buy' | 'sell' | '' (unknown — will be classified)

# ---------------------------------------------------------------------------
# 1. Tick rule + Lee-Ready trade classification
# ---------------------------------------------------------------------------
def tick_rule(prices: list[float]) -> list[str]:
    """Classify each trade as 'buy' or 'sell' using tick rule."""
    sides = []
    last_side = 'buy'
    prev = prices[0]
    for i, p in enumerate(prices):
        if i == 0:
            sides.append('buy')
            continue
        if p > prev:
            last_side = 'buy'
        elif p < prev:
            last_side = 'sell'
        # uptick/downtick/zero-tick: hold last_side
        sides.append(last_side)
        prev = p
    return sides

def lee_ready(prices: list[float], midpoints: list[float]) -> list[str]:
    """
    Lee-Ready (1991) classification:
    - If price > midpoint → buy
    - If price < midpoint → sell
    - If price == midpoint → use tick rule
    """
    tick = tick_rule(prices)
    sides = []
    for i, (p, m) in enumerate(zip(prices, midpoints)):
        if p > m:
            sides.append('buy')
        elif p < m:
            sides.append('sell')
        else:
            sides.append(tick[i])
    return sides

# ---------------------------------------------------------------------------
# 2. Volume bucket filling
# ---------------------------------------------------------------------------
def fill_volume_buckets(trades: list[Trade], bucket_size: float) -> list[dict]:
    """
    Fill fixed-size volume buckets. Each bucket records
    {buy_vol, sell_vol, total_vol, n_trades, vwap}.
    """
    buckets = []
    current = {'buy_vol': 0.0, 'sell_vol': 0.0, 'total_vol': 0.0,
                'n_trades': 0, 'price_vol': 0.0}
    remaining = bucket_size

    for t in trades:
        vol_left = t.volume
        while vol_left > 0:
            fill = min(vol_left, remaining)
            if t.side == 'buy':
                current['buy_vol'] += fill
            else:
                current['sell_vol'] += fill
            current['total_vol'] += fill
            current['price_vol'] += t.price * fill
            current['n_trades'] += 1
            vol_left -= fill
            remaining -= fill
            if remaining <= 1e-10:
                current['vwap'] = current['price_vol'] / max(current['total_vol'], 1e-10)
                buckets.append(dict(current))
                current = {'buy_vol': 0.0, 'sell_vol': 0.0, 'total_vol': 0.0,
                           'n_trades': 0, 'price_vol': 0.0}
                remaining = bucket_size

    # Partial bucket at end
    if current['total_vol'] > 0:
        current['vwap'] = current['price_vol'] / max(current['total_vol'], 1e-10)
        buckets.append(current)
    return buckets

# ---------------------------------------------------------------------------
# 3. VPIN computation
# ---------------------------------------------------------------------------
class VPIN:
    """
    VPIN = (1/n) * sum_{i=1}^{n} |V_i^B - V_i^S| / V_bucket
    where n is the rolling window of buckets.
    """
    def __init__(self, n_buckets: int = 50, bucket_size: float = 10_000.0):
        self.n = n_buckets
        self.bucket_size = bucket_size
        self._history: list[float] = []  # |buy - sell| / bucket per bucket

    def compute(self, trades: list[Trade]) -> float:
        """Compute VPIN over all trades. Returns current VPIN value."""
        buckets = fill_volume_buckets(trades, self.bucket_size)
        toxicity_vals = []
        for b in buckets:
            imb = abs(b['buy_vol'] - b['sell_vol']) / max(b['total_vol'], 1e-10)
            toxicity_vals.append(imb)
            self._history.append(imb)

        window = self._history[-self.n:]
        return sum(window) / max(len(window), 1)

    def rolling_vpin(self, trades: list[Trade]) -> list[float]:
        """Return VPIN for each bucket after the first n buckets."""
        buckets = fill_volume_buckets(trades, self.bucket_size)
        imbalances = [abs(b['buy_vol'] - b['sell_vol']) / max(b['total_vol'], 1e-10)
                      for b in buckets]
        vpins = []
        for i in range(len(imbalances)):
            window = imbalances[max(0, i - self.n + 1): i + 1]
            vpins.append(sum(window) / len(window))
        return vpins

# ---------------------------------------------------------------------------
# 4. PIN (Probability of Informed Trading) — simplified Easley-O'Hara model
# ---------------------------------------------------------------------------
def pin_score(buy_counts: list[int], sell_counts: list[int],
              n_iter: int = 100) -> dict:
    """
    Estimate PIN via EM algorithm (Easley, Hvidkjaer, O'Hara 2002).
    Model: with prob alpha, an information event occurs.
    On event days: informed buys with rate mu+eps, sells with rate eps
    (buy-news) or buys with eps, sells with mu+eps (sell-news).
    On no-event days: buys ~ Pois(eps_b), sells ~ Pois(eps_s).
    PIN = alpha*mu / (alpha*mu + 2*eps)
    Simplified 3-param version here.
    """
    # MOM initializer
    mean_buy = sum(buy_counts) / max(len(buy_counts), 1)
    mean_sell = sum(sell_counts) / max(len(sell_counts), 1)
    eps = (mean_buy + mean_sell) / 2
    mu = max(abs(mean_buy - mean_sell), 1.0)
    alpha = 0.5
    delta = 0.5  # prob of bad news | event

    def log_likelihood(alpha, delta, mu, eps):
        ll = 0.0
        for b, s in zip(buy_counts, sell_counts):
            # No event: Pois(eps)*Pois(eps)
            p_no = (1 - alpha) * _poisson_ll(b, eps) * _poisson_ll(s, eps)
            # Good news: Pois(mu+eps) * Pois(eps)
            p_good = alpha * (1 - delta) * _poisson_ll(b, mu + eps) * _poisson_ll(s, eps)
            # Bad news: Pois(eps) * Pois(mu+eps)
            p_bad = alpha * delta * _poisson_ll(b, eps) * _poisson_ll(s, mu + eps)
            total = p_no + p_good + p_bad
            ll += math.log(max(total, 1e-300))
        return ll

    def _poisson_ll(k, lam):
        if lam <= 0:
            return 0.0 if k > 0 else 1.0
        return math.exp(-lam + k * math.log(max(lam, 1e-300)) -
                        sum(math.log(i+1) for i in range(k)))

    # Simple grid / gradient-free search (EM is complex; use grid here)
    best_ll = float('-inf')
    best_params = (alpha, delta, mu, eps)
    for a in [0.2, 0.4, 0.6, 0.8]:
        for d in [0.3, 0.5, 0.7]:
            for m_mult in [0.5, 1.0, 2.0]:
                m = mu * m_mult
                e = eps
                ll = log_likelihood(a, d, m, e)
                if ll > best_ll:
                    best_ll = ll
                    best_params = (a, d, m, e)

    a, d, m, e = best_params
    pin = a * m / max(a * m + 2 * e, 1e-10)
    return {'alpha': a, 'delta': d, 'mu': m, 'epsilon': e,
            'PIN': pin, 'log_likelihood': best_ll}

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import random
    rng = random.Random(42)

    print("=" * 60)
    print("Market Microstructure: VPIN & PIN")
    print("=" * 60)

    # Simulate 2000 trades: 1000 normal + 1000 informed
    print("\n1. Trade Classification (Tick Rule & Lee-Ready)")
    prices = [100.0]
    for _ in range(199):
        prices.append(prices[-1] + rng.gauss(0, 0.05))
    midpoints = [p + rng.uniform(-0.02, 0.02) for p in prices]
    tick_sides = tick_rule(prices)
    lr_sides = lee_ready(prices, midpoints)
    buy_pct_tick = sum(1 for s in tick_sides if s == 'buy') / len(tick_sides)
    buy_pct_lr = sum(1 for s in lr_sides if s == 'buy') / len(lr_sides)
    print(f"   Tick rule:  {buy_pct_tick:.1%} buy-initiated")
    print(f"   Lee-Ready:  {buy_pct_lr:.1%} buy-initiated")

    print("\n2. VPIN — Normal Market Conditions")
    normal_trades = []
    for i in range(500):
        side = 'buy' if rng.random() > 0.5 else 'sell'
        normal_trades.append(Trade(price=100 + rng.gauss(0, 0.1),
                                    volume=rng.uniform(500, 1500), side=side))
    vpin_calc = VPIN(n_buckets=10, bucket_size=5000.0)
    vpin_normal = vpin_calc.compute(normal_trades)
    print(f"   VPIN (normal):  {vpin_normal:.4f}  (expected ~0.50 for 50/50 flow)")

    print("\n3. VPIN — Informed Trading Spike (80% buy pressure)")
    informed_trades = []
    for i in range(500):
        side = 'buy' if rng.random() > 0.20 else 'sell'
        informed_trades.append(Trade(price=100 + rng.gauss(0, 0.1),
                                      volume=rng.uniform(500, 1500), side=side))
    vpin_calc2 = VPIN(n_buckets=10, bucket_size=5000.0)
    vpin_informed = vpin_calc2.compute(informed_trades)
    print(f"   VPIN (informed): {vpin_informed:.4f}  (higher = more toxic flow)")

    print("\n4. Rolling VPIN")
    all_trades = normal_trades[:200] + informed_trades[:100] + normal_trades[200:]
    vpin_roll = VPIN(n_buckets=5, bucket_size=5000.0)
    rolling = vpin_roll.rolling_vpin(all_trades)
    print(f"   Buckets computed: {len(rolling)}")
    step = max(1, len(rolling) // 6)
    for i, v in enumerate(rolling[::step][:6]):
        print(f"   Bucket {i*step:>3}: VPIN = {v:.4f}")

    print("\n5. PIN Estimation (Easley-Hvidkjaer-O'Hara)")
    # Simulate 20 days of trade counts
    buys_normal  = [int(rng.gauss(100, 15)) for _ in range(10)]
    sells_normal = [int(rng.gauss(100, 15)) for _ in range(10)]
    buys_informed  = [int(rng.gauss(150, 20)) for _ in range(10)]
    sells_informed = [int(rng.gauss(70,  15)) for _ in range(10)]
    all_buys  = buys_normal  + buys_informed
    all_sells = sells_normal + sells_informed
    pin_result = pin_score(all_buys, all_sells)
    print(f"   Alpha (info event prob): {pin_result['alpha']:.3f}")
    print(f"   Delta (bad news prob):   {pin_result['delta']:.3f}")
    print(f"   Mu (informed rate):      {pin_result['mu']:.2f}")
    print(f"   Epsilon (noise rate):    {pin_result['epsilon']:.2f}")
    print(f"   PIN:                     {pin_result['PIN']:.4f}")
    print(f"   (>0.2 suggests elevated information asymmetry)")

    print("\n[Done] VPIN & PIN microstructure analysis.")
