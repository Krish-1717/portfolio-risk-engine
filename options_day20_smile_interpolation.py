"""
Vol Smile Interpolation and Calibration

Implements:
- SVI (Stochastic Volatility Inspired) parameterization (Gatheral 2004)
- SSVI (Surface SVI) with wing model extension
- Cubic spline smile interpolation
- Arbitrage-free constraints (butterfly, calendar spread)
- Implied vol surface stitching across expiries

Reference: Gatheral & Jacquier (2014) "Arbitrage-free SVI vol surfaces"
"""

import math
import random
from typing import List, Tuple, Optional, NamedTuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SVIParams(NamedTuple):
    """SVI raw parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    a: float       # overall level of variance
    b: float       # angle between left and right asymptotes
    rho: float     # orientation of smile (-1 < rho < 1)
    m: float       # translation in log-strike
    sigma: float   # smoothness at vertex; sigma > 0


class SmilePoint(NamedTuple):
    strike: float
    log_moneyness: float   # k = log(K/F)
    implied_vol: float
    expiry: float          # in years


class CalibrationResult(NamedTuple):
    params: SVIParams
    rmse: float
    max_error: float
    butterfly_ok: bool
    calendar_ok: bool


# ---------------------------------------------------------------------------
# SVI core
# ---------------------------------------------------------------------------

def svi_total_variance(k: float, params: SVIParams) -> float:
    """
    SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))

    Args:
        k: log-moneyness log(K/F)
        params: SVI parameters

    Returns:
        Total variance w (non-negative when well-calibrated)
    """
    a, b, rho, m, sigma = params
    z = k - m
    return a + b * (rho * z + math.sqrt(z * z + sigma * sigma))


def svi_implied_vol(k: float, expiry: float, params: SVIParams) -> float:
    """Convert SVI total variance to implied vol."""
    w = svi_total_variance(k, params)
    w = max(w, 1e-10)
    return math.sqrt(w / expiry)


def svi_gradient(k: float, params: SVIParams) -> List[float]:
    """
    Gradient of w(k) w.r.t. (a, b, rho, m, sigma).
    Used in Levenberg-Marquardt calibration.
    """
    a, b, rho, m, sigma = params
    z = k - m
    d = math.sqrt(z * z + sigma * sigma)

    dw_da = 1.0
    dw_db = rho * z + d
    dw_drho = b * z
    dw_dm = b * (-rho - z / d)
    dw_dsigma = b * sigma / d

    return [dw_da, dw_db, dw_drho, dw_dm, dw_dsigma]


def svi_butterfly_check(params: SVIParams, k_grid: Optional[List[float]] = None) -> bool:
    """
    Check butterfly arbitrage: g(k) = (1 - k*w'/(2w))^2 - (w'/2)^2*(1/4 + 1/w) + w''/2 >= 0

    A simplified local vol density check.
    """
    if k_grid is None:
        k_grid = [i * 0.05 for i in range(-40, 41)]

    a, b, rho, m, sigma = params

    for k in k_grid:
        z = k - m
        d = math.sqrt(z * z + sigma * sigma)

        w = a + b * (rho * z + d)
        if w <= 0:
            return False

        # First derivative
        wp = b * (rho + z / d)

        # Second derivative
        wpp = b * sigma * sigma / (d * d * d)

        # Gatheral's g(k)
        term1 = (1.0 - k * wp / (2.0 * w)) ** 2
        term2 = (wp ** 2 / 4.0) * (1.0 / 4.0 + 1.0 / w)
        g = term1 - term2 + wpp / 2.0

        if g < -1e-8:
            return False

    return True


def svi_calendar_check(params_t1: SVIParams, params_t2: SVIParams,
                        t1: float, t2: float,
                        k_grid: Optional[List[float]] = None) -> bool:
    """
    Check calendar spread: total variance must be non-decreasing in time.
    w(k, T2) >= w(k, T1) for all k.
    """
    if k_grid is None:
        k_grid = [i * 0.05 for i in range(-40, 41)]

    for k in k_grid:
        w1 = svi_total_variance(k, params_t1)
        w2 = svi_total_variance(k, params_t2)
        if w2 < w1 - 1e-8:
            return False
    return True


# ---------------------------------------------------------------------------
# SVI calibration via Nelder-Mead simplex
# ---------------------------------------------------------------------------

def _svi_loss(params_vec: List[float], smile_points: List[SmilePoint],
              expiry: float, penalty_weight: float = 100.0) -> float:
    """Loss function: RMSE + butterfly penalty."""
    a, b, rho, m, sigma = params_vec

    # Hard constraints
    if b <= 0 or sigma <= 0 or abs(rho) >= 1:
        return 1e12
    if a + b * sigma * math.sqrt(1 - rho * rho) < 0:
        return 1e12

    params = SVIParams(a, b, rho, m, sigma)

    # RMSE
    sse = 0.0
    for pt in smile_points:
        vol_fit = svi_implied_vol(pt.log_moneyness, expiry, params)
        sse += (vol_fit - pt.implied_vol) ** 2
    rmse = math.sqrt(sse / len(smile_points))

    # Butterfly penalty
    butterfly_ok = svi_butterfly_check(params)
    penalty = 0.0 if butterfly_ok else penalty_weight

    return rmse + penalty


def _nelder_mead(func, x0: List[float], max_iter: int = 2000,
                 tol: float = 1e-8) -> Tuple[List[float], float]:
    """
    Pure Python Nelder-Mead simplex optimizer.
    """
    n = len(x0)
    # Build initial simplex
    simplex = [list(x0)]
    for i in range(n):
        xi = list(x0)
        xi[i] += 0.1 if abs(x0[i]) < 1e-6 else 0.1 * abs(x0[i])
        simplex.append(xi)

    fvals = [func(x) for x in simplex]

    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho_nm = 0.5  # contraction
    sigma_nm = 0.5  # shrink

    for iteration in range(max_iter):
        # Sort
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        # Convergence check
        spread = max(abs(fvals[i] - fvals[0]) for i in range(1, n + 1))
        if spread < tol:
            break

        # Centroid of all but worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

        # Reflection
        xr = [centroid[j] + alpha * (centroid[j] - simplex[n][j]) for j in range(n)]
        fr = func(xr)

        if fvals[0] <= fr < fvals[n - 1]:
            simplex[n] = xr
            fvals[n] = fr
            continue

        if fr < fvals[0]:
            # Expansion
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = func(xe)
            if fe < fr:
                simplex[n] = xe
                fvals[n] = fe
            else:
                simplex[n] = xr
                fvals[n] = fr
            continue

        # Contraction
        if fr < fvals[n]:
            xc = [centroid[j] + rho_nm * (xr[j] - centroid[j]) for j in range(n)]
            fc = func(xc)
            if fc <= fr:
                simplex[n] = xc
                fvals[n] = fc
                continue
        else:
            xc = [centroid[j] + rho_nm * (simplex[n][j] - centroid[j]) for j in range(n)]
            fc = func(xc)
            if fc < fvals[n]:
                simplex[n] = xc
                fvals[n] = fc
                continue

        # Shrink
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = [best[j] + sigma_nm * (simplex[i][j] - best[j]) for j in range(n)]
            fvals[i] = func(simplex[i])

    return simplex[0], fvals[0]


def calibrate_svi(smile_points: List[SmilePoint],
                  expiry: float,
                  n_restarts: int = 5,
                  seed: int = 42) -> CalibrationResult:
    """
    Calibrate SVI parameters to a set of market implied vols.

    Args:
        smile_points: List of (strike, log_moneyness, implied_vol, expiry) tuples
        expiry: Option expiry in years
        n_restarts: Number of random restarts for global search
        seed: Random seed

    Returns:
        CalibrationResult with optimal params and diagnostics
    """
    rng = random.Random(seed)
    atm_var = min(pt.implied_vol for pt in smile_points) ** 2 * expiry

    best_loss = float("inf")
    best_params = None

    for _ in range(n_restarts):
        # Random starting point
        a0 = atm_var * rng.uniform(0.5, 1.5)
        b0 = rng.uniform(0.01, 0.3)
        rho0 = rng.uniform(-0.7, 0.0)
        m0 = rng.uniform(-0.1, 0.1)
        sigma0 = rng.uniform(0.1, 0.5)

        x0 = [a0, b0, rho0, m0, sigma0]

        def loss(pv):
            return _svi_loss(pv, smile_points, expiry)

        try:
            x_opt, f_opt = _nelder_mead(loss, x0)
        except Exception:
            continue

        if f_opt < best_loss:
            best_loss = f_opt
            best_params = x_opt

    if best_params is None:
        # Fallback: flat smile
        atm_vol = sum(pt.implied_vol for pt in smile_points) / len(smile_points)
        best_params = [atm_vol ** 2 * expiry, 0.01, -0.3, 0.0, 0.3]

    a, b, rho, m, sigma = best_params
    params = SVIParams(a, b, rho, m, sigma)

    # Compute diagnostics
    errors = []
    for pt in smile_points:
        vol_fit = svi_implied_vol(pt.log_moneyness, expiry, params)
        errors.append(abs(vol_fit - pt.implied_vol))

    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    max_err = max(errors)
    bf_ok = svi_butterfly_check(params)

    return CalibrationResult(
        params=params,
        rmse=rmse,
        max_error=max_err,
        butterfly_ok=bf_ok,
        calendar_ok=True   # Checked externally across expiries
    )


# ---------------------------------------------------------------------------
# Cubic spline smile interpolation
# ---------------------------------------------------------------------------

def _cubic_spline_coeffs(x: List[float], y: List[float]) -> List[Tuple[float, float, float, float]]:
    """
    Natural cubic spline coefficients using Thomas algorithm.
    Returns list of (a, b, c, d) per interval.
    """
    n = len(x)
    assert n >= 2

    h = [x[i + 1] - x[i] for i in range(n - 1)]

    # Build tridiagonal system for second derivatives
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i] * (y[i + 1] - y[i]) -
                    3.0 / h[i - 1] * (y[i] - y[i - 1]))

    l = [1.0] * n
    mu = [0.0] * n
    z = [0.0] * n

    for i in range(1, n - 1):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = [0.0] * n
    b = [0.0] * (n - 1)
    d = [0.0] * (n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]

    coeffs = []
    for i in range(n - 1):
        b_i = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        d_i = (c[i + 1] - c[i]) / (3.0 * h[i])
        coeffs.append((y[i], b_i, c[i], d_i))

    return coeffs


def spline_smile_interpolate(strikes: List[float], vols: List[float],
                              query_strikes: List[float]) -> List[float]:
    """
    Cubic spline interpolation of the vol smile.

    Args:
        strikes: Market strike levels (sorted ascending)
        vols: Corresponding implied vols
        query_strikes: Strikes to evaluate

    Returns:
        Interpolated (and extrapolated flat) implied vols
    """
    n = len(strikes)
    assert n >= 2
    coeffs = _cubic_spline_coeffs(strikes, vols)

    result = []
    for k in query_strikes:
        # Clamp to boundary for extrapolation (flat extrapolation)
        if k <= strikes[0]:
            result.append(vols[0])
            continue
        if k >= strikes[-1]:
            result.append(vols[-1])
            continue

        # Find interval
        idx = 0
        for i in range(n - 2):
            if k <= strikes[i + 1]:
                idx = i
                break
        else:
            idx = n - 2

        a, b, c, d = coeffs[idx]
        dx = k - strikes[idx]
        vol = a + b * dx + c * dx * dx + d * dx * dx * dx
        result.append(max(vol, 0.001))   # Floor at 0.1%

    return result


# ---------------------------------------------------------------------------
# SSVI (Surface SVI) — consistent parameterization across strikes and expiries
# ---------------------------------------------------------------------------

class SSVIParams(NamedTuple):
    """
    SSVI total variance: w(k, theta) = theta/2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1-rho^2))
    where theta = atm_total_var, phi = phi(theta) is the Heston-like wing slope.
    """
    rho: float     # correlation parameter
    gamma: float   # power law exponent for phi(theta) = eta / theta^gamma
    eta: float     # phi scaling


def ssvi_phi(theta: float, params: SSVIParams) -> float:
    """SSVI wing slope function phi(theta) = eta * theta^(-gamma)."""
    return params.eta * (theta ** (-params.gamma))


def ssvi_total_variance(k: float, theta: float, params: SSVIParams) -> float:
    """
    SSVI total variance slice.

    Args:
        k: log-moneyness
        theta: ATM total variance theta = sigma_atm^2 * T
        params: SSVI parameters

    Returns:
        Total variance w(k, theta)
    """
    rho = params.rho
    phi = ssvi_phi(theta, params)
    inner = (phi * k + rho) ** 2 + (1.0 - rho * rho)
    return theta / 2.0 * (1.0 + rho * phi * k + math.sqrt(inner))


def ssvi_arbitrage_free_check(params: SSVIParams,
                               thetas: List[float],
                               k_grid: Optional[List[float]] = None) -> bool:
    """
    Gatheral-Jacquier (2014) sufficient no-arb conditions:
    eta*(1 + |rho|) <= 2, and gamma in [0, 1].
    """
    rho = params.rho
    eta = params.eta
    gamma = params.gamma

    if not (0.0 <= gamma <= 1.0):
        return False
    if eta * (1.0 + abs(rho)) > 2.0 + 1e-8:
        return False
    if abs(rho) >= 1.0:
        return False

    return True


# ---------------------------------------------------------------------------
# Vol surface stitching across expiries
# ---------------------------------------------------------------------------

def build_svi_surface(expiries: List[float],
                       strikes_per_expiry: List[List[float]],
                       vols_per_expiry: List[List[float]],
                       forward_per_expiry: Optional[List[float]] = None) -> List[CalibrationResult]:
    """
    Calibrate SVI slice-by-slice, then check calendar spread constraints.

    Args:
        expiries: List of expiries in years (sorted ascending)
        strikes_per_expiry: Strikes for each expiry
        vols_per_expiry: Implied vols for each expiry
        forward_per_expiry: Forward prices (default = 1.0)

    Returns:
        List of CalibrationResult for each expiry
    """
    n = len(expiries)
    if forward_per_expiry is None:
        forward_per_expiry = [1.0] * n

    results = []
    for i in range(n):
        F = forward_per_expiry[i]
        T = expiries[i]
        strikes = strikes_per_expiry[i]
        vols = vols_per_expiry[i]

        smile_pts = [SmilePoint(K, math.log(K / F), v, T)
                     for K, v in zip(strikes, vols)]
        result = calibrate_svi(smile_pts, T, seed=i)
        results.append(result)

    # Mark calendar spread violations
    calendar_ok_flags = [True] * n
    for i in range(1, n):
        ok = svi_calendar_check(results[i - 1].params, results[i].params,
                                 expiries[i - 1], expiries[i])
        if not ok:
            calendar_ok_flags[i] = False

    # Rebuild results with calendar flag
    updated = []
    for i, r in enumerate(results):
        updated.append(CalibrationResult(
            params=r.params,
            rmse=r.rmse,
            max_error=r.max_error,
            butterfly_ok=r.butterfly_ok,
            calendar_ok=calendar_ok_flags[i]
        ))

    return updated


def interpolate_surface(expiries: List[float],
                         results: List[CalibrationResult],
                         query_expiry: float,
                         query_strikes: List[float],
                         forward: float = 1.0) -> List[float]:
    """
    Interpolate the SVI surface at a query (expiry, strikes) point.

    For expiries within calibrated range: linearly interpolate total variances.
    For out-of-range: flat extrapolation of nearest slice.
    """
    n = len(expiries)

    if query_expiry <= expiries[0]:
        # Flat extrapolation
        params = results[0].params
        return [svi_implied_vol(math.log(K / forward), query_expiry, params)
                for K in query_strikes]

    if query_expiry >= expiries[-1]:
        params = results[-1].params
        return [svi_implied_vol(math.log(K / forward), query_expiry, params)
                for K in query_strikes]

    # Find bracketing expiries
    idx = 0
    for i in range(n - 1):
        if expiries[i + 1] >= query_expiry:
            idx = i
            break

    t1, t2 = expiries[idx], expiries[idx + 1]
    alpha = (query_expiry - t1) / (t2 - t1)

    vols = []
    for K in query_strikes:
        k = math.log(K / forward)
        w1 = svi_total_variance(k, results[idx].params)
        w2 = svi_total_variance(k, results[idx + 1].params)
        # Linear interpolation in total variance
        w = (1 - alpha) * w1 + alpha * w2
        w = max(w, 1e-10)
        vols.append(math.sqrt(w / query_expiry))

    return vols


# ---------------------------------------------------------------------------
# Utility: risk-reversal and butterfly quotes → smile
# ---------------------------------------------------------------------------

def rr_bf_to_smile(atm: float, rr25: float, bf25: float,
                    call25_delta_moneyness: float = 0.15,
                    put25_delta_moneyness: float = -0.15) -> Tuple[float, float, float]:
    """
    Convert ATM, 25-delta risk-reversal and butterfly to smile vols.

    Market conventions:
        ATM (DNS): sigma_atm
        RR25 = sigma_25C - sigma_25P
        BF25 = (sigma_25C + sigma_25P)/2 - sigma_atm

    Args:
        atm: ATM vol (0.20 = 20%)
        rr25: 25-delta risk reversal
        bf25: 25-delta butterfly (market strangle)
        call25_delta_moneyness: approx log-moneyness for 25-delta call
        put25_delta_moneyness: approx log-moneyness for 25-delta put

    Returns:
        (vol_25P, vol_atm, vol_25C)
    """
    vol_25c = atm + bf25 + 0.5 * rr25
    vol_25p = atm + bf25 - 0.5 * rr25
    return vol_25p, atm, vol_25c


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def smile_diagnostics(params: SVIParams, expiry: float,
                       strikes: List[float], forward: float = 1.0) -> dict:
    """
    Return a diagnostic dict for a calibrated SVI smile.
    """
    vols = [svi_implied_vol(math.log(K / forward), expiry, params)
            for K in strikes]

    atm_vol = svi_implied_vol(0.0, expiry, params)
    atm_var = svi_total_variance(0.0, params)

    # Skew: d(sigma)/dk at k=0
    dk = 0.001
    skew = (svi_implied_vol(dk, expiry, params) -
            svi_implied_vol(-dk, expiry, params)) / (2 * dk)

    # Curvature (butterfly / convexity)
    d2 = (svi_implied_vol(dk, expiry, params) +
          svi_implied_vol(-dk, expiry, params) - 2 * atm_vol) / (dk ** 2)

    butterfly_ok = svi_butterfly_check(params)

    return {
        "atm_vol": atm_vol,
        "atm_total_variance": atm_var,
        "skew_at_atm": skew,
        "curvature_at_atm": d2,
        "svi_a": params.a,
        "svi_b": params.b,
        "svi_rho": params.rho,
        "svi_m": params.m,
        "svi_sigma": params.sigma,
        "butterfly_arbitrage_free": butterfly_ok,
        "min_vol": min(vols),
        "max_vol": max(vols),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SVI Smile Calibration Demo ===\n")

    # Synthetic market data: 3-month smile
    T = 0.25
    F = 100.0

    # (strike, vol) pairs simulating a realistic smile
    market_smile = [
        (80,  0.28), (85, 0.245), (90, 0.215), (95, 0.195),
        (100, 0.180), (105, 0.175), (110, 0.178), (115, 0.185), (120, 0.195)
    ]

    smile_pts = [SmilePoint(K, math.log(K / F), v, T) for K, v in market_smile]
    result = calibrate_svi(smile_pts, T, n_restarts=8)

    print(f"SVI Calibration (T={T}):")
    print(f"  RMSE: {result.rmse * 100:.4f} vol pts")
    print(f"  Max error: {result.max_error * 100:.4f} vol pts")
    print(f"  Butterfly arb-free: {result.butterfly_ok}")
    print(f"  Params: a={result.params.a:.6f}, b={result.params.b:.4f}, "
          f"rho={result.params.rho:.4f}, m={result.params.m:.4f}, "
          f"sigma={result.params.sigma:.4f}")

    diag = smile_diagnostics(result.params, T,
                              [pt.strike for pt in smile_pts], forward=F)
    print(f"\n  ATM vol: {diag['atm_vol']:.4f}")
    print(f"  Skew (d_vol/dk @ ATM): {diag['skew_at_atm']:.4f}")
    print(f"  Curvature: {diag['curvature_at_atm']:.4f}")

    # Cubic spline comparison
    print("\n=== Cubic Spline Interpolation ===")
    ks = [pt.strike for pt in smile_pts]
    vs = [pt.implied_vol for pt in smile_pts]
    query = [82, 88, 94, 100, 107, 113]
    interp_vols = spline_smile_interpolate(ks, vs, query)
    for K, v in zip(query, interp_vols):
        print(f"  K={K:6.1f}  vol={v:.4f}")

    # Multi-expiry surface
    print("\n=== Multi-Expiry SVI Surface ===")
    expiries = [0.25, 0.5, 1.0]
    strikes_all = [[80, 90, 100, 110, 120]] * 3
    # Slightly different vols for each expiry (term structure)
    vols_all = [
        [0.28, 0.215, 0.180, 0.175, 0.195],
        [0.255, 0.205, 0.175, 0.172, 0.185],
        [0.235, 0.195, 0.170, 0.168, 0.178],
    ]
    surface = build_svi_surface(expiries, strikes_all, vols_all)
    for T_i, r in zip(expiries, surface):
        print(f"  T={T_i:.2f}: RMSE={r.rmse * 100:.4f}%, bf_ok={r.butterfly_ok}, cal_ok={r.calendar_ok}")

    # Interpolate at T=0.75
    T_query = 0.75
    K_query = [85, 95, 100, 105, 115]
    vols_interp = interpolate_surface(expiries, surface, T_query, K_query, forward=F)
    print(f"\n  Interpolated vols at T={T_query}:")
    for K, v in zip(K_query, vols_interp):
        print(f"    K={K}  vol={v:.4f}")

    # RR/BF to smile
    print("\n=== Risk-Reversal / Butterfly to Smile ===")
    p25, atm_v, c25 = rr_bf_to_smile(atm=0.18, rr25=0.02, bf25=0.003)
    print(f"  25P vol: {p25:.4f}, ATM: {atm_v:.4f}, 25C vol: {c25:.4f}")

    # SSVI check
    print("\n=== SSVI Arbitrage-Free Check ===")
    ssvi = SSVIParams(rho=-0.4, gamma=0.5, eta=0.8)
    ok = ssvi_arbitrage_free_check(ssvi, thetas=[0.04, 0.08, 0.16])
    print(f"  SSVI no-arb conditions met: {ok}")
    for T_s, theta in [(0.25, 0.04), (0.5, 0.08), (1.0, 0.16)]:
        w0 = ssvi_total_variance(0.0, theta, ssvi)
        print(f"    T={T_s:.2f}: ATM total var={w0:.4f}, ATM vol={math.sqrt(w0/T_s):.4f}")
