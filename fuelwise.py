"""FuelWise engine: clean a fuel log, compute true consumption and costs, flag oddities.

Everything is pure pandas/numpy and runs in memory. Internally distances are km and
volumes are litres; conversion happens on the way in and out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KM_PER_MILE = 1.609344
L_PER_US_GAL = 3.785411784
L_PER_UK_GAL = 4.54609

DIST_UNITS = {"km": 1.0, "miles": KM_PER_MILE}
VOL_UNITS = {"L": 1.0, "US gal": L_PER_US_GAL, "UK gal": L_PER_UK_GAL}
CONS_METRICS = ["L/100km", "mpg (US)", "mpg (UK)", "km/L"]

# thresholds (documented in README)
MAX_LITRES_PER_FILL = 250.0
PLAUSIBLE_L100 = (1.5, 40.0)
RISE_PCT = 0.15          # interval this far above own baseline is flagged
MIN_BASELINE_INTERVALS = 3
BASELINE_WINDOW = 6


# ----------------------------------------------------------------- parsing helpers
def to_num(v) -> float:
    """Parse '1,234.5', '45,6', '$ 52.30', '12 L' into a float (NaN if impossible)."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = re.sub(r"[^0-9,.\-]", "", str(v))
    if s in ("", "-", ".", ","):
        return np.nan
    if "," in s and "." in s:
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        tail = s.rpartition(",")[2]
        s = s.replace(",", ".") if s.count(",") == 1 and len(tail) != 3 else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def to_flag(v):
    """Parse a yes/no style value into True/False/None."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in {"1", "1.0", "true", "yes", "y", "full", "x", "t", "✓"}:
        return True
    if s in {"0", "0.0", "false", "no", "n", "partial", "f", ""}:
        return False
    return None


def detect_columns(columns) -> dict:
    """Guess which column is which. Returns {role: column_name}; roles may be missing.

    Roles: date, odometer, volume, price, total, full, partial, missed.
    'partial' is an inverted full-tank flag (Fuelly style).
    """
    found: dict = {}
    for c in columns:
        n = str(c).strip().lower()
        if "missed" in n and "missed" not in found:
            found["missed"] = c
        elif "partial" in n and "partial" not in found:
            found["partial"] = c
        elif ("full" in n or "tank" in n) and "full" not in found:
            found["full"] = c
        elif ("date" in n or n in {"time", "datetime", "day"}) and "date" not in found:
            found["date"] = c
        elif re.search(r"odo|mileage|odometer", n) and "odometer" not in found:
            found["odometer"] = c
        elif re.search(r"total|cost|paid|amount|sum", n) and "total" not in found:
            found["total"] = c
        elif re.search(r"price|per ?l|/ ?l|per ?gal|/ ?gal|unit", n) and "price" not in found:
            found["price"] = c
        elif re.search(r"litre|liter|gallon|fuel|qty|quantity|volume|vol\b", n) and not re.search(
            r"econom|mpg|consum|l/100|type", n
        ) and "volume" not in found:
            found["volume"] = c
    return found


@dataclass
class CleanResult:
    data: pd.DataFrame                       # cleaned rows, SI units internally
    notes: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)


def _parse_dayfirst(raw):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(raw, errors="coerce", dayfirst=True)


def clean_log(df: pd.DataFrame, mapping: dict | None = None, dist_unit: str = "km",
              vol_unit: str = "L") -> CleanResult:
    """Normalise a raw log to columns: date, odo_km, litres, price_per_l, total, full, missed, issue."""
    if dist_unit not in DIST_UNITS or vol_unit not in VOL_UNITS:
        raise ValueError("unknown unit")
    mapping = mapping or detect_columns(df.columns)
    notes: list = []
    for need in ("odometer", "volume"):
        if need not in mapping:
            raise ValueError(f"Could not find a '{need}' column. Please rename it or map it manually.")
    if "price" not in mapping and "total" not in mapping:
        raise ValueError("Need a price-per-unit column or a total-cost column.")
    dk, vl = DIST_UNITS[dist_unit], VOL_UNITS[vol_unit]
    out = pd.DataFrame(index=df.index)
    if "date" in mapping:
        raw = df[mapping["date"]]
        d = pd.to_datetime(raw, errors="coerce")
        d2 = _parse_dayfirst(raw)
        out["date"] = d2 if d2.notna().sum() > d.notna().sum() else d
    else:
        out["date"] = pd.NaT
        notes.append("No date column found: monthly and yearly figures are unavailable.")
    out["odo_km"] = df[mapping["odometer"]].map(to_num) * dk
    out["litres"] = df[mapping["volume"]].map(to_num) * vl
    price_unit = df[mapping["price"]].map(to_num) if "price" in mapping else pd.Series(np.nan, index=df.index)
    total = df[mapping["total"]].map(to_num) if "total" in mapping else pd.Series(np.nan, index=df.index)
    out["price_per_l"] = price_unit / vl
    calc_total = out["price_per_l"] * out["litres"]
    out["total"] = total.where(total.notna() & (total > 0), calc_total)
    derived = out["litres"] > 0
    out["price_per_l"] = out["price_per_l"].where(out["price_per_l"].notna(), out["total"] / out["litres"].where(derived))
    if "full" in mapping:
        out["full"] = df[mapping["full"]].map(to_flag)
    elif "partial" in mapping:
        out["full"] = df[mapping["partial"]].map(lambda v: None if to_flag(v) is None else not to_flag(v))
    else:
        out["full"] = None
        notes.append("No full-tank column found: every fill-up is treated as a full tank.")
    out["full"] = out["full"].map(lambda v: True if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)).astype(bool)
    out["missed"] = df[mapping["missed"]].map(to_flag).fillna(False).astype(bool) if "missed" in mapping else False

    out = out.dropna(subset=["odo_km", "litres"], how="all")
    out["issue"] = ""
    out = _sort_and_flag(out)
    return CleanResult(out.reset_index(drop=True), notes, mapping)


def _sort_and_flag(out: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and mark rows that cannot be used."""
    if out["date"].notna().any():
        out = out.sort_values(["date", "odo_km"], kind="stable")
    else:
        out = out.sort_values("odo_km", kind="stable")
    out = out.copy()
    issues = []
    prev_odo = -np.inf
    med_price = out["price_per_l"].where(out["price_per_l"] > 0).median()
    for _, r in out.iterrows():
        msg = []
        if pd.isna(r.odo_km) or r.odo_km <= 0:
            msg.append("odometer missing or not positive")
        elif r.odo_km <= prev_odo:
            msg.append("odometer is not higher than the previous entry (typo or wrong order?)")
        if pd.isna(r.litres) or r.litres <= 0:
            msg.append("volume missing or not positive")
        elif r.litres > MAX_LITRES_PER_FILL:
            msg.append(f"volume above {MAX_LITRES_PER_FILL:.0f} L is unlikely for a car")
        if pd.notna(med_price) and pd.notna(r.price_per_l) and r.price_per_l > 0 and not (
            med_price / 3 <= r.price_per_l <= med_price * 3
        ):
            msg.append("price per litre is far from the other entries")
        if pd.notna(r.total) and r.total < 0:
            msg.append("negative cost")
        if not msg and pd.notna(r.odo_km):
            prev_odo = r.odo_km
        issues.append("; ".join(msg))
    out["issue"] = issues
    return out


# ----------------------------------------------------------------- core calculation
@dataclass
class Analysis:
    rows: pd.DataFrame          # all rows with issue column
    intervals: pd.DataFrame     # one row per full-to-full interval
    flags: list                 # list of dict(kind, text)
    stats: dict


def analyse(clean: pd.DataFrame) -> Analysis:
    """Full-tank-to-full-tank consumption.

    For each full fill-up we take the fuel added at that fill-up plus any partial fills since the
    previous full fill-up, and divide by the distance between those two full fill-ups. The fuel in
    the *first* full fill-up is not counted (we do not know how empty the tank was before it).
    A row marked 'missed' breaks the chain, as does an implausible result.
    """
    rows = clean.copy()
    ok = rows[rows["issue"] == ""].reset_index(drop=True)
    recs = []
    last_full = None          # position of previous full fill in ok
    acc_l = acc_cost = 0.0
    acc_cost_ok = True
    for i, r in ok.iterrows():
        if r.missed:
            last_full = None
            acc_l = acc_cost = 0.0
            acc_cost_ok = True
            if r.full:
                last_full = i
            continue
        acc_l += r.litres
        if pd.isna(r.total):
            acc_cost_ok = False
        else:
            acc_cost += r.total
        if not r.full:
            continue
        if last_full is not None:
            prev = ok.loc[last_full]
            dist = r.odo_km - prev.odo_km
            if dist > 0:
                recs.append({
                    "date": r.date, "odo_km": r.odo_km, "start_odo_km": prev.odo_km,
                    "dist_km": dist, "litres": acc_l,
                    "cost": acc_cost if acc_cost_ok else np.nan,
                    "l_per_100km": acc_l / dist * 100,
                    "price_per_l": r.price_per_l,
                })
        last_full = i
        acc_l = acc_cost = 0.0
        acc_cost_ok = True
    iv = pd.DataFrame(recs, columns=["date", "odo_km", "start_odo_km", "dist_km", "litres", "cost",
                                     "l_per_100km", "price_per_l"])
    flags: list = []
    iv["suspect"] = False
    iv["rise"] = False
    if len(iv):
        lo, hi = PLAUSIBLE_L100
        bad = (iv.l_per_100km < lo) | (iv.l_per_100km > hi)
        med = iv.l_per_100km[~bad].median() if (~bad).any() else np.nan
        if pd.notna(med):
            bad |= (iv.l_per_100km > med * 2.5) | (iv.l_per_100km < med / 2.5)
        iv.loc[bad, "suspect"] = True
        for _, r in iv[iv.suspect].iterrows():
            flags.append({"kind": "implausible", "text": (
                f"Around odometer {r.odo_km:,.0f} km the computed consumption ({r.l_per_100km:.1f} L/100km) looks "
                "implausible - a fill-up may be missing or a value mistyped. It is left out of averages.")})
        good = iv[~iv.suspect].reset_index()
        # baseline = median of up to BASELINE_WINDOW preceding good intervals (the car's own history)
        for k, r in good.iterrows():
            prior = good.l_per_100km.iloc[max(0, k - BASELINE_WINDOW):k]
            if len(prior) >= MIN_BASELINE_INTERVALS and r.l_per_100km > prior.median() * (1 + RISE_PCT):
                iv.loc[r["index"], "rise"] = True
    # rolling (distance-weighted over the last 3 good intervals)
    gi = iv[~iv.suspect]
    roll = []
    for idx in iv.index:
        w = gi[gi.index <= idx].tail(3)
        roll.append(w.litres.sum() / w.dist_km.sum() * 100 if len(w) else np.nan)
    iv["rolling_l100"] = roll
    iv["cost_per_km"] = iv.cost / iv.dist_km

    # sustained shift: best single step change (two-segment fit) that persists to the latest fill-up
    shift = _step_change(gi)
    if shift:
        k, b_val, r_val, pct = shift
        when = gi.date.iloc[k]
        when_txt = f" since about {when.strftime('%b %Y')}" if pd.notna(when) else " recently"
        flags.append({"kind": "rise", "text": (
            f"Consumption has averaged {r_val:.1f} L/100km{when_txt}, about {pct*100:.0f}% above this car's own earlier "
            f"level of {b_val:.1f}. It may be worth checking things like tyre pressure, brakes, air filter or service "
            "history, and whether your driving mix changed (winter, more city driving, towing, a new route). "
            "This is only a statistical pattern in your log, not a diagnosis.")})
        iv.loc[gi.index[k:], "rise"] = True
    elif iv.rise.any():
        flags.append({"kind": "spike", "text": f"{int(iv.rise.sum())} interval(s) were noticeably above the car's own baseline but it did not persist. Usually a trip pattern or a partly-missed entry; worth a glance."})

    for _, r in rows[rows.issue != ""].iterrows():
        flags.append({"kind": "data", "text": f"Entry at odometer {r.odo_km:,.0f} km"
                      f"{'' if pd.isna(r.date) else ' on ' + r.date.strftime('%Y-%m-%d')} was ignored: {r.issue}."})
    stats = _stats(ok, iv, rows)
    return Analysis(rows, iv, flags, stats)


def _step_change(gi: pd.DataFrame):
    """Return (index, before_L100, after_L100, pct) for the most convincing persistent rise, else None."""
    n = len(gi)
    if n < 2 * MIN_BASELINE_INTERVALS:
        return None
    best = None
    for k in range(MIN_BASELINE_INTERVALS, n - 2):          # at least 3 intervals in the later segment
        bef, aft = gi.iloc[:k], gi.iloc[k:]
        b_val = bef.litres.sum() / bef.dist_km.sum() * 100
        a_val = aft.litres.sum() / aft.dist_km.sum() * 100
        pct = a_val / b_val - 1
        sse = ((bef.l_per_100km - bef.l_per_100km.mean()) ** 2).sum() + ((aft.l_per_100km - aft.l_per_100km.mean()) ** 2).sum()
        if pct > RISE_PCT and (best is None or sse < best[0]):
            best = (sse, k, b_val, a_val, pct)
    if not best:
        return None
    _, k, b_val, a_val, pct = best
    bef, aft = gi.l_per_100km.iloc[:k], gi.l_per_100km.iloc[k:]
    se = np.sqrt(bef.var(ddof=1) / len(bef) + aft.var(ddof=1) / len(aft))
    if not np.isfinite(se) or (aft.mean() - bef.mean()) / max(se, 1e-9) < 3:
        return None
    return k, b_val, a_val, pct


def _stats(ok: pd.DataFrame, iv: pd.DataFrame, rows: pd.DataFrame) -> dict:
    s: dict = {"n_rows": len(rows), "n_used": len(ok), "n_intervals": len(iv)}
    gi = iv[~iv.suspect] if len(iv) else iv
    if len(gi):
        s["avg_l100"] = gi.litres.sum() / gi.dist_km.sum() * 100
        s["last_l100"] = gi.rolling_l100.iloc[-1]
        s["interval_km"] = gi.dist_km.sum()
        cc = gi.dropna(subset=["cost"])
        if len(cc) and cc.dist_km.sum() > 0:
            s["cost_per_km"] = cc.cost.sum() / cc.dist_km.sum()
    if len(ok):
        s["total_spent"] = ok.total.sum(skipna=True)
        s["total_litres"] = ok.litres.sum()
        s["km_span"] = ok.odo_km.max() - ok.odo_km.min()
        dates = ok.date.dropna()
        if len(dates) >= 2:
            span = (dates.max() - dates.min()).days
            s["days"] = span
            if span >= 30 and s["km_span"] > 0:
                # exclude first fill (its fuel covers driving before the log began)
                first = ok.iloc[0]
                spent = ok.total.sum(skipna=True) - (0 if pd.isna(first.total) else first.total)
                s["per_year"] = spent / span * 365
                s["per_month"] = s["per_year"] / 12
                s["km_per_year"] = s["km_span"] / span * 365
    return s


def monthly_costs(clean: pd.DataFrame) -> pd.DataFrame:
    ok = clean[(clean.issue == "") & clean.date.notna()]
    if ok.empty:
        return pd.DataFrame(columns=["month", "spent", "litres", "avg_price_per_l"])
    g = ok.groupby(ok.date.dt.to_period("M"))
    out = pd.DataFrame({"spent": g.total.sum(), "litres": g.litres.sum(),
                        "avg_price_per_l": g.total.sum() / g.litres.sum()}).reset_index()
    out["month"] = out["date"].dt.to_timestamp()
    return out.drop(columns="date")[["month", "spent", "litres", "avg_price_per_l"]]


def yearly_costs(clean: pd.DataFrame) -> pd.DataFrame:
    ok = clean[(clean.issue == "") & clean.date.notna()]
    if ok.empty:
        return pd.DataFrame(columns=["year", "spent", "litres"])
    g = ok.groupby(ok.date.dt.year)
    return pd.DataFrame({"year": g.total.sum().index, "spent": g.total.sum().values,
                         "litres": g.litres.sum().values})


# ----------------------------------------------------------------- unit output
def conv_consumption(l100, metric: str):
    """Convert L/100km to the requested display metric."""
    l100 = np.asarray(l100, dtype=float) if not np.isscalar(l100) else float(l100)
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "L/100km":
            return l100
        if metric == "mpg (US)":
            return 235.214583 / l100
        if metric == "mpg (UK)":
            return 282.480936 / l100
        if metric == "km/L":
            return 100 / l100
    raise ValueError(metric)


def cost_per_distance(cost_per_km, dist_unit: str):
    return cost_per_km * DIST_UNITS[dist_unit]


def price_per_volume(price_per_l, vol_unit: str):
    return price_per_l * VOL_UNITS[vol_unit]


def lower_is_better(metric: str) -> bool:
    return metric == "L/100km"


def tco(annual_distance_km: float, fuel_per_year: float, insurance: float, maintenance: float,
        depreciation: float) -> dict:
    """Total cost of ownership from user-entered yearly figures."""
    total = fuel_per_year + insurance + maintenance + depreciation
    return {
        "fuel": fuel_per_year, "insurance": insurance, "maintenance": maintenance,
        "depreciation": depreciation, "per_year": total, "per_month": total / 12,
        "per_km": total / annual_distance_km if annual_distance_km > 0 else np.nan,
        "fuel_share": fuel_per_year / total if total > 0 else np.nan,
    }


def summary_text(a: Analysis, metric: str, dist_unit: str, vol_unit: str, currency: str) -> str:
    s = a.stats
    if not s.get("n_intervals"):
        return ("Not enough data yet: FuelWise needs at least two full-tank fill-ups with increasing odometer "
                "readings to work out true consumption.")
    f = lambda x: f"{conv_consumption(x, metric):.1f} {metric}"
    parts = [f"Across {s['n_intervals']} full-tank intervals ({s['interval_km']/DIST_UNITS[dist_unit]:,.0f} {dist_unit}) "
             f"your car averages **{f(s['avg_l100'])}**; the latest rolling figure is **{f(s['last_l100'])}**."]
    if "cost_per_km" in s:
        parts.append(f"Fuel costs about **{currency}{cost_per_distance(s['cost_per_km'], dist_unit):.3f} per {dist_unit[:-1] if dist_unit=='miles' else dist_unit}**.")
    if "per_month" in s:
        parts.append(f"That is roughly **{currency}{s['per_month']:,.0f} a month** or **{currency}{s['per_year']:,.0f} a year** at your current pace.")
    kinds = {x["kind"] for x in a.flags}
    if "rise" in kinds:
        parts.append("Your recent consumption is noticeably above the car's own baseline - it may be worth checking tyre pressure and servicing (details below).")
    elif not kinds - {"data"}:
        parts.append("Nothing unusual stands out in consumption.")
    return " ".join(parts)


# ----------------------------------------------------------------- demo data
def make_demo(seed: int = 7) -> pd.DataFrame:
    """Synthetic log: ~6.8 L/100km, two partial fills, then a planted jump to ~8.4 from fill 20."""
    rng = np.random.default_rng(seed)
    odo, date = 42000.0, pd.Timestamp("2024-01-06")
    rows = []
    for i in range(34):
        dist = rng.uniform(520, 760)
        l100 = (6.8 if i < 20 else 8.4) + rng.normal(0, 0.18)
        litres = dist * l100 / 100
        odo += dist
        date += pd.Timedelta(days=int(dist / 38) + int(rng.integers(0, 3)))
        price = 1.72 + 0.12 * np.sin(i / 5) + rng.normal(0, 0.015)
        full = True
        if i in (8, 21):          # partial top-ups: split one fill into two entries
            rows.append((date - pd.Timedelta(days=2), odo - dist * 0.4, round(litres * 0.35, 2), round(price, 3), False))
            litres *= 0.65
        rows.append((date, odo, round(litres, 2), round(price, 3), full))
    df = pd.DataFrame(rows, columns=["Date", "Odometer (km)", "Fuel (L)", "Price per L (EUR)", "Full tank"])
    df["Total cost (EUR)"] = (df["Fuel (L)"] * df["Price per L (EUR)"]).round(2)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Odometer (km)"] = df["Odometer (km)"].round(0).astype(int)
    df["Full tank"] = df["Full tank"].map({True: 1, False: 0})
    return df
