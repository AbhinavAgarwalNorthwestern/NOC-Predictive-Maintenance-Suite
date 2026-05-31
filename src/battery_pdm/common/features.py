"""Shared feature pipeline — extensible registry of feature groups.

Each feature group computes a DataFrame indexed by _row_id (one row per
observation). Groups are composable — pick which ones to use at training
time and the same set is required at inference.

Registered groups (current):
    - alarm_history: counters from alarm stream, point-in-time at mains_fail_h
    - site_static: per-site config (manufacturer, load, install)

Adding telemetry later: write a new group that takes telemetry as input,
register it, include it in the training feature list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class FeatureGroup:
    name: str
    compute_fn: Callable
    requires: tuple[str, ...]  # which input DataFrames the group needs


_REGISTRY: dict[str, FeatureGroup] = {}


def register_feature_group(name: str, requires: tuple[str, ...]):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = FeatureGroup(name=name, compute_fn=fn, requires=requires)
        return fn

    return decorator


def list_groups() -> list[str]:
    return list(_REGISTRY.keys())


def compute_features(
    labels: pd.DataFrame,
    groups: list[str],
    inputs: dict[str, pd.DataFrame],
    ref_time_col: str = "mains_fail_h",
) -> pd.DataFrame:
    """Compute selected feature groups and concatenate column-wise.

    Parameters
    ----------
    labels : DataFrame with site_id and a reference time column
    groups : list of group names to compute
    inputs : dict of source DataFrames (e.g., {'alarms': ..., 'site_static': ..., 'telemetry': ...})
    ref_time_col : name of the timestamp column in labels (default 'mains_fail_h'
        for the autonomy model; use 'event_hour' for long-term failure labels)
    """
    # Internally we always use 'ref_time_h' so feature group implementations
    # don't need to know which model they're serving
    base = labels[["site_id", ref_time_col]].copy().reset_index(drop=True)
    base = base.rename(columns={ref_time_col: "ref_time_h"})
    base["_row_id"] = base.index
    out_pieces = [base]

    for name in groups:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown feature group: {name}. Available: {list_groups()}")
        spec = _REGISTRY[name]
        missing = [r for r in spec.requires if r not in inputs]
        if missing:
            raise ValueError(
                f"Group '{name}' requires inputs {spec.requires} but missing: {missing}"
            )
        group_df = spec.compute_fn(labels=base, **{k: inputs[k] for k in spec.requires})
        if "_row_id" not in group_df.columns:
            raise ValueError(f"Group '{name}' must preserve _row_id in its output")
        dup_cols = [c for c in group_df.columns if c in base.columns and c != "_row_id"]
        group_df = group_df.drop(columns=dup_cols)
        out_pieces.append(group_df.set_index("_row_id"))

    out = base.set_index("_row_id")
    for piece in out_pieces[1:]:
        out = out.join(piece, how="left")
    # Rename ref_time_h back to caller's column name so output is intuitive
    out = out.reset_index(drop=True).rename(columns={"ref_time_h": ref_time_col})
    return out


# ---------------------------------------------------------------------------
# Built-in feature groups
# ---------------------------------------------------------------------------


def _compute_soc_ledger(
    site_alarms: pd.DataFrame,
    t0: float,
    load_A: float,
    capacity_Ah: float,
    charger_A: float = 15.0,
    lifecycle_start_h: float = 0.0,
) -> dict:
    """Reconstruct approximate SoC trajectory from alarm timestamps up to t0.

    Treats each AC_MAINS_FAIL as start-of-discharge, next event (or t0) as
    end-of-discharge. During grid-on periods, battery recharges at charger_A.
    Returns aggregated state-of-charge proxies usable as features.

    `lifecycle_start_h` lets multi-lifecycle sites reset SoC=1 at battery
    install/replacement time rather than at t=0.

    Note: SoC values here are approximate — they don't account for rectifier
    faults (which our simulator marks as RECTIFIER_FAULT alarm but still keep
    grid 'on' from an alarm perspective). Good enough as a feature.
    """
    if site_alarms.empty:
        return {
            "estimated_soc_at_ref": 1.0,
            "frac_time_offgrid_30d": 0.0,
            "hours_recharged_30d": 720.0,
            "hours_discharged_30d": 0.0,
            "hours_since_full_recharge": 0.0,
            "avg_recharge_gap_30d": 720.0,
            "longest_outage_30d": 0.0,
            "outage_duration_p95_lifetime": 0.0,
            "cumulative_discharge_hours": 0.0,
        }

    ts = site_alarms["timestamp_h"].values
    codes = site_alarms["alarm_code"].values

    # Identify outage start events (AC_MAINS_FAIL) and termination events
    # (LOAD_DISCONNECT in same outage, or next AC_MAINS_FAIL = grid restored)
    mf_mask = codes == "AC_MAINS_FAIL"
    mf_times = ts[mf_mask]

    outage_durations = []  # (start, duration_h)
    for i, start in enumerate(mf_times):
        next_start = mf_times[i + 1] if i + 1 < len(mf_times) else t0
        # Find LOAD_DISCONNECT after this outage start, before next outage
        future = (ts > start) & (ts < next_start)
        lvd_mask = future & (codes == "LOAD_DISCONNECT")
        if lvd_mask.any():
            end = ts[lvd_mask][0]
        else:
            # Assume grid restored before next outage; treat outage as ending
            # mid-way (since we don't have telemetry to know exact restore time).
            # Conservative: cap outage at min(next_start - start, 12h typical)
            end = min(next_start, start + 12.0)
        if end > t0:
            end = t0
        outage_durations.append((start, end - start))
        if end >= t0:
            break  # don't process outages beyond ref time

    # Coulomb ledger: SoC=1 at lifecycle start, then evolve event by event
    soc = 1.0
    last_t = float(lifecycle_start_h)
    cumulative_disch = 0.0
    last_full_recharge_t = float(lifecycle_start_h)  # last time SoC >= 0.98
    for start, dur in outage_durations:
        # Grid was on from last_t -> start: SoC recharged
        on_grid_dur = max(0.0, start - last_t)
        # Add SoC: charge_A * hours / capacity, capped at 1.0
        soc = min(1.0, soc + (charger_A * on_grid_dur) / max(capacity_Ah, 1.0))
        if soc >= 0.98:
            last_full_recharge_t = start
        # Discharge for `dur` hours
        delta_soc = (load_A * dur) / max(capacity_Ah, 1.0)
        soc = max(0.0, soc - delta_soc)
        cumulative_disch += dur
        last_t = start + dur

    # Final segment: from last_t -> t0 grid was on, battery recharges
    soc = min(1.0, soc + (charger_A * max(0.0, t0 - last_t)) / max(capacity_Ah, 1.0))
    if soc >= 0.98 and t0 > last_t:
        last_full_recharge_t = t0

    hours_since_full = t0 - last_full_recharge_t

    # Last 30 days windowed aggregates
    cutoff_30 = t0 - 30 * 24
    recent = [(s, d) for s, d in outage_durations if s + d > cutoff_30]
    hours_discharged_30d = sum(min(s + d, t0) - max(s, cutoff_30) for s, d in recent)
    hours_recharged_30d = max(0.0, 30 * 24 - hours_discharged_30d)
    longest_outage_30d = max((d for s, d in recent), default=0.0)
    if len(recent) >= 2:
        gaps = [
            recent[i + 1][0] - (recent[i][0] + recent[i][1])
            for i in range(len(recent) - 1)
        ]
        avg_recharge_gap_30d = float(sum(gaps) / len(gaps))
    else:
        avg_recharge_gap_30d = float(30 * 24)

    if outage_durations:
        all_durs = [d for _, d in outage_durations]
        p95 = float(pd.Series(all_durs).quantile(0.95))
    else:
        p95 = 0.0

    return {
        "estimated_soc_at_ref": float(soc),
        "frac_time_offgrid_30d": float(hours_discharged_30d / (30 * 24)),
        "hours_recharged_30d": float(hours_recharged_30d),
        "hours_discharged_30d": float(hours_discharged_30d),
        "hours_since_full_recharge": float(hours_since_full),
        "avg_recharge_gap_30d": float(avg_recharge_gap_30d),
        "longest_outage_30d": float(longest_outage_30d),
        "outage_duration_p95_lifetime": p95,
        "cumulative_discharge_hours": float(cumulative_disch),
    }


@register_feature_group("soc_proxy", requires=("alarms", "site_static"))
def _soc_proxy_features(
    labels: pd.DataFrame, alarms: pd.DataFrame, site_static: pd.DataFrame
) -> pd.DataFrame:
    """SoC ledger reconstructed from alarm timing + site load.

    For each label, replays the alarm history up to ref_time_h, tracking an
    approximate state-of-charge via coulomb counting. Produces SoC-proxy
    features that capture battery state without needing voltage telemetry.
    """
    alarms = alarms.sort_values(["site_id", "timestamp_h"]).reset_index(drop=True)
    grouped = {s: g.reset_index(drop=True) for s, g in alarms.groupby("site_id")}

    static_lookup = {}
    for _, srow in site_static.iterrows():
        static_lookup[srow["site_id"]] = srow

    has_lifecycle_start = "lifecycle_start_h" in labels.columns

    row_id_col = labels["_row_id"].values
    site_id_col = labels["site_id"].values
    ref_time_col = labels["ref_time_h"].values
    lifecycle_col = labels["lifecycle_start_h"].values if has_lifecycle_start else None

    rows = []
    for i in range(len(labels)):
        site_id = site_id_col[i]
        t0 = ref_time_col[i]
        row_id = row_id_col[i]
        g = grouped.get(site_id)
        site_cfg = static_lookup.get(site_id)
        if g is None or site_cfg is None:
            rows.append({"_row_id": row_id})
            continue
        lifecycle_start = float(lifecycle_col[i]) if lifecycle_col is not None else 0.0
        past = g[(g["timestamp_h"] >= lifecycle_start) & (g["timestamp_h"] < t0)]
        ledger = _compute_soc_ledger(
            past,
            t0,
            load_A=float(site_cfg["load_A"]),
            capacity_Ah=float(site_cfg["nominal_capacity_ah"]),
            lifecycle_start_h=lifecycle_start,
        )
        ledger["_row_id"] = row_id
        rows.append(ledger)
    return pd.DataFrame(rows)


@register_feature_group("alarm_history", requires=("alarms",))
def _alarm_history_features(labels: pd.DataFrame, alarms: pd.DataFrame) -> pd.DataFrame:
    """Alarm-derived counters up to each label's ref_time_h (point-in-time correct).

    Vectorized via sorted-merge: for each label, find alarms with timestamp < t0
    using searchsorted, then slice and count by alarm_code.

    Used by both:
      - autonomy model (ref_time_h = AC_MAINS_FAIL timestamp)
      - long-term failure model (ref_time_h = event_hour at lifecycle end)
    """
    alarms = alarms.sort_values(["site_id", "timestamp_h"]).reset_index(drop=True)
    alarms["_is_critical"] = (alarms["severity"] == "critical").astype(int)
    alarms["_is_ticket"] = (
        alarms["alarm_code"].astype(str).str.startswith("TICKET").astype(int)
    )

    codes_of_interest = [
        "RECTIFIER_FAULT",
        "CELL_IMBALANCE",
        "AC_MAINS_FAIL",
        "BATT_UNDERVOLTAGE",
        "LOAD_DISCONNECT",
        "REPEAT_FAILURE_FLAG",
        "BATT_HIGH_TEMP",
    ]
    for code in codes_of_interest:
        alarms[f"_is_{code}"] = (alarms["alarm_code"] == code).astype(int)

    grouped = {site: g.reset_index(drop=True) for site, g in alarms.groupby("site_id")}

    row_id_col = labels["_row_id"].values
    site_id_col = labels["site_id"].values
    ref_time_col = labels["ref_time_h"].values

    rows = []
    for i in range(len(labels)):
        site_id = site_id_col[i]
        t0 = ref_time_col[i]
        row_id = row_id_col[i]
        g = grouped.get(site_id)
        if g is None:
            rows.append({"_row_id": row_id})
            continue

        ts = g["timestamp_h"].values
        idx_t0 = ts.searchsorted(t0, side="left")
        idx_30 = ts.searchsorted(t0 - 30 * 24, side="left")
        idx_7 = ts.searchsorted(t0 - 7 * 24, side="left")

        past = g.iloc[:idx_t0]
        recent_30d = g.iloc[idx_30:idx_t0]
        recent_7d = g.iloc[idx_7:idx_t0]

        prior_mf_mask = past["_is_AC_MAINS_FAIL"].values == 1
        if prior_mf_mask.any():
            last_mf_t = past["timestamp_h"].values[prior_mf_mask][-1]
            hours_since_last_mf = float(t0 - last_mf_t)
        else:
            hours_since_last_mf = 99999.0

        rows.append(
            {
                "_row_id": row_id,
                "rectifier_fault_count_lifetime": int(
                    past["_is_RECTIFIER_FAULT"].sum()
                ),
                "rectifier_fault_count_30d": int(
                    recent_30d["_is_RECTIFIER_FAULT"].sum()
                ),
                "cell_imbalance_count_lifetime": int(past["_is_CELL_IMBALANCE"].sum()),
                "cell_imbalance_count_30d": int(recent_30d["_is_CELL_IMBALANCE"].sum()),
                "mains_fail_count_lifetime": int(past["_is_AC_MAINS_FAIL"].sum()),
                "mains_fail_count_30d": int(recent_30d["_is_AC_MAINS_FAIL"].sum()),
                "mains_fail_count_7d": int(recent_7d["_is_AC_MAINS_FAIL"].sum()),
                "hours_since_last_mains_fail": hours_since_last_mf,
                "critical_alarm_count_30d": int(recent_30d["_is_critical"].sum()),
                "undervoltage_count_30d": int(
                    recent_30d["_is_BATT_UNDERVOLTAGE"].sum()
                ),
                "undervoltage_count_lifetime": int(past["_is_BATT_UNDERVOLTAGE"].sum()),
                "high_temp_count_30d": int(recent_30d["_is_BATT_HIGH_TEMP"].sum()),
                "lvd_count_30d": int(recent_30d["_is_LOAD_DISCONNECT"].sum()),
                "lvd_count_lifetime": int(past["_is_LOAD_DISCONNECT"].sum()),
                "has_repeat_failure": int(past["_is_REPEAT_FAILURE_FLAG"].sum() > 0),
                "ticket_count_lifetime": int(past["_is_ticket"].sum()),
            }
        )
    return pd.DataFrame(rows)


@register_feature_group("load_shedding_schedule", requires=("schedule", "site_static"))
def _load_shedding_features(
    labels: pd.DataFrame,
    schedule: pd.DataFrame,
    site_static: pd.DataFrame,
) -> pd.DataFrame:
    """Region-level load-shedding schedule features.

    For each label, looks up the utility's published off-grid schedule for the
    site's region around ref_time_h. Computes:
        - schedule_offgrid_hours_past_30d  (regional grid stress recently)
        - schedule_severity_mean_past_30d  (smoothed regional signal)
        - schedule_offgrid_hours_next_24h  (forecast of grid availability)
        - schedule_severity_max_next_24h
        - currently_in_peak_window
        - expected_daily_offgrid_now
    """
    import warnings as _warnings

    schedule = schedule.copy()
    schedule = schedule.sort_values(["region", "timestamp_h"]).reset_index(drop=True)

    site_to_region = dict(zip(site_static["site_id"], site_static["region"]))

    # Pre-index schedule by region for faster lookup
    region_dfs = {r: g.set_index("timestamp_h") for r, g in schedule.groupby("region")}

    # Warn loudly if any sites' regions are missing from the schedule
    site_regions_in_use = {
        site_to_region.get(s) for s in labels["site_id"].unique() if s in site_to_region
    }
    missing_regions = site_regions_in_use - set(region_dfs.keys()) - {None}
    if missing_regions:
        _warnings.warn(
            f"load_shedding_schedule: regions missing from schedule: {sorted(missing_regions)}. "
            f"Sites in these regions will get zero-filled schedule features.",
            stacklevel=2,
        )

    rows = []
    for _, row in labels.iterrows():
        site_id = row["site_id"]
        t0 = int(row["ref_time_h"])
        region = site_to_region.get(site_id)
        if region is None or region not in region_dfs:
            rows.append({"_row_id": row["_row_id"]})
            continue
        rdf = region_dfs[region]

        past_idx = rdf.index[(rdf.index < t0) & (rdf.index >= t0 - 30 * 24)]
        future_idx = rdf.index[(rdf.index >= t0) & (rdf.index < t0 + 24)]

        past = rdf.loc[past_idx]
        future = rdf.loc[future_idx]

        now_row = rdf.loc[t0] if t0 in rdf.index else None

        rows.append(
            {
                "_row_id": row["_row_id"],
                "schedule_offgrid_hours_past_30d": float(
                    past["scheduled_offgrid"].sum()
                )
                if not past.empty
                else 0.0,
                "schedule_severity_mean_past_30d": float(past["severity_score"].mean())
                if not past.empty
                else 0.0,
                "schedule_offgrid_hours_next_24h": float(
                    future["scheduled_offgrid"].sum()
                )
                if not future.empty
                else 0.0,
                "schedule_severity_max_next_24h": float(future["severity_score"].max())
                if not future.empty
                else 0.0,
                "currently_in_peak_window": int(now_row["scheduled_offgrid"])
                if now_row is not None
                else 0,
                "expected_daily_offgrid_now": float(
                    now_row["expected_daily_offgrid_hours"]
                )
                if now_row is not None
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


@register_feature_group("site_static", requires=("site_static",))
def _site_static_features(
    labels: pd.DataFrame, site_static: pd.DataFrame
) -> pd.DataFrame:
    """Static site config (manufacturer, load, install month, n_cells).

    Battery age is computed at each label's ref_time_h, so this group works
    for both AC_MAINS_FAIL events and lifecycle-end events.
    """
    cols = [
        "site_id",
        "load_A",
        "n_cells",
        "install_month",
        "region",
        "manufacturer",
        "nominal_capacity_ah",
        "charger_misconfigured",
    ]
    avail = [c for c in cols if c in site_static.columns]
    static = site_static[avail].copy()

    out = labels.merge(static, on="site_id", how="left")
    if "install_month" in out.columns:
        out["battery_age_months_at_ref"] = (out["ref_time_h"] / (30 * 24)) - out[
            "install_month"
        ]
        out["battery_age_months_at_ref"] = out["battery_age_months_at_ref"].clip(
            lower=0
        )

    for col in ["region", "manufacturer"]:
        if col in out.columns:
            sorted_categories = sorted(out[col].dropna().unique())
            cat = pd.Categorical(out[col], categories=sorted_categories, ordered=True)
            out[f"{col}_encoded"] = cat.codes
            out = out.drop(columns=[col])

    keep = [
        c for c in out.columns if c not in ("site_id", "ref_time_h") and c != "_row_id"
    ]
    return out[["_row_id"] + keep]
