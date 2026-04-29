

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Publication style
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi":    150,
    "lines.linewidth": 1.8,
})


@dataclass
class _StepRecord:
    """One timestep snapshot for a single UGV."""
    t:            float   # simulation time [s]
    x:            float
    y:            float
    speed:        float   # [m/s]
    battery:      float   # [%]
    in_coverage:  bool
    step_cost:    float   # full cell cost this step
    step_dist:    float   # arc-length this step [m]
    # Cost term breakdown
    c_distance:   float = 0.0
    c_energy:     float = 0.0
    c_time:       float = 0.0
    c_uncertainty:float = 0.0
    c_risk:       float = 0.0


class MetricsLogger:
    """
    Records per-step metrics for one UGV and generates thesis figures.

    Parameters
    ----------
    ugv_id  : string identifier used in plot titles and legends.
    dt      : simulation step time [s].
    label   : short label for comparison plots (e.g. "UAV+UGV", "UGV only").
    """

    def __init__(
        self,
        ugv_id: str = "ugv1",
        dt:     float = 0.1,
        label:  str | None = None,
    ) -> None:
        self.ugv_id  = ugv_id
        self.dt      = dt
        self.label   = label or ugv_id
        self._records: list[_StepRecord] = []
        self._prev_x: float | None = None
        self._prev_y: float | None = None

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        ugv,
        in_coverage:   bool  = False,
        step_cost:     float = 0.0,
        cost_terms:    tuple | None = None,
    ) -> None:
        """
        Record one simulation step.

        Parameters
        ----------
        ugv          : UGVTwin instance.
        in_coverage  : True if the UGV's current cell is inside UAV footprint.
        step_cost    : total cell cost returned by cell_cost().
        cost_terms   : optional (c_dist, c_energy, c_time, c_uncert, c_risk).
        """
        x      = float(ugv.state[0, 0])
        y      = float(ugv.state[1, 0])
        t      = len(self._records) * self.dt

        # Instantaneous speed from displacement
        if self._prev_x is not None:
            step_dist = math.hypot(x - self._prev_x, y - self._prev_y)
            speed     = step_dist / self.dt
        else:
            step_dist = 0.0
            speed     = 0.0

        self._prev_x, self._prev_y = x, y

        # Battery (UGVTwin attribute)
        battery = float(getattr(ugv, 'battery_status', 100.0))

        cd, ce, ct, cu, cr = cost_terms if cost_terms else (0,0,0,0,0)

        self._records.append(_StepRecord(
            t=t, x=x, y=y,
            speed=speed, battery=battery,
            in_coverage=in_coverage,
            step_cost=step_cost,
            step_dist=step_dist,
            c_distance=cd, c_energy=ce, c_time=ct,
            c_uncertainty=cu, c_risk=cr,
        ))

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a flat dict of key metrics for comparison tables."""
        if not self._records:
            return {}
        r = self._records
        path_length    = sum(s.step_dist for s in r)
        energy_consumed= r[0].battery - r[-1].battery
        uncertainty_int= sum(s.step_dist * (
            0.02 if s.in_coverage else 2.0) for s in r)
        pct_covered    = 100 * sum(s.in_coverage for s in r) / len(r)
        mean_speed     = float(np.mean([s.speed for s in r]))
        replan_battery = r[-1].battery

        return {
            "path_length_m":          round(path_length, 3),
            "energy_consumed_pct":    round(energy_consumed, 2),
            "uncertainty_integral":   round(uncertainty_int, 3),
            "pct_steps_in_coverage":  round(pct_covered, 1),
            "mean_speed_m_s":         round(mean_speed, 3),
            "final_battery_pct":      round(replan_battery, 1),
            "total_steps":            len(r),
            "total_time_s":           round(r[-1].t, 1),
        }

    # ── Thesis figure generation ──────────────────────────────────────────────

    def plot_figures(
        self,
        save_dir:  str  = ".",
        show:      bool = True,
        file_fmt:  str  = "pdf",
    ) -> None:
        """
        Generate and save all individual thesis figures.

        Parameters
        ----------
        save_dir : directory to write figure files.
        show     : whether to call plt.show() after generating.
        file_fmt : 'pdf' for LaTeX, 'png' for quick preview.
        """
        os.makedirs(save_dir, exist_ok=True)

        figs = [
            self._fig_distance_velocity(),
            self._fig_battery_energy(),
            self._fig_uncertainty(),
            #self._fig_coverage(),
            self._fig_cost_breakdown(),
            self._fig_trajectory(),
        ]
        names = [
            "distance_velocity",
            "battery_energy",
            "uncertainty",
            "coverage",
            "cost_breakdown",
            "trajectory",
        ]
        for fig, name in zip(figs, names):
            path = os.path.join(save_dir, f"{self.ugv_id}_{name}.{file_fmt}")
            fig.savefig(path, bbox_inches="tight")
            print(f"[MetricsLogger] Saved: {path}")

        if show:
            plt.show()

    # ── Figure helpers ────────────────────────────────────────────────────────

    def _t(self) -> np.ndarray:
        return np.array([s.t for s in self._records])

    def _fig_distance_velocity(self) -> plt.Figure:
        """Figure 1: cumulative distance + instantaneous speed vs time."""
        r   = self._records
        t   = self._t()
        dist_cumulative = np.cumsum([s.step_dist for s in r])
        speeds          = np.array([s.speed for s in r])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        fig.suptitle(f"{self.label} — Distance & Speed", fontweight='bold')

        ax1.plot(t, dist_cumulative, color='steelblue')
        ax1.set_ylabel("Cumulative distance [m]")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, speeds, color='darkorange', alpha=0.8)
        # Smoothed overlay
        if len(speeds) > 10:
            kernel = np.ones(10) / 10
            smooth = np.convolve(speeds, kernel, mode='same')
            ax2.plot(t, smooth, color='red', linewidth=2, label="10-step avg")
            ax2.legend()
        ax2.set_ylabel("Speed [m/s]")
        ax2.set_xlabel("Time [s]")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _fig_battery_energy(self) -> plt.Figure:
        """Figure 2: battery state of charge + cumulative energy vs time."""
        r       = self._records
        t       = self._t()
        battery = np.array([s.battery for s in r])
        energy  = r[0].battery - battery   # consumed so far

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        fig.suptitle(f"{self.label} — Battery & Energy", fontweight='bold')

        ax1.plot(t, battery, color='forestgreen')
        ax1.axhline(15, color='red', linestyle='--', linewidth=1,
                    label="Safety reserve (15%)")
        ax1.set_ylabel("Battery SoC [%]")
        ax1.set_ylim(0, 105)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(t, energy, alpha=0.4, color='orangered')
        ax2.plot(t, energy, color='orangered')
        ax2.set_ylabel("Cumulative energy consumed [%]")
        ax2.set_xlabel("Time [s]")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _fig_uncertainty(self) -> plt.Figure:
        """Figure 3: position uncertainty proxy vs time (step × σ²)."""
        r   = self._records
        t   = self._t()
        sig = np.array([
            s.step_dist * (0.02 if s.in_coverage else 2.0) for s in r
        ])
        unc_integral = np.cumsum(sig)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        fig.suptitle(f"{self.label} — Position Uncertainty", fontweight='bold')

        ax1.fill_between(t, sig, alpha=0.4, color='mediumpurple')
        ax1.plot(t, sig, color='mediumpurple', linewidth=1)
        ax1.axhline(0.02, color='green', linestyle='--', linewidth=1,
                    label="Covered: σ²=0.02")
        ax1.axhline(2.0,  color='red',   linestyle='--', linewidth=1,
                    label="Uncovered: σ²=2.0")
        ax1.set_ylabel("Per-step uncertainty [m²]")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, unc_integral, color='indigo')
        ax2.set_ylabel("Uncertainty integral [m²·m]")
        ax2.set_xlabel("Time [s]")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _fig_coverage(self) -> plt.Figure:
        """Figure 4: % of steps spent inside UAV coverage."""
        r      = self._records
        t      = self._t()
        in_cov = np.array([float(s.in_coverage) for s in r])
        rolling= np.convolve(in_cov, np.ones(20)/20, mode='same') * 100

        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.suptitle(f"{self.label} — UAV Coverage Over Time", fontweight='bold')
        ax.fill_between(t, rolling, alpha=0.3, color='deepskyblue')
        ax.plot(t, rolling, color='deepskyblue', label="20-step rolling avg [%]")
        ax.set_ylabel("Steps in UAV coverage [%]")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _fig_cost_breakdown(self) -> plt.Figure:
        """Figure 5: stacked bar chart of cost term contributions."""
        r = self._records
        if not any(s.c_distance for s in r):
            # Cost terms not recorded — skip
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.text(0.5, 0.5, "Cost terms not recorded\n(pass cost_terms to .record())",
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        terms = {
            "Distance":    sum(s.c_distance    for s in r),
            "Energy":      sum(s.c_energy      for s in r),
            "Time":        sum(s.c_time        for s in r),
            "Uncertainty": sum(s.c_uncertainty for s in r),
            "Risk":        sum(s.c_risk        for s in r),
        }
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle(f"{self.label} — Cost Function Breakdown", fontweight='bold')
        ax.bar(terms.keys(), terms.values(), color=colors, edgecolor='k', linewidth=0.6)
        ax.set_ylabel("Cumulative cost contribution")
        ax.set_xlabel("Cost term")
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        return fig

    def _fig_trajectory(self) -> plt.Figure:
        """Figure 6: XY trajectory coloured by speed."""
        r = self._records
        xs = np.array([s.x for s in r])
        ys = np.array([s.y for s in r])
        speeds = np.array([s.speed for s in r])

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle(f"{self.label} — Trajectory (coloured by speed)",
                     fontweight='bold')

        sc = ax.scatter(xs, ys, c=speeds, cmap='plasma',
                        s=4, zorder=3, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Speed [m/s]", fraction=0.046, pad=0.04)

        # Start / end markers
        ax.plot(xs[0], ys[0], 'go', markersize=10, label="Start", zorder=5)
        ax.plot(xs[-1], ys[-1], 'rs', markersize=10, label="End",  zorder=5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ── Comparison figure (two loggers) ──────────────────────────────────────────

def plot_comparison(
    logger_a: MetricsLogger,
    logger_b: MetricsLogger,
    save_dir: str  = ".",
    show:     bool = True,
    file_fmt: str  = "pdf",
) -> None:
    """
    Side-by-side comparison of two runs (e.g. UAV-enabled vs ground-only).
    Generates a single figure with 4 paired subplots.

    Parameters
    ----------
    logger_a : first run (typically UAV-enabled).
    logger_b : second run (typically ground-only baseline).
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        f"Comparison:  {logger_a.label}  vs  {logger_b.label}",
        fontsize=13, fontweight='bold',
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    def _t(lg):  return np.array([s.t for s in lg._records])
    def _arr(lg, attr): return np.array([getattr(s, attr) for s in lg._records])

    # ── Battery ───────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for lg, color in ((logger_a, 'forestgreen'), (logger_b, 'firebrick')):
        ax.plot(_t(lg), _arr(lg, 'battery'), color=color, label=lg.label)
    ax.axhline(15, color='k', linestyle='--', linewidth=0.8, label="Reserve")
    ax.set_title("Battery State of Charge")
    ax.set_ylabel("%"); ax.set_xlabel("t [s]")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Cumulative distance ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for lg, color in ((logger_a, 'steelblue'), (logger_b, 'darkorange')):
        dist = np.cumsum(_arr(lg, 'step_dist'))
        ax.plot(_t(lg), dist, color=color, label=lg.label)
    ax.set_title("Cumulative Path Length")
    ax.set_ylabel("m"); ax.set_xlabel("t [s]")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Uncertainty integral ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for lg, color in ((logger_a, 'mediumpurple'), (logger_b, 'tomato')):
        sig  = _arr(lg, 'step_dist') * np.where(
            [s.in_coverage for s in lg._records], 0.02, 2.0
        )
        ax.plot(_t(lg), np.cumsum(sig), color=color, label=lg.label)
    ax.set_title("Uncertainty Integral  (↓ better)")
    ax.set_ylabel("m²·m"); ax.set_xlabel("t [s]")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Summary bar chart ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    keys   = ["path_length_m", "energy_consumed_pct",
               "uncertainty_integral", "final_battery_pct"]
    labels = ["Path [m]", "Energy [%]", "Uncert. integral", "Battery left [%]"]
    sa, sb = logger_a.summary(), logger_b.summary()

    x     = np.arange(len(keys))
    width = 0.35
    ax.bar(x - width/2, [sa.get(k, 0) for k in keys],
           width, label=logger_a.label, color='steelblue', edgecolor='k', lw=0.6)
    ax.bar(x + width/2, [sb.get(k, 0) for k in keys],
           width, label=logger_b.label, color='tomato', edgecolor='k', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_title("Key Metrics Comparison")
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    path = os.path.join(save_dir, f"comparison_{logger_a.ugv_id}.{file_fmt}")
    fig.savefig(path, bbox_inches="tight")
    print(f"[MetricsLogger] Comparison saved: {path}")

    if show:
        plt.show()
