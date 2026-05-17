from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# Save the style as a dictionary instead of applying it globally
PUBLICATION_STYLE = {
    "font.family":       "serif",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.fontsize":   13,
    "figure.dpi":        300,
    "lines.linewidth":   1.8,
    "grid.alpha":        0.3,
}

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
        """Record one simulation step."""
        x = float(ugv.state[0, 0])
        y = float(ugv.state[1, 0])
        t = len(self._records) * self.dt

        if self._prev_x is not None:
            step_dist = math.hypot(x - self._prev_x, y - self._prev_y)
            speed     = step_dist / self.dt
        else:
            step_dist = 0.0
            speed     = 0.0

        self._prev_x, self._prev_y = x, y
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

    # ── Summary & Export ──────────────────────────────────────────────────────

    def _compute_kinematics(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculates acceleration, jerk, and mean squared jerk (MSJ)."""
        r = self._records
        if len(r) < 2:
            return np.array([0.0]), np.array([0.0]), 0.0

        t = self._t()
        speeds = np.array([s.speed for s in r])
        dt = np.mean(np.diff(t))

        accel = np.diff(speeds, prepend=speeds[0]) / dt
        jerk = np.diff(accel, prepend=accel[0]) / dt
        msj = float(np.mean(jerk**2))

        return accel, jerk, msj

    def summary(self) -> dict:
        """Return a flat dict of key metrics for comparison tables and specs."""
        if not self._records:
            return {}
        r = self._records

        path_length    = sum(s.step_dist for s in r)
        energy_consumed= r[0].battery - r[-1].battery
        uncertainty_int= sum(s.step_dist * (0.02 if s.in_coverage else 2.0) for s in r)
        pct_covered    = 100 * sum(s.in_coverage for s in r) / len(r)
        mean_speed     = float(np.mean([s.speed for s in r]))
        replan_battery = r[-1].battery
        
        _, _, msj = self._compute_kinematics()

        return {
            "ugv_id":                 self.ugv_id,
            "label":                  self.label,
            "total_time_s":           round(r[-1].t, 1),
            "total_steps":            len(r),
            "path_length_m":          round(path_length, 3),
            "mean_speed_m_s":         round(mean_speed, 3),
            "mean_squared_jerk":      round(msj, 4),
            "energy_consumed_pct":    round(energy_consumed, 2),
            "final_battery_pct":      round(replan_battery, 1),
            "uncertainty_integral":   round(uncertainty_int, 3),
            "pct_steps_in_coverage":  round(pct_covered, 1),
        }

    def save_specifications(self, save_dir: str = ".") -> None:
        """Dumps the run specifications into a JSON file."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.ugv_id}_run_specifications.json")

        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=4)
        print(f"[MetricsLogger] Run specifications saved to: {path}")

    def export_time_series(self, save_dir: str = ".") -> None:
        """Exports time, speed, and acceleration arrays for later multi-run comparison."""
        os.makedirs(save_dir, exist_ok=True)
        
        t = self._t().tolist()
        speeds = [s.speed for s in self._records]
        accel, _, _ = self._compute_kinematics()
        
        data = {
            "ugv_id": self.ugv_id,
            "label": self.label,
            "t": t,
            "speed": speeds,
            "accel": accel.tolist()
        }
        
        path = os.path.join(save_dir, f"{self.ugv_id}_timeseries.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[MetricsLogger] Time series data saved to: {path}")

    # ── Thesis figure generation ──────────────────────────────────────────────

    def plot_figures(
        self,
        save_dir:  str  = ".",
        show:      bool = True,
        file_fmt:  str  = "pdf",
    ) -> None:
        """Generate and save all individual thesis figures and specifications."""
        os.makedirs(save_dir, exist_ok=True)

        self.save_specifications(save_dir)
        self.export_time_series(save_dir)  # Automatically export the arrays

        with plt.rc_context(PUBLICATION_STYLE):
            figs_and_names = [
                (self._fig_distance(),            "distance"),
                (self._fig_velocity(),            "velocity"),
                (self._fig_acceleration(),        "acceleration"),
                (self._fig_battery(),             "battery"),
                (self._fig_energy(),              "energy"),
                (self._fig_step_uncertainty(),    "step_uncertainty"),
                (self._fig_uncertainty_integral(),"uncertainty_integral"),
                (self._fig_coverage(),            "coverage"),
                (self._fig_cost_breakdown(),      "cost_breakdown"),
                (self._fig_trajectory(),          "trajectory"),
            ]
            
            for fig, name in figs_and_names:
                path = os.path.join(save_dir, f"{self.ugv_id}_{name}.{file_fmt}")
                fig.savefig(path, bbox_inches="tight")
                print(f"[MetricsLogger] Saved: {path}")

            if show:
                plt.show()

    # ── Individual Figure Helpers ─────────────────────────────────────────────

    def _t(self) -> np.ndarray:
        return np.array([s.t for s in self._records])

    def _fig_distance(self) -> plt.Figure:
        t = self._t()
        dist = np.cumsum([s.step_dist for s in self._records])

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle("Cumulative Distance", fontweight='bold')
        ax.plot(t, dist, color='steelblue')
        ax.set_ylabel("Distance [m]")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"Total Distance: {dist[-1]:.2f} m", loc='right', fontsize=10)
        ax.grid(True)
        return fig

    def _fig_velocity(self) -> plt.Figure:
        t = self._t()
        speeds = np.array([s.speed for s in self._records])

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle("Speed Profile", fontweight='bold')
        ax.plot(t, speeds, color='darkorange', alpha=0.8)
        
        if len(speeds) > 10:
            kernel = np.ones(10) / 10
            smooth = np.convolve(speeds, kernel, mode='same')
            ax.plot(t, smooth, color='red', linewidth=2, label="10-step avg")
            ax.legend()
            
        ax.set_ylabel("Speed [m/s]")
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        return fig

    def _fig_acceleration(self) -> plt.Figure:
        t = self._t()
        accel, _, msj = self._compute_kinematics()

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle("Acceleration Profile", fontweight='bold')
        ax.plot(t, accel, color='firebrick')
        ax.set_ylabel("Accel [m/s²]")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"Mean Squared Jerk (MSJ): {msj:.2f} m²/s⁶", loc='right', fontsize=10)
        ax.grid(True)
        return fig

    def _fig_battery(self) -> plt.Figure:
        t = self._t()
        battery = np.array([s.battery for s in self._records])

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle(f"{self.label} — Battery SoC", fontweight='bold')
        ax.plot(t, battery, color='forestgreen')
        ax.axhline(15, color='red', linestyle='--', linewidth=1, label="Safety reserve (15%)")
        ax.set_ylabel("Battery SoC [%]")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True)
        return fig

    def _fig_energy(self) -> plt.Figure:
        t = self._t()
        energy = self._records[0].battery - np.array([s.battery for s in self._records])

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle("Energy Consumed", fontweight='bold')
        ax.fill_between(t, energy, alpha=0.4, color='orangered')
        ax.plot(t, energy, color='orangered')
        ax.set_ylabel("Energy consumed [%]")
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        return fig

    def _fig_step_uncertainty(self) -> plt.Figure:
        t = self._t()
        sig = np.array([s.step_dist * (0.02 if s.in_coverage else 2.0) for s in self._records])

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle("Per-Step Uncertainty", fontweight='bold')
        ax.fill_between(t, sig, alpha=0.4, color='mediumpurple')
        ax.plot(t, sig, color='mediumpurple', linewidth=1)
        ax.axhline(0.02, color='green', linestyle='--', linewidth=1, label="Covered: σ²=0.02")
        ax.axhline(2.0,  color='red',   linestyle='--', linewidth=1, label="Uncovered: σ²=2.0")
        ax.set_ylabel("Step uncert. [m²]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)
        ax.grid(True)
        return fig

    def _fig_uncertainty_integral(self) -> plt.Figure:
        t = self._t()
        sig = np.array([s.step_dist * (0.02 if s.in_coverage else 2.0) for s in self._records])
        unc_integral = np.cumsum(sig)

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle("Uncertainty Integral", fontweight='bold')
        ax.plot(t, unc_integral, color='indigo')
        ax.set_ylabel("Integral [m²·m]")
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        return fig

    def _fig_coverage(self) -> plt.Figure:
        t = self._t()
        in_cov = np.array([float(s.in_coverage) for s in self._records])
        rolling = np.convolve(in_cov, np.ones(20)/20, mode='same') * 100

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle("UAV Coverage Over Time", fontweight='bold')
        ax.fill_between(t, rolling, alpha=0.3, color='deepskyblue')
        ax.plot(t, rolling, color='deepskyblue', label="20-step rolling avg [%]")
        ax.set_ylabel("Coverage [%]")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True)
        return fig

    def _fig_cost_breakdown(self) -> plt.Figure:
        r = self._records
        if not any(s.c_distance for s in r):
            fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
            ax.text(0.5, 0.5, "Cost terms not recorded", ha='center', va='center')
            return fig

        terms = {
            "Distance": sum(s.c_distance    for s in r),
            "Energy":   sum(s.c_energy      for s in r),
            "Time":     sum(s.c_time        for s in r),
            "Uncert.":  sum(s.c_uncertainty for s in r),
            "Risk":     sum(s.c_risk        for s in r),
        }
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        fig.suptitle("Cost Function Breakdown", fontweight='bold')
        ax.bar(terms.keys(), terms.values(), color=colors, edgecolor='k', linewidth=0.6)
        ax.set_ylabel("Cumulative cost contribution")
        ax.grid(True, axis='y')
        return fig

    def _fig_trajectory(self) -> plt.Figure:
        r = self._records
        xs = np.array([s.x for s in r])
        ys = np.array([s.y for s in r])
        speeds = np.array([s.speed for s in r])

        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
        fig.suptitle("Trajectory", fontweight='bold')

        sc = ax.scatter(xs, ys, c=speeds, cmap='plasma', s=4, zorder=3, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Speed [m/s]", fraction=0.046, pad=0.04)

        ax.plot(xs[0], ys[0], 'go', markersize=10, label="Start", zorder=5)
        ax.plot(xs[-1], ys[-1], 'rs', markersize=10, label="End",  zorder=5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True)
        return fig


# ── Comparison figure (Split into separate plots) ─────────────────────────────

def plot_comparison(
    logger_a: MetricsLogger,
    logger_b: MetricsLogger,
    save_dir: str  = ".",
    show:     bool = True,
    file_fmt: str  = "pdf",
) -> None:
    """Side-by-side comparison saved as individual figures instead of a grid."""
    os.makedirs(save_dir, exist_ok=True)

    def _t(lg): return np.array([s.t for s in lg._records])
    def _arr(lg, attr): return np.array([getattr(s, attr) for s in lg._records])

    figs_to_show = []

    with plt.rc_context(PUBLICATION_STYLE):
        
        # 1. Battery Comparison
        fig_batt, ax_batt = plt.subplots(figsize=(6, 4), layout="constrained")
        fig_batt.suptitle("Comparison: Battery State of Charge", fontweight='bold')
        for lg, color in ((logger_a, 'forestgreen'), (logger_b, 'firebrick')):
            ax_batt.plot(_t(lg), _arr(lg, 'battery'), color=color, label=lg.label)
        ax_batt.axhline(15, color='k', linestyle='--', linewidth=0.8, label="Reserve")
        ax_batt.set_ylabel("[%]")
        ax_batt.set_xlabel("Time [s]")
        ax_batt.legend()
        ax_batt.grid(True)
        figs_to_show.append((fig_batt, "comp_battery"))

        # 2. Distance Comparison
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4), layout="constrained")
        fig_dist.suptitle("Comparison: Cumulative Path Length", fontweight='bold')
        for lg, color in ((logger_a, 'steelblue'), (logger_b, 'darkorange')):
            ax_dist.plot(_t(lg), np.cumsum(_arr(lg, 'step_dist')), color=color, label=lg.label)
        ax_dist.set_ylabel("[m]")
        ax_dist.set_xlabel("Time [s]")
        ax_dist.legend()
        ax_dist.grid(True)
        figs_to_show.append((fig_dist, "comp_distance"))

        # 3. Uncertainty Comparison
        fig_unc, ax_unc = plt.subplots(figsize=(6, 4), layout="constrained")
        fig_unc.suptitle("Comparison: Uncertainty Integral", fontweight='bold')
        for lg, color in ((logger_a, 'mediumpurple'), (logger_b, 'tomato')):
            sig = _arr(lg, 'step_dist') * np.where([s.in_coverage for s in lg._records], 0.02, 2.0)
            ax_unc.plot(_t(lg), np.cumsum(sig), color=color, label=lg.label)
        ax_unc.set_ylabel("[m²·m]")
        ax_unc.set_xlabel("Time [s]")
        ax_unc.legend()
        ax_unc.grid(True)
        figs_to_show.append((fig_unc, "comp_uncertainty"))

        # 4. Summary Bar Chart
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4), layout="constrained")
        fig_bar.suptitle("Comparison: Key Metrics Summary", fontweight='bold')
        keys   = ["path_length_m", "energy_consumed_pct", "uncertainty_integral", "final_battery_pct"]
        labels = ["Path [m]", "Energy [%]", "Uncert. Int.", "Batt. left [%]"]
        sa, sb = logger_a.summary(), logger_b.summary()

        x = np.arange(len(keys))
        width = 0.35
        ax_bar.bar(x - width/2, [sa.get(k, 0) for k in keys], width, label=logger_a.label, color='steelblue', edgecolor='k', lw=0.6)
        ax_bar.bar(x + width/2, [sb.get(k, 0) for k in keys], width, label=logger_b.label, color='tomato', edgecolor='k', lw=0.6)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, rotation=15, ha='right')
        ax_bar.legend()
        ax_bar.grid(True, axis='y')
        figs_to_show.append((fig_bar, "comp_metrics"))

        # Save all comparison figures
        for fig, name in figs_to_show:
            path = os.path.join(save_dir, f"{name}_{logger_a.ugv_id}_vs_{logger_b.ugv_id}.{file_fmt}")
            fig.savefig(path, bbox_inches="tight")
            print(f"[MetricsLogger] Saved comparison: {path}")

        if show:
            plt.show()

# ── Multi-Run Offline Plotting ───────────────────────────────────────────────

def plot_combined_timeseries(
    filepaths: list[str],
    save_dir: str = ".",
    file_fmt: str = "pdf",
    show: bool = True
) -> None:
    """
    Loads multiple exported timeseries JSON files and plots Speed and Acceleration
    together on a single figure for easy visual comparison.
    
    Parameters
    ----------
    filepaths : List of file paths to the '*_timeseries.json' files.
    save_dir  : Directory to save the resulting combined plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    data_runs = []
    for path in filepaths:
        with open(path, 'r') as f:
            data_runs.append(json.load(f))
            
    if not data_runs:
        print("[MetricsLogger] No filepaths provided for multi-run comparison.")
        return

    with plt.rc_context(PUBLICATION_STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")
        fig.suptitle("Combined Kinematics Comparison", fontweight='bold')

        # We can cycle through some standard colors if you have many lines
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, run in enumerate(data_runs):
            t = run["t"]
            label = run.get("label", run.get("ugv_id", f"Run {i+1}"))
            c = colors[i % len(colors)]

            # Speed Plot
            ax1.plot(t, run["speed"], label=label, color=c, alpha=0.85)
            # Accel Plot
            ax2.plot(t, run["accel"], label=label, color=c, alpha=0.85)

        ax1.set_ylabel("Speed [m/s]")
        ax1.grid(True)
        ax1.legend()

        ax2.set_ylabel("Accel [m/s²]")
        ax2.set_xlabel("Time [s]")
        ax2.grid(True)
        ax2.legend()

        out_path = os.path.join(save_dir, f"combined_kinematics_comparison.{file_fmt}")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[MetricsLogger] Combined kinematics plot saved to: {out_path}")

        if show:
            plt.show()