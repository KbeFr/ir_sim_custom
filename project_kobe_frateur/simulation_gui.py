# simulation_gui.py
"""
PyQt6 GUI for IRSim + OverArchingTwin.

Design pattern: Adapter + Observer.
  - Embeds env._env_plot.fig via FigureCanvasQTAgg  (zero ir-sim changes)
  - QTimer replaces plt.pause() entirely
  - Visibility toggling via filtered `objects` list passed to env._env_plot.step()

Usage (from custom_world.py):
    from simulation_gui import launch
    launch(env=env, adt=adt, ugv_twins=ugv_twins, controllers=controllers,
           uav_twins=uav_twins, max_steps=MAX_STEPS, step_ms=100,
           perception_mode=PERCEPTION_MODE)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("QtAgg")  # Must be set before any plt/irsim import
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from overarchingTwin.overarching_twin import PerceptionMode


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QGroupBox, QScrollArea,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget, QTextEdit,
    QFormLayout,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from overarchingTwin.mission_planner import Mission, MissionType


class SimulationGUI(QMainWindow):
    """Main window: matplotlib canvas (left) + tabbed control panel (right)."""

    def __init__(self, env, adt, ugv_twins, controllers, uav_twins,
                 max_steps=800, step_ms=100, perception_mode="ugv"):
        super().__init__()

        # ── Core references ────────────────────────────────────────────────
        self.env             = env
        self.adt             = adt
        self.ugv_twins       = ugv_twins
        self.controllers     = controllers
        self.uav_twins       = uav_twins
        self.max_steps       = max_steps
        self.perception_mode = perception_mode

        # ── State ──────────────────────────────────────────────────────────
        self._step            = 0
        self._running         = False
        self._use_global_plan = False
        self._visible_robots  = {r.id: True for r in env.robot_list}
        self._show_sensors    = True
        self._overlay_active  = False        # imshow overlay on sim canvas
        self._overlay_im      = None         # AxesImage handle for overlay
        self._map_fig         = Figure(figsize=(6, 6), tight_layout=True)
        self._map_ax          = self._map_fig.add_subplot(111)

        self.setWindowTitle("HDT Simulation — Control Panel")
        self.resize(1440, 860)

        self._build_ui()

        self.timer = QTimer()
        self.timer.setInterval(step_ms)
        self.timer.timeout.connect(self._sim_step)

    # ══════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(self._build_canvas_panel(), stretch=1)
        layout.addWidget(self._build_control_panel())

    # ── Canvas panel (left) ────────────────────────────────────────────────

    def _build_canvas_panel(self):
        """Left area: tabbed between the live sim canvas and the map viewer."""
        panel     = QWidget()
        vbox      = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._canvas_tabs = QTabWidget()

        # ── Tab 0: Simulation ──────────────────────────────────────────────
        sim_widget  = QWidget()
        sim_vbox    = QVBoxLayout(sim_widget)
        sim_vbox.setContentsMargins(0, 0, 0, 0)
        self.canvas = FigureCanvasQTAgg(self.env._env_plot.fig)
        sim_toolbar = NavigationToolbar2QT(self.canvas, sim_widget)
        sim_vbox.addWidget(sim_toolbar)
        sim_vbox.addWidget(self.canvas, stretch=1)

        # ── Tab 1: Map Viewer ──────────────────────────────────────────────
        map_widget  = QWidget()
        map_vbox    = QVBoxLayout(map_widget)
        map_vbox.setContentsMargins(0, 0, 0, 0)
        self.map_canvas  = FigureCanvasQTAgg(self._map_fig)
        map_toolbar      = NavigationToolbar2QT(self.map_canvas, map_widget)
        map_vbox.addWidget(map_toolbar)
        map_vbox.addWidget(self.map_canvas, stretch=1)

        self._canvas_tabs.addTab(sim_widget, "🌐  Simulation")
        self._canvas_tabs.addTab(map_widget, "🗺  Map View")

        vbox.addWidget(self._canvas_tabs, stretch=1)
        vbox.addLayout(self._build_playback_bar())
        return panel

    def _build_playback_bar(self):
        bar = QHBoxLayout()

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setFixedWidth(90)
        self.btn_play.clicked.connect(self._toggle_play)

        self.btn_step_once = QPushButton("⏭  Step")
        self.btn_step_once.setFixedWidth(80)
        self.btn_step_once.clicked.connect(self._single_step)

        self.lbl_step = QLabel(f"Step: 0 / {self.max_steps}")
        self.lbl_step.setMinimumWidth(130)

        spd_lbl        = QLabel("Speed:")
        self.sld_speed = QSlider(Qt.Orientation.Horizontal)
        self.sld_speed.setRange(1, 30)
        self.sld_speed.setValue(10)
        self.sld_speed.setMaximumWidth(140)
        self.sld_speed.valueChanged.connect(self._on_speed_changed)
        self.lbl_hz = QLabel("10 Hz")
        self.lbl_hz.setMinimumWidth(45)

        bar.addWidget(self.btn_play)
        bar.addWidget(self.btn_step_once)
        bar.addWidget(self.lbl_step)
        bar.addStretch()
        bar.addWidget(spd_lbl)
        bar.addWidget(self.sld_speed)
        bar.addWidget(self.lbl_hz)
        return bar

    # ── Control panel (right) ──────────────────────────────────────────────

    def _build_control_panel(self):
        tabs = QTabWidget()
        tabs.setFixedWidth(310)
        tabs.addTab(self._build_visibility_tab(), "👁  Visibility")
        tabs.addTab(self._build_robots_tab(),     "🤖  Robots")
        tabs.addTab(self._build_mission_tab(),    "🎯  Missions")
        tabs.addTab(self._build_maps_tab(),       "🗺  Maps")
        tabs.addTab(self._build_log_tab(),        "📋  Log")
        return tabs

    # ── Visibility tab ─────────────────────────────────────────────────────

    def _build_visibility_tab(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        # Perception mode
        grp_perc   = QGroupBox("Perception Mode")
        perc_vbox  = QVBoxLayout(grp_perc)
        self.combo_perc = QComboBox()
        self.combo_perc.addItems(PerceptionMode.get_names())
        self.combo_perc.setCurrentText(self.perception_mode)
        self.combo_perc.currentTextChanged.connect(self._on_perc_changed)
        perc_vbox.addWidget(self.combo_perc)
        layout.addWidget(grp_perc)

        # Sensor layer toggle
        grp_sens  = QGroupBox("Sensor Layers")
        sens_vbox = QVBoxLayout(grp_sens)
        self.chk_sensors = QCheckBox("Show sensor footprints")
        self.chk_sensors.setChecked(True)
        self.chk_sensors.toggled.connect(self._on_sensor_toggle)
        sens_vbox.addWidget(self.chk_sensors)
        layout.addWidget(grp_sens)

        # Per-robot checkboxes
        grp_robots  = QGroupBox("Individual Robots")
        robots_vbox = QVBoxLayout(grp_robots)

        scroll   = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(280)
        inner    = QWidget()
        in_vbox  = QVBoxLayout(inner)
        in_vbox.setSpacing(2)
        self._robots_inner_layout = in_vbox   # kept for dynamic add/remove

        self._robot_checks = {}
        for robot in self.env.robot_list:
            lbl = f"{type(robot).__name__}  [id={robot.id}]"
            chk = QCheckBox(lbl)
            chk.setChecked(True)
            chk.toggled.connect(
                lambda checked, rid=robot.id: self._on_robot_toggle(rid, checked)
            )
            self._robot_checks[robot.id] = chk
            in_vbox.addWidget(chk)

        in_vbox.addStretch()
        scroll.setWidget(inner)
        robots_vbox.addWidget(scroll)

        btn_row = QHBoxLayout()
        for label, state in [("All", True), ("None", False)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, s=state: self._set_all_robots(s))
            btn_row.addWidget(btn)
        robots_vbox.addLayout(btn_row)
        layout.addWidget(grp_robots)
        layout.addStretch()
        return w

    # ── Robots tab (spawn / delete) ────────────────────────────────────────

    def _build_robots_tab(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        # ── Spawn ──────────────────────────────────────────────────────────
        grp_spawn = QGroupBox("Spawn Robot")
        form      = QFormLayout(grp_spawn)

        self.spawn_name_edit = QLineEdit("robot_new")

        self.spawn_type_combo = QComboBox()
        self.spawn_type_combo.addItems(["ugv", "uav", "robot"])

        self.spawn_kin_combo = QComboBox()
        self.spawn_kin_combo.addItems(["diff", "omni", "acker"])

        self.spawn_x_spin = QDoubleSpinBox()
        self.spawn_x_spin.setRange(-500, 500); self.spawn_x_spin.setValue(5.0)

        self.spawn_y_spin = QDoubleSpinBox()
        self.spawn_y_spin.setRange(-500, 500); self.spawn_y_spin.setValue(5.0)

        self.spawn_theta_spin = QDoubleSpinBox()
        self.spawn_theta_spin.setRange(-3.15, 3.15); self.spawn_theta_spin.setValue(0.0)
        self.spawn_theta_spin.setSingleStep(0.1)

        self.spawn_goal_x_spin = QDoubleSpinBox()
        self.spawn_goal_x_spin.setRange(-500, 500); self.spawn_goal_x_spin.setValue(10.0)

        self.spawn_goal_y_spin = QDoubleSpinBox()
        self.spawn_goal_y_spin.setRange(-500, 500); self.spawn_goal_y_spin.setValue(10.0)

        form.addRow("Name:",       self.spawn_name_edit)
        form.addRow("Type:",       self.spawn_type_combo)
        form.addRow("Kinematics:", self.spawn_kin_combo)
        form.addRow("X [m]:",      self.spawn_x_spin)
        form.addRow("Y [m]:",      self.spawn_y_spin)
        form.addRow("θ [rad]:",    self.spawn_theta_spin)
        form.addRow("Goal X:",     self.spawn_goal_x_spin)
        form.addRow("Goal Y:",     self.spawn_goal_y_spin)

        btn_spawn = QPushButton("🚀  Spawn Robot")
        btn_spawn.clicked.connect(self._spawn_robot)

        # ── Delete ─────────────────────────────────────────────────────────
        grp_del   = QGroupBox("Delete Robot")
        del_vbox  = QVBoxLayout(grp_del)

        self.del_combo = QComboBox()
        self._refresh_delete_combo()

        btn_del = QPushButton("🗑  Delete Selected")
        btn_del.clicked.connect(self._delete_robot)

        del_vbox.addWidget(self.del_combo)
        del_vbox.addWidget(btn_del)

        layout.addWidget(grp_spawn)
        layout.addWidget(btn_spawn)
        layout.addWidget(grp_del)
        layout.addStretch()
        return w

    # ── Mission tab ────────────────────────────────────────────────────────

    def _build_mission_tab(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        grp  = QGroupBox("New Mission")
        form = QFormLayout(grp)

        self.mission_id_edit = QLineEdit("mission_1")

        self.mission_type_combo = QComboBox()
        self.mission_type_combo.addItems([m.name for m in MissionType])

        self.goal_x_spin = QDoubleSpinBox()
        self.goal_x_spin.setRange(-500, 500)
        self.goal_x_spin.setValue(10.0)
        self.goal_x_spin.setSingleStep(0.5)

        self.goal_y_spin = QDoubleSpinBox()
        self.goal_y_spin.setRange(-500, 500)
        self.goal_y_spin.setValue(35.0)
        self.goal_y_spin.setSingleStep(0.5)

        self.posture_combo = QComboBox()
        self.posture_combo.addItems(["EXPLORE", "DEFEND", "ATTACK"])

        form.addRow("Mission ID:",  self.mission_id_edit)
        form.addRow("Type:",        self.mission_type_combo)
        form.addRow("Goal X [m]:",  self.goal_x_spin)
        form.addRow("Goal Y [m]:",  self.goal_y_spin)
        form.addRow("Posture:",     self.posture_combo)

        btn_add = QPushButton("➕  Add Mission")
        btn_add.clicked.connect(self._add_mission)

        grp_active = QGroupBox("Active Missions")
        act_vbox   = QVBoxLayout(grp_active)
        self.mission_list_label = QLabel("None")
        self.mission_list_label.setWordWrap(True)
        self.mission_list_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        act_vbox.addWidget(self.mission_list_label)

        layout.addWidget(grp)
        layout.addWidget(btn_add)
        layout.addWidget(grp_active)
        layout.addStretch()
        return w


    # ── Maps tab ───────────────────────────────────────────────────────────

    def _build_maps_tab(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        # ── Layer selector ─────────────────────────────────────────────────
        grp_sel  = QGroupBox("Layer")
        sel_vbox = QVBoxLayout(grp_sel)

        self.map_layer_combo = QComboBox()
        self._rebuild_map_layer_combo()          # populate from current adt state
        sel_vbox.addWidget(self.map_layer_combo)

        btn_refresh_list = QPushButton("🔄  Refresh layer list")
        btn_refresh_list.clicked.connect(self._rebuild_map_layer_combo)
        sel_vbox.addWidget(btn_refresh_list)
        layout.addWidget(grp_sel)

        # ── Colormap ───────────────────────────────────────────────────────
        grp_cmap  = QGroupBox("Colormap")
        cmap_vbox = QVBoxLayout(grp_cmap)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["YlOrRd", "viridis", "plasma", "gray", "RdYlGn_r"])
        cmap_vbox.addWidget(self.cmap_combo)
        layout.addWidget(grp_cmap)

        # ── UGV posture weights (for cost maps) ───────────────────────────
        grp_w   = QGroupBox("Cost map — UGV params")
        w_form  = QFormLayout(grp_w)
        self.cost_mass_spin    = QDoubleSpinBox(); self.cost_mass_spin.setRange(0.1, 200); self.cost_mass_spin.setValue(20.0)
        self.cost_speed_spin   = QDoubleSpinBox(); self.cost_speed_spin.setRange(0.01, 10); self.cost_speed_spin.setValue(1.0)
        self.cost_anc_spin     = QDoubleSpinBox(); self.cost_anc_spin.setRange(0, 100); self.cost_anc_spin.setValue(0.0)
        self.posture_map_combo = QComboBox(); self.posture_map_combo.addItems(["EXPLORE", "DEFEND", "ATTACK"])
        w_form.addRow("Mass [kg]:",    self.cost_mass_spin)
        w_form.addRow("Avg speed:",    self.cost_speed_spin)
        w_form.addRow("Anc. drain:",   self.cost_anc_spin)
        w_form.addRow("Posture:",      self.posture_map_combo)
        layout.addWidget(grp_w)

        # ── Overlay on sim ─────────────────────────────────────────────────
        self.chk_overlay = QCheckBox("Overlay on simulation view (α=0.35)")
        self.chk_overlay.toggled.connect(self._on_overlay_toggle)
        layout.addWidget(self.chk_overlay)

        # ── Auto-refresh on sim step ───────────────────────────────────────
        self.chk_map_autorefresh = QCheckBox("Auto-refresh every N steps")
        self.chk_map_autorefresh.setChecked(False)
        layout.addWidget(self.chk_map_autorefresh)

        self.map_refresh_spin = QSpinBox()
        self.map_refresh_spin.setRange(1, 200)
        self.map_refresh_spin.setValue(20)
        row = QHBoxLayout()
        row.addWidget(QLabel("Refresh every:"))
        row.addWidget(self.map_refresh_spin)
        row.addWidget(QLabel("steps"))
        layout.addLayout(row)

        # ── Manual render button ───────────────────────────────────────────
        btn_render = QPushButton("🖼  Render selected layer")
        btn_render.clicked.connect(self._render_map_layer)
        layout.addWidget(btn_render)

        layout.addStretch()
        return w

    # ── Log tab ────────────────────────────────────────────────────────────

    def _build_log_tab(self):
        w      = QWidget()
        layout = QVBoxLayout(w)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 8))
        layout.addWidget(self.log_text)
        return w

    # ══════════════════════════════════════════════════════════════════════
    # Simulation loop
    # ══════════════════════════════════════════════════════════════════════

    def _sim_step(self):
        if self._step >= self.max_steps:
            self.timer.stop()
            self.btn_play.setChecked(False)
            self.btn_play.setText("▶  Play")
            self._log("✅ Simulation complete.")
            return

        # 1. OverArchingTwin tick
        self.adt.step()

        # 2. Local controllers → actions
        actions, ids = [], []
        for ugv in self.ugv_twins:
            if self._use_global_plan and ugv.assigned_mission is None:
                continue
            obstacles = ugv.get_ugv_view()
            action    = self.controllers[ugv.id].get_action(ugv, obstacles)
            actions.append(action)
            ids.append(ugv.id)

        # 3. Physics
        self.env.step(action=actions, action_id=ids)

        # 4. Render — bypass plt.pause() entirely.
        # Always pass the FULL robot list so every robot's transform is updated
        # each frame (patches use set_transform, not remove+redraw).
        # Visibility is controlled separately via set_visible() on plot_patch_list.
        self._env_plot_step(self.env.robot_list)
        self.canvas.draw_idle()

        # 5. Bookkeeping
        self._step += 1
        self.lbl_step.setText(f"Step: {self._step} / {self.max_steps}")

        if self.env.done():
            self.timer.stop()
            self.btn_play.setChecked(False)
            self.btn_play.setText("▶  Play")
            self._log(f"🏁 Done at step {self._step}")

        if self._step % 15 == 0:
            self._refresh_mission_list()

        # Auto-refresh map view
        if (self.chk_map_autorefresh.isChecked()
                and self._step % max(1, self.map_refresh_spin.value()) == 0):
            self._render_map_layer()

    def _env_plot_step(self, objects):
        """
        Call env_plot.step() with the full object list.
        mode="dynamic" (default) skips static obstacles — correct behaviour.
        """
        ep = self.env._env_plot
        try:
            ep.step(objects=objects)
        except Exception as e:
            self._log(f"[render error] {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Playback controls
    # ══════════════════════════════════════════════════════════════════════

    def _toggle_play(self, checked):
        self._running = checked
        self.btn_play.setText("⏸  Pause" if checked else "▶  Play")
        if checked:
            self.timer.start()
        else:
            self.timer.stop()

    def _single_step(self):
        self._sim_step()

    def _on_speed_changed(self, val):
        ms = max(1, 1000 // val)
        self.timer.setInterval(ms)
        self.lbl_hz.setText(f"{val} Hz")

    # ══════════════════════════════════════════════════════════════════════
    # Visibility controls
    # ══════════════════════════════════════════════════════════════════════

    def _on_robot_toggle(self, rid, checked):
        self._visible_robots[rid] = checked
        # Find the robot and toggle all its patches + sensor patches
        for robot in self.env.robot_list:
            if robot.id != rid:
                continue
            for patch in getattr(robot, 'plot_patch_list', []):
                try:
                    patch.set_visible(checked)
                except Exception:
                    pass
            # Sensors may live on robot.sensors (list) or robot.sensor
            for sensor in getattr(robot, 'sensors', []) or [getattr(robot, 'sensor', None)]:
                if sensor is None:
                    continue
                for patch in getattr(sensor, 'plot_patch_list', []):
                    try:
                        patch.set_visible(checked and self._show_sensors)
                    except Exception:
                        pass
        self.canvas.draw_idle()

    def _set_all_robots(self, state: bool):
        for chk in self._robot_checks.values():
            chk.setChecked(state)

    def _on_sensor_toggle(self, checked):
        """Toggle sensor footprint patches independently of robot body visibility."""
        self._show_sensors = checked
        for robot in self.env.robot_list:
            robot_visible = self._visible_robots.get(robot.id, True)
            sensor_objs = list(getattr(robot, 'sensors', None) or [])
            single = getattr(robot, 'sensor', None)
            if single is not None and single not in sensor_objs:
                sensor_objs.append(single)
            for sensor in sensor_objs:
                for patch in getattr(sensor, 'plot_patch_list', []):
                    try:
                        patch.set_visible(checked and robot_visible)
                    except Exception:
                        pass
        self.canvas.draw_idle()

    def _on_perc_changed(self, mode: str):
        self.perception_mode = mode
        if hasattr(self.adt, 'set_perception_mode'):
            self.adt.set_perception_mode(mode)
        self._log(f"Perception mode → {mode}")

    # ══════════════════════════════════════════════════════════════════════
    # Map layer rendering
    # ══════════════════════════════════════════════════════════════════════

    def _rebuild_map_layer_combo(self):
        """Populate the layer dropdown from adt.grid_map and mission_logger."""
        self.map_layer_combo.clear()

        gm = getattr(self.adt, 'grid_map', None)
        if gm is not None:
            self.map_layer_combo.addItem("Occupancy Grid",    ("occ",  None, None))
            self.map_layer_combo.addItem("Risk Layer",        ("risk", None, None))
        
        mp = getattr(self.adt, 'mission_planner', None)
        if mp is not None:
            ml = getattr(mp, 'mission_logger', None)
            if ml is not None:
                for mission_id, ugv_logs in ml._per_mission_log.items():
                    for entry in ugv_logs:
                        label = f"Cost map : Mission {mission_id}, UGV {entry['id']}"
                        self.map_layer_combo.addItem(
                            label, ("cost", mission_id, entry['id'])
                        )

        if self.map_layer_combo.count() == 0:
            self.map_layer_combo.addItem("No layers available", None)

    def _get_grid_extent(self, gm):
        """Return [xmin, xmax, ymin, ymax] for imshow extent."""
        ox = getattr(gm, '_ox', getattr(gm, 'ox', 0))
        oy = getattr(gm, '_oy', getattr(gm, 'oy', 0))
        W  = getattr(gm, '_W',  getattr(gm, 'width',  40))
        H  = getattr(gm, '_H',  getattr(gm, 'height', 40))
        return [ox, ox + W, oy, oy + H]

    def _render_map_layer(self):
        """Render the currently selected map layer onto the Map View canvas."""
        data = self.map_layer_combo.currentData()
        if data is None:
            return

        kind, mission_id, ugv_id = data
        gm   = getattr(self.adt, 'grid_map', None)
        mp = getattr(self.adt, 'mission_planner', None)
        if mp is not None:
            ml = getattr(mp, 'mission_logger', None)
        cmap = self.cmap_combo.currentText()

        self._map_ax.clear()

        try:
            if kind == "occ" and gm is not None:
                grid   = gm._occ
                extent = self._get_grid_extent(gm)
                im = self._map_ax.imshow(
                    grid.T, origin='lower', extent=extent,
                    cmap='gray_r', vmin=0, vmax=100,
                )
                self._map_ax.set_title("Occupancy Grid")
                self._add_map_colorbar(im, "Occupancy (0=free, 100=occ)")

            elif kind == "risk" and gm is not None:
                grid   = gm._risk
                extent = self._get_grid_extent(gm)
                im = self._map_ax.imshow(
                    grid.T, origin='lower', extent=extent, cmap=cmap,
                )
                self._map_ax.set_title("Risk Layer")
                self._add_map_colorbar(im, "Risk score")

            elif kind == "cost" and gm is not None and ml is not None:
                from overarchingTwin.mission import POSTURE_WEIGHTS

                # Get mission from adt 
                mission = next(
                    (m for m in self.adt.missions if m.mission_id == mission_id), None
                )
                # Get posture from this mission 
                posture  = (mission.mission_posture
                            if mission else self.posture_map_combo.currentText())
                weights  = POSTURE_WEIGHTS[posture]

                cost_img = gm.get_cost_image(
                    weights   = weights,
                    robot_mass= self.cost_mass_spin.value(),
                    v_avg     = self.cost_speed_spin.value(),
                    Ka        = self.cost_anc_spin.value(),
                )
                extent = self._get_grid_extent(gm)
                im = self._map_ax.imshow(
                    cost_img.T, origin='lower', extent=extent,
                    cmap=cmap, vmin=0, vmax=1,
                )
                # Draw the planned path if available
                if ml is not None and mission_id in ml._per_mission_log:
                    for entry in ml._per_mission_log[mission_id]:
                        if entry['id'] == ugv_id:
                            path = entry.get('path')
                            if path is not None and hasattr(path, 'ndim') and path.ndim == 2:
                                self._map_ax.plot(path[0], path[1], 'b-', lw=1.5, label="Path")
                                self._map_ax.plot(path[0][-1], path[1][-1], 'go', ms=5)
                                self._map_ax.plot(path[0][0],  path[1][0],  'rx', ms=7)
                            break
                if(ml._per_mission_assignement_log[mission_id][0]["ugv_id"] == ugv_id):
                    assigned_str = ": [Winner]"
                else:
                    assigned_str = " - [Loser]" 

                self._map_ax.set_title(f"Cost Map - Mission: {mission_id} - UGV: {ugv_id} - Cost:{entry['cost']:.2f} \n[{posture}]" + assigned_str)
                self._add_map_colorbar(im, "Normalised cost")

            self._map_ax.set_xlabel("x [m]")
            self._map_ax.set_ylabel("y [m]")
            self.map_canvas.draw_idle()

            # Switch to Map View tab automatically
            self._canvas_tabs.setCurrentIndex(1)

        except Exception as e:
            self._log(f"[Map render error] {e}")

        # Update overlay on sim canvas if active
        if self._overlay_active:
            self._draw_overlay(kind, mission_id, ugv_id)

    def _add_map_colorbar(self, im, label: str):
        """Add or replace colorbar on the map figure."""
        # Remove old colorbars to avoid stacking
        for ax in self._map_fig.axes[1:]:
            self._map_fig.delaxes(ax)
        self._map_fig.colorbar(im, ax=self._map_ax, label=label, fraction=0.046, pad=0.04)

    def _on_overlay_toggle(self, checked: bool):
        self._overlay_active = checked
        if not checked and self._overlay_im is not None:
            self._overlay_im.remove()
            self._overlay_im = None
            self.canvas.draw_idle()
        elif checked:
            self._render_map_layer()   # immediately update

    def _draw_overlay(self, kind, mission_id, ugv_id):
        """Draw a semi-transparent imshow on the *sim* canvas axes."""
        gm  = getattr(self.adt, 'grid_map', None)
        ml  = getattr(self.adt, 'mission_logger', None)
        ax  = self.env._env_plot.ax

        if self._overlay_im is not None:
            try:
                self._overlay_im.remove()
            except Exception:
                pass
            self._overlay_im = None

        try:
            extent = self._get_grid_extent(gm)
            cmap   = self.cmap_combo.currentText()

            if kind == "occ" and gm is not None:
                grid = gm._occ
                self._overlay_im = ax.imshow(
                    grid.T, origin='lower', extent=extent,
                    cmap='gray_r', vmin=0, vmax=100, alpha=0.35, zorder=2,
                )
            elif kind == "risk" and gm is not None:
                self._overlay_im = ax.imshow(
                    gm._risk.T, origin='lower', extent=extent,
                    cmap=cmap, alpha=0.35, zorder=2,
                )
            elif kind == "cost" and gm is not None:
                from overarchingTwin.mission import POSTURE_WEIGHTS
                mission = next(
                    (m for m in self.adt.missions if m.mission_id == mission_id), None
                )
                posture = (mission.mission_posture
                           if mission else self.posture_map_combo.currentText())
                cost_img = gm.env_map.get_cost_image(
                    weights=POSTURE_WEIGHTS[posture],
                    robot_mass=self.cost_mass_spin.value(),
                    v_avg=self.cost_speed_spin.value(),
                    Ka=self.cost_anc_spin.value(),
                )
                self._overlay_im = ax.imshow(
                    cost_img.T, origin='lower', extent=extent,
                    cmap=cmap, vmin=0, vmax=1, alpha=0.35, zorder=2,
                )
            self.canvas.draw_idle()
        except Exception as e:
            self._log(f"[Overlay error] {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Robot spawn / delete
    # ══════════════════════════════════════════════════════════════════════

    def _spawn_robot(self):
        """
        Create a robot at runtime and wire it into env + GUI.

        Flow:
          1. ObjectFactory.create_robot() builds the object (UAVTwin / UGVTwin / ObjectBase)
          2. env.add_object() handles _init_plot + _step_plot + build_tree atomically
          3. GUI checkbox panel and delete combo are refreshed
        """
        try:
            name    = self.spawn_name_edit.text().strip() or "robot_new"
            rtype   = self.spawn_type_combo.currentText()     # "ugv" | "uav" | "robot"
            kin     = self.spawn_kin_combo.currentText()      # "diff" | "omni" | "acker"
            x       = self.spawn_x_spin.value()
            y       = self.spawn_y_spin.value()
            theta   = self.spawn_theta_spin.value()
            gx      = self.spawn_goal_x_spin.value()
            gy      = self.spawn_goal_y_spin.value()

            robot = self.env.object_factory.create_robot(
                type       = rtype,
                kinematics = {"name": kin},
                state      = [x, y, theta],
                goal       = [gx, gy, 0.0],
                name       = name,
            )

            # add_object: sets _env, calls _init_plot + _step_plot, rebuilds tree
            self.env.add_object(robot)

            # Wire into twin lists so adt / controllers can pick it up
            from irsim.world.robots.uav_twin import UAVTwin
            from irsim.world.robots.ugv_twin import UGVTwin

            if isinstance(robot, UGVTwin):
                self.ugv_twins.append(robot)
                # Add a default controller (same type as first UGV if available)
                if self.controllers and self.ugv_twins:
                    from local_planners.c3bf_qp import CollisionConeCBFController
                    self.controllers[robot.id] = CollisionConeCBFController(
                        robot_type=robot.kinematics, safety_margin=0.05, goal_gain=0.8
                    )
            elif isinstance(robot, UAVTwin):
                self.uav_twins.append(robot)

            self._refresh_robot_panel(robot, add=True)
            self._refresh_delete_combo()
            self.canvas.draw_idle()
            self._log(f"🚀 Spawned {rtype} '{name}' [id={robot.id}] @ ({x:.1f}, {y:.1f})")

        except Exception as e:
            self._log(f"[Spawn error] {e}")

    def _delete_robot(self):
        """
        Delete a robot at runtime.

        Flow:
          1. env.delete_object(id) calls obj.plot_clear() + removes from _objects + build_tree
          2. Remove from twin lists and controllers
          3. Remove GUI checkbox
        """
        text = self.del_combo.currentText()
        if not text:
            return
        try:
            # combo text format: "TypeName [id=N]"
            rid = int(text.split("id=")[1].rstrip("]"))
        except (IndexError, ValueError):
            self._log(f"[Delete error] Cannot parse id from '{text}'")
            return

        robot = next((r for r in self.env.robot_list if r.id == rid), None)
        if robot is None:
            self._log(f"[Delete] Robot id={rid} not found")
            return

        # env.delete_object handles plot_clear + _objects removal + build_tree
        self.env.delete_object(rid)

        # Clean up twin lists and controllers
        self.ugv_twins  = [r for r in self.ugv_twins  if r.id != rid]
        self.uav_twins  = [r for r in self.uav_twins  if r.id != rid]
        self.controllers.pop(rid, None)

        self._refresh_robot_panel(robot, add=False)
        self._refresh_delete_combo()
        self.canvas.draw_idle()
        self._log(f"🗑  Deleted robot '{getattr(robot, 'name', rid)}' [id={rid}]")

    def _refresh_robot_panel(self, robot, add: bool):
        """Add or remove a robot's checkbox in the Visibility tab scroll area."""
        rid = robot.id
        if add:
            lbl = f"{type(robot).__name__}  [id={rid}]"
            chk = QCheckBox(lbl)
            chk.setChecked(True)
            chk.toggled.connect(
                lambda checked, r=rid: self._on_robot_toggle(r, checked)
            )
            self._robot_checks[rid] = chk
            self._visible_robots[rid] = True
            # Insert into the scroll area inner layout
            self._robots_inner_layout.insertWidget(
                self._robots_inner_layout.count() - 1, chk  # before the stretch
            )
        else:
            chk = self._robot_checks.pop(rid, None)
            if chk:
                chk.deleteLater()
            self._visible_robots.pop(rid, None)

    def _refresh_delete_combo(self):
        """Repopulate the delete dropdown from the current robot list."""
        self.del_combo.clear()
        for r in self.env.robot_list:
            self.del_combo.addItem(f"{type(r).__name__}  [id={r.id}]  '{getattr(r,'name','')}'" )

    # ══════════════════════════════════════════════════════════════════════
    # Mission controls
    # ══════════════════════════════════════════════════════════════════════

    def _add_mission(self):
        try:
            mid     = self.mission_id_edit.text().strip() or "mission_dyn"
            mtype   = MissionType[self.mission_type_combo.currentText()]
            gx      = self.goal_x_spin.value()
            gy      = self.goal_y_spin.value()
            posture = self.posture_combo.currentText()

            mission = Mission(
                mission_id     = mid,
                mission_type   = mtype,
                goal_xy        = (gx, gy),
                mission_posture= posture,
            )
            self.adt.add_mission(mission)
            self._use_global_plan = True
            self._log(f"➕ Mission '{mid}' added → ({gx:.1f}, {gy:.1f}) [{posture}]")
            self._refresh_mission_list()
        except Exception as e:
            self._log(f"[Error adding mission] {e}")

    def _refresh_mission_list(self):
        missions = getattr(self.adt, 'missions', [])
        if not missions:
            self.mission_list_label.setText("None")
            return
        lines = [
            f"• {m.mission_id}: {m.mission_type.name} → {m.goal_xy}"
            for m in missions
        ]
        self.mission_list_label.setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════════════
    # Logging & cleanup
    # ══════════════════════════════════════════════════════════════════════

    def _log(self, msg: str):
        self.log_text.append(msg)

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.env._env_plot.close()
        except Exception:
            pass
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def launch(env, adt, ugv_twins, controllers, uav_twins,
           max_steps: int = 800, step_ms: int = 100,
           perception_mode: str = "ugv"):
    """
    Drop-in replacement for the manual for-loop in custom_world.py.
    Blocks until the Qt window is closed, then returns so post-run
    logging (mission_logger, metric_logger) can execute normally.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    gui = SimulationGUI(
        env             = env,
        adt             = adt,
        ugv_twins       = ugv_twins,
        controllers     = controllers,
        uav_twins       = uav_twins,
        max_steps       = max_steps,
        step_ms         = step_ms,
        perception_mode = perception_mode,
    )
    gui.show()
    app.exec()   # blocks here; returns when window is closed