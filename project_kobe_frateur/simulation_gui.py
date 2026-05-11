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
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import to_rgba
from matplotlib.collections import PatchCollection
from shapely import Point
from overarchingTwin.overarching_twin import PerceptionMode
from overarchingTwin.mission import MissionPosture

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QGroupBox, QScrollArea,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget, QTextEdit,
    QFormLayout,QMenu
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QAction

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
        self._visibility_robots  = {r.id: True for r in env.robot_list}
        self._visibility_objects = {o.id: True for o in env.obstacle_list}
        self._show_sensors    = True
        self._show_borders    = False         # Borders on object 
        self._overlay_active  = False        # imshow overlay on sim canvas
        self._overlay_im      = None         # AxesImage handle for overlay
        self._map_fig         = Figure(figsize=(6, 6), tight_layout=True)
        self._map_ax          = self._map_fig.add_subplot(111)
        self._perception_artists = []   # matplotlib patches drawn over sim canvas

        self.setWindowTitle("HDT Simulation — Control Panel")
        self.resize(1440, 860)

        self._build_ui()

        self.timer = QTimer()
        self.timer.setInterval(step_ms)
        self.timer.timeout.connect(self._sim_step)

        self._draw_perception_highlights()

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

        self._click_cid = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

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

        # Detection border layer toggle
        grp_brdr  = QGroupBox("Border Layers")
        brdr_vbox = QVBoxLayout(grp_brdr)
        self.chk_border = QCheckBox("Show border detection")
        self.chk_border.setChecked(False)
        self.chk_border.toggled.connect(self._on_border_toggle)
        brdr_vbox.addWidget(self.chk_border)
        layout.addWidget(grp_brdr)


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

        self.mission_id_edit = QLineEdit("1")

        self.mission_type_combo = QComboBox()
        self.mission_type_combo.addItems(MissionType.get_names())

        self.goal_x_spin = QDoubleSpinBox()
        self.goal_x_spin.setRange(-500, 500)
        self.goal_x_spin.setValue(10.0)
        self.goal_x_spin.setSingleStep(0.5)

        self.goal_y_spin = QDoubleSpinBox()
        self.goal_y_spin.setRange(-500, 500)
        self.goal_y_spin.setValue(35.0)
        self.goal_y_spin.setSingleStep(0.5)

        self.posture_combo = QComboBox()
        self.posture_combo.addItems(MissionPosture.get_names())

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

        # ── Overlay on sim ─────────────────────────────────────────────────

        grp_ovrl  = QGroupBox("Overlay on simulation view")
        ovrl_vbox = QFormLayout(grp_ovrl)
        self.chk_overlay = QCheckBox("Show overlay on simulation view")
        self.chk_overlay.setChecked(False)
        self.chk_overlay.toggled.connect(self._on_overlay_toggle)
        self.alpha_overlay    = QDoubleSpinBox(); self.alpha_overlay.setRange(0.1, 1); self.alpha_overlay.setValue(0.3)
        self.alpha_overlay.setSingleStep(0.05)
        ovrl_vbox.addRow(self.chk_overlay)
        ovrl_vbox.addRow("Overlay alpha:",    self.alpha_overlay)
        layout.addWidget(grp_ovrl)

        
        # ── Auto-refresh on sim step ───────────────────────────────────────
        grp_rfrs  = QGroupBox("Auto-refresh")
        rfrs_vbox = QFormLayout(grp_rfrs)
        self.chk_map_autorefresh = QCheckBox("Enable Auto-refresh")
        self.chk_map_autorefresh.setChecked(False)
        rfrs_vbox.addRow(self.chk_map_autorefresh)
        self.map_refresh_spin = QSpinBox()
        self.map_refresh_spin.setRange(1, 200)
        self.map_refresh_spin.setValue(20)
        row = QHBoxLayout()
        row.addWidget(QLabel("Refresh every:"))
        row.addWidget(self.map_refresh_spin)
        row.addWidget(QLabel("steps"))
        rfrs_vbox.addRow(row)
        layout.addWidget(grp_rfrs)

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
    # Canvas Click
    # ══════════════════════════════════════════════════════════════════════


    def _on_canvas_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        mx, my = event.xdata, event.ydata

        click_point = Point(mx, my)

        # Find nearest robot/obstacle within 1 m tolerance
        hit = None
        best = 1.0  # metres
        for obj in self.env.objects:
            geom = getattr(obj, 'geometry', getattr(obj, '_geometry', None))
            if geom is None:
                continue

            dist = click_point.distance(geom)

            if dist < best:
                best, hit = dist, obj

        if hit is None:
            return

        # Build context menu at cursor
        menu  = QMenu(self)
        title = QAction(f"{type(hit).__name__}  id={hit.id}", self)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()
        menu.addAction("🗑  Delete",       lambda: self._delete_by_obj(hit))
        menu.addAction("👁  Toggle visible", lambda: self._toggle_visible_by_obj(hit))
        menu.addAction("🚫 Toggle UAV Fault", lambda: self._fault_inject_uav(hit))
        menu.exec(self.canvas.mapToGlobal(
            self.canvas.mapFromGlobal(
                self.canvas.cursor().pos()
            )
        ))

    def _delete_by_obj(self, obj):
        #Find the object

        # Perform the actual deletion in the environment
        self.env.delete_object(obj.id)

        # Handle UI updates if it was a robot
        if getattr(obj, 'role', None) == "robot":
            self._refresh_robot_panel_by_id(obj.id, remove=True)
            self._refresh_delete_combo()
        
        # Refresh visuals
        self._draw_perception_highlights()
        self.canvas.draw_idle()

    def _toggle_visible_by_obj(self, obj):

        # Get object visibility status 
        if obj.id not in self._visibility_objects.keys():
            self.legger.warning(f"Object selected not in visibility dict")
            return
        new_state = not self._visibility_objects.get(obj.id)
        self._visibility_objects[obj.id] = new_state

        # For the robot toggles on the side 
        if getattr(obj, 'role', None) == "robot":
            self._visibility_robots[obj.id] = new_state
            chk = self._robot_checks.get(obj.id)
            if chk:
                chk.setChecked(new_state)   # triggers _on_robot_toggle via signal
            return
        
        #Hide/Show the base irsim visual patch
        if hasattr(obj, 'plot_patch_list'):
            for patch in obj.plot_patch_list:
                try:
                    patch.set_visible(new_state)
                except Exception:
                    pass

        #Set object unobstructed 
        obj.unobstructed = not new_state

        # Update percieved obstacles in adt 
        if new_state:
            self.adt.add_percieved_obstacle(obj)
            self._log(f"Unhiding obj id={obj.id} (unobstructed)")

        else:
            self.adt.remove_percieved_obstacles(obj)
            self._log(f"Hiding obj id={obj.id} (unobstructed)")

        #force redraw
        self._draw_perception_highlights()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _fault_inject_uav(self, obj):
        """Remove object from UAV fleet sensor detections (fault injection)."""
        fleet = getattr(self.adt, 'uav_fleet', None)
        if fleet and hasattr(fleet, 'hidden_objects'):
            if obj in fleet.hidden_objects:
                fleet.hidden_objects.remove(obj)
            else:
                fleet.hidden_objects.add(obj)
        #force redraw
        self._draw_perception_highlights()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

        self._log(f"🚫 UAV fault: hiding obj id={obj.id} from UAV perception")

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

        #Draw border around object that are percieved
        self._draw_perception_highlights()

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

        self._draw_perception_highlights()

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
        self._visibility_robots[rid] = checked
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
            robot_visible = self._visibility_robots.get(robot.id, True)
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

        self._map_fig.clear()   # Clear the whole figure
        self._map_ax = self._map_fig.add_subplot(111) # Recreate a fresh, full-size axes
        
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
                posture  = (mission.mission_posture)
                weights  = POSTURE_WEIGHTS[posture]

                # Safely find the corresponding UGV object in the OverArchingTwin
                ugv = next((u for u in self.adt._ugvs if getattr(u, 'id', str(id(u))) == ugv_id), None)


                cost_img = gm.get_cost_image(
                    weights   = weights,
                    robot_mass= ugv.mass,
                    v_avg     = ugv.avg_speed,
                    Ka        = ugv.ancillary_drain)
                
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
        if not checked and hasattr(self, '_overlay_elements'):
            for item in self._overlay_elements:
                try:
                    item.remove()
                except Exception:
                    pass
            self._overlay_elements = []
            self.canvas.draw_idle()
        elif checked:
            self._render_map_layer()

    def _draw_overlay(self, kind, mission_id, ugv_id):
        gm = getattr(self.adt, 'grid_map', None)
        mp = getattr(self.adt, 'mission_planner', None)
        ml = getattr(mp, 'mission_logger', None) if mp else None
        ax = self.env._env_plot.ax # Simulation axes

        # Clear previous elements
        if hasattr(self, '_overlay_elements'):
            for item in self._overlay_elements:
                try:
                    item.remove()
                except: pass
        self._overlay_elements = []

        try:
            extent = self._get_grid_extent(gm)
            alpha_val = self.alpha_overlay.value()
            
            # draw overlay map
            img_data = None
            if kind == "occ":
                img_data = gm._occ.T
            elif kind == "risk":
                img_data = gm._risk.T
            elif kind == "cost":
                mission = next((m for m in self.adt.missions if m.mission_id == mission_id), None)
                from overarchingTwin.mission import POSTURE_WEIGHTS, MissionPosture
                posture = mission.mission_posture
                
                weights = POSTURE_WEIGHTS[posture]
                ugv = next((u for u in self.adt._ugvs if getattr(u, 'id', None) == ugv_id), None)
                
                if ugv:
                    img_data = gm.get_cost_image(weights=weights, robot_mass=ugv.mass, 
                                               v_avg=ugv.avg_speed, Ka=ugv.ancillary_drain).T

            if img_data is not None:
                im = ax.imshow(img_data, origin='lower', extent=extent,
                               cmap=self.cmap_combo.currentText(), 
                               alpha=alpha_val, zorder=2) # Base layer
                self._overlay_elements.append(im)

            # draw path
            if ml is not None and mission_id in ml._per_mission_log:
                logs = ml._per_mission_log[mission_id]
                for entry in logs:
                    
                    if str(entry['id']) == str(ugv_id): # Force string comparison to be safe
                        path = entry.get('path')
                        if path is not None and hasattr(path, 'ndim') and path.ndim == 2:
                            # Use high Z-ORDER to ensure it's on top of everything
                            ln, = ax.plot(path[0], path[1], color='cyan', lw=2.5, zorder=10, label="Overlay Path")
                            g_dot, = ax.plot(path[0][-1], path[1][-1], 'go', ms=8, zorder=11)
                            s_x, = ax.plot(path[0][0], path[1][0], 'rx', ms=10, zorder=11)
                            
                            self._overlay_elements.extend([ln, g_dot, s_x])
                            # print(f"Successfully drew path for UGV {ugv_id}")
                        break

            self.canvas.draw_idle()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"[Overlay error] {e}")

    def _on_border_toggle(self, checked):
        self._show_borders = checked
        #force redraw
        self._draw_perception_highlights()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()


    def _draw_perception_highlights(self):
        # 1. Fast cleanup: We will now be clearing 1 Collection instead of 100s of patches
        for a in self._perception_artists:
            try: a.remove()
            except Exception: pass
        self._perception_artists.clear()

        if not getattr(self, '_show_borders', True):  # Safer attribute check
            return

        ax = self.env._env_plot.ax
        gm = getattr(self.adt, 'grid_map', None)
        if gm is None: return

        # 2. O(1) Lookups: Pre-compute sets outside the loop to avoid redundant calculations
        try: uav_obs_ids = {o.id for o in self.adt.get_uavs_view()}
        except Exception: uav_obs_ids = set()
            
        try: ugv_obs_ids = {o.id for o in self.adt.get_ugvs_view()}
        except Exception: ugv_obs_ids = set()

        fleet = getattr(self.adt, 'uav_fleet', None)
        fleet_hidden_ids = {getattr(o, 'id', o) for o in getattr(fleet, 'hidden_objects', [])}
        
        # CRITICAL FIX: Convert list to a set of IDs for O(1) lookup speed
        adt_perceived = getattr(self.adt, 'percieved_obstacles', [])
        adt_perceived_ids = {getattr(o, 'id', id(o)) for o in adt_perceived}

        # Prepare lists for the PatchCollection
        patches = []
        facecolors = []
        edgecolors = []
        linewidths = []
        linestyles = []

        for obj in self.env.obstacle_list:
            obj_id = getattr(obj, 'id', None)
            if obj_id is None: continue

            in_uav = obj_id in uav_obs_ids
            in_ugv = obj_id in ugv_obs_ids
            in_adt = obj_id in adt_perceived_ids 
            
            is_fleet_hidden = obj_id in fleet_hidden_ids
            is_gui_hidden = not self._visibility_objects.get(obj_id, True) 

            # Skip if hidden and nobody sees it
            if not (is_gui_hidden or is_fleet_hidden) and not (in_uav or in_ugv or in_adt):
                continue

            state = getattr(obj, 'state', None)
            if state is None: continue
            x, y = float(state.flat[0]), float(state.flat[1])

            # ==========================================
            # STYLING RULES
            # ==========================================
            if is_gui_hidden:
                color, lw, facecolor, alpha, ls = '#888888', 1.5, '#DDDDDD', 0.2, ':'
            elif is_fleet_hidden:
                color, lw, facecolor, alpha, ls = 'red', 2.5, 'red', 0.4, '-.'
            elif in_uav and in_ugv:
                color, lw, facecolor, alpha, ls = 'white', 2.0, 'none', 1.0, '--'
            elif in_uav:
                color, lw, facecolor, alpha, ls = 'cyan', 2.5, 'none', 1.0, '--'
            elif in_ugv:
                color, lw, facecolor, alpha, ls = 'orange', 2.5, 'none', 1.0, '--'
            elif in_adt:
                color, lw, facecolor, alpha, ls = 'yellow', 2.0, 'none', 1.0, '--'
            else:
                continue

            vertices = getattr(obj, 'vertices', None)
            shape = getattr(obj, 'shape', 'circle')

            # Create geometry but DO NOT add to axes yet
            if vertices is not None and shape != 'circle':
                v_array = np.array(vertices)
                if v_array.shape[0] == 2: v_array = v_array.T 
                
                center = np.array([x, y])
                direction = v_array - center
                # Optimized math: +1e-5 avoids np.maximum overhead
                norms = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-5
                v_padded = center + direction * (1 + 0.1 / norms)

                patch = Polygon(v_padded, closed=True)
            else:
                radius = getattr(obj, 'radius', 0.5) + 0.15
                patch = Circle((x, y), radius)

            # Store properties for batching. We bake the alpha into an RGBA tuple 
            # because PatchCollection applies a single alpha to the whole collection otherwise.
            fc_rgba = to_rgba(facecolor, alpha=alpha) if facecolor != 'none' else (0, 0, 0, 0)
            ec_rgba = to_rgba(color, alpha=alpha)

            patches.append(patch)
            facecolors.append(fc_rgba)
            edgecolors.append(ec_rgba)
            linewidths.append(lw)
            linestyles.append(ls)

        # 3. Batch Render: Draw everything at once via PatchCollection
        if patches:
            collection = PatchCollection(
                patches, 
                facecolors=facecolors, 
                edgecolors=edgecolors, 
                linewidths=linewidths, 
                linestyles=linestyles, 
                zorder=6,
                match_original=False # Forces collection to use our explicit arrays
            )
            ax.add_collection(collection)
            self._perception_artists.append(collection)    # ══════════════════════════════════════════════════════════════════════
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
            self._visibility_robots[rid] = True
            # Insert into the scroll area inner layout
            self._robots_inner_layout.insertWidget(
                self._robots_inner_layout.count() - 1, chk  # before the stretch
            )
        else:
            chk = self._robot_checks.pop(rid, None)
            if chk:
                chk.deleteLater()
            self._visibility_robots.pop(rid, None)

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