# simulation_gui.py
"""
PyQt6 GUI for IRSim + OverArchingTwin.

Design pattern: 
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
import matplotlib
matplotlib.use("QtAgg")  # Must be set before any plt/irsim import

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QGroupBox, QScrollArea,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget, QTextEdit,
    QFormLayout,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from overarchingTwin.mission_planner import Mission, MissionType
from overarchingTwin.overarching_twin import PerceptionMode

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
        panel = QWidget()
        vbox  = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.canvas = FigureCanvasQTAgg(self.env._env_plot.fig)
        toolbar     = NavigationToolbar2QT(self.canvas, panel)

        vbox.addWidget(toolbar)
        vbox.addWidget(self.canvas, stretch=1)
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
