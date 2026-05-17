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
import datetime
import os
import sys

import matplotlib
import numpy as np
from matplotlib.widgets import PolygonSelector

matplotlib.use("QtAgg")  # Must be set before any plt/irsim import
import matplotlib.pyplot as plt
from local_controllers.c3bf_qp import CollisionConeCBFController
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon
from overarching_twin.mission import POSTURE_WEIGHTS, MissionPosture , Mission ,MissionType
from overarching_twin.overarching_twin import PerceptionMode
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from shapely import Point

from irsim.world.robots.uav_twin import UAVTwin
from irsim.world.robots.ugv_twin import UGVTwin

plt.rc('axes', labelsize=17)    # Font size of the x and y axis labels ("x [m]", "y [m]")
plt.rc('xtick', labelsize=15)   # Font size of the x-axis tick numbers
plt.rc('ytick', labelsize=15)   # Font size of the y-axis tick numbers
plt.rc('font', size=13)         # General default font size
plt.rc('axes', titlesize=16)     # Font size for the title above the axes
plt.rc('legend', fontsize=12)   # Global font size for all legends

class SimulationGUI(QMainWindow):
    """Main window: matplotlib canvas (left) + tabbed control panel (right)."""

    def __init__(self, env, adt, controllers,
                 max_steps=800, step_ms=100, perception_mode="ALL"):
        super().__init__()

        # ── Core references ────────────────────────────────────────────────
        self.env = env
        self.adt = adt
        self.controllers = controllers
        self.max_steps = max_steps
        self.perception_mode = perception_mode

        # ── State ──────────────────────────────────────────────────────────
        self._step = 0
        self._running = False
        self._use_global_plan = False
        self._visibility_robots = {r.id: True for r in env.robot_list}
        self._visibility_objects = {o.id: True for o in env.obstacle_list}
        self._show_sensors = True
        self._show_borders = False  # Borders on object
        self._overlay_active = False  # imshow overlay on sim canvas
        self._overlay_im = None  # AxesImage handle for overlay
        self._map_fig = Figure(figsize=(6, 6), tight_layout=True)
        self._map_ax = self._map_fig.add_subplot(111)
        self._perception_artists = []  # matplotlib patches drawn over sim canvas

        # Saving runs
        self._saved_runs = {}  # key: unique label → {rid, xs, ys, artist}
        self._ugv_traces = {ugv.id: ([], []) for ugv in self.adt.all_ugvs}
        self._saved_artists = {}
        self._run_colors = ['lime','red', 'cyan', 'magenta', 'yellow', 'white']

        self._last_robot_status = {}

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
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._canvas_tabs = QTabWidget()

        # ── Tab 0: Simulation ──────────────────────────────────────────────
        sim_widget = QWidget()
        sim_vbox = QVBoxLayout(sim_widget)
        sim_vbox.setContentsMargins(0, 0, 0, 0)
        self.canvas = FigureCanvasQTAgg(self.env._env_plot.fig)
        sim_toolbar = CustomNavigationToolbar(self.canvas, sim_widget, self.export_active_view)
        sim_vbox.addWidget(sim_toolbar)
        sim_vbox.addWidget(self.canvas, stretch=1)

        self._click_cid = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        # ── Tab 1: Map Viewer ──────────────────────────────────────────────
        map_widget = QWidget()
        map_vbox = QVBoxLayout(map_widget)
        map_vbox.setContentsMargins(0, 0, 0, 0)
        self.map_canvas = FigureCanvasQTAgg(self._map_fig)
        map_toolbar = CustomNavigationToolbar(self.map_canvas, map_widget, self.export_active_view)
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

        btn_reset = QPushButton("⏮ Reset")
        btn_reset.clicked.connect(self._reset_sim)
        bar.addWidget(btn_reset)

        self.lbl_step = QLabel("Step: 0")
        self.lbl_step.setMinimumWidth(130)

        spd_lbl = QLabel("Speed:")
        self.sld_speed = QSlider(Qt.Orientation.Horizontal)
        self.sld_speed.setRange(1, 50)
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
        tabs.addTab(self._build_robots_tab(), "🤖  Robots")
        tabs.addTab(self._build_mission_tab(), "🎯  Missions")
        tabs.addTab(self._build_maps_tab(), "🗺  Maps")
        tabs.addTab(self._build_log_tab(), "📋  Log")
        return tabs

    # ── Visibility tab ─────────────────────────────────────────────────────

    def _build_visibility_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        # Perception mode
        grp_perc = QGroupBox("Perception Mode")
        perc_vbox = QVBoxLayout(grp_perc)
        self.combo_perc = QComboBox()
        self.combo_perc.addItems(PerceptionMode.get_names())
        self.combo_perc.setCurrentText(self.perception_mode)
        self.combo_perc.currentTextChanged.connect(self._on_perc_changed)
        perc_vbox.addWidget(self.combo_perc)
        layout.addWidget(grp_perc)

        # Sensor layer toggle
        grp_sens = QGroupBox("Sensor Layers")
        sens_vbox = QVBoxLayout(grp_sens)
        self.chk_sensors = QCheckBox("Show sensor footprints")
        self.chk_sensors.setChecked(True)
        self.chk_sensors.toggled.connect(self._on_sensor_toggle)
        sens_vbox.addWidget(self.chk_sensors)
        layout.addWidget(grp_sens)

        # Detection border layer toggle
        grp_brdr = QGroupBox("Border Layers")
        brdr_vbox = QVBoxLayout(grp_brdr)
        self.chk_border = QCheckBox("Show border detection")
        self.chk_border.setChecked(False)
        self.chk_border.toggled.connect(self._on_border_toggle)
        brdr_vbox.addWidget(self.chk_border)
        layout.addWidget(grp_brdr)

        # objects arrow layer toggle
        grp_arr = QGroupBox("Arrow Layer")
        arr_vbox = QVBoxLayout(grp_arr)
        self.chk_arrows = QCheckBox("Show orientation arrows")
        self.chk_arrows.setChecked(True)
        self.chk_arrows.toggled.connect(self._on_arrows_toggle)
        arr_vbox.addWidget(self.chk_arrows)
        layout.addWidget(grp_arr)

        # Per-robot checkboxes
        grp_robots = QGroupBox("Individual Robots")
        robots_vbox = QVBoxLayout(grp_robots)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(280)
        inner = QWidget()
        in_vbox = QVBoxLayout(inner)
        in_vbox.setSpacing(2)
        self._robots_inner_layout = in_vbox  # kept for dynamic add/remove

        self._robot_status_labels = {}
        self._robot_checks = {}

        for robot in self.env.robot_list:
            row = QWidget()
            hbox = QHBoxLayout(row)
            hbox.setContentsMargins(0, 0, 0, 0)

            chk = QCheckBox(f"{type(robot).__name__} [id={robot.id}]")
            chk.setChecked(True)
            chk.toggled.connect(lambda checked, rid=robot.id: self._on_robot_toggle(rid, checked))
            self._robot_checks[robot.id] = chk

            lbl = QLabel("⏳")
            lbl.setFixedWidth(60)
            lbl.setStyleSheet("color: gray; font-size: 9px;")
            self._robot_status_labels[robot.id] = lbl

            hbox.addWidget(chk)
            hbox.addWidget(lbl)
            in_vbox.addWidget(row)

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

        # Saved runs list
        grp_saved = QGroupBox("Saved Runs")
        saved_vbox = QVBoxLayout(grp_saved)
        self.saved_runs_list = QListWidget()
        self.saved_runs_list.itemChanged.connect(self._on_saved_run_toggle)
        saved_vbox.addWidget(self.saved_runs_list)
        layout.addWidget(grp_saved)

        layout.addStretch()

        return w

    # ── Robots tab (spawn / delete) ────────────────────────────────────────

    def _build_robots_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        # ── Spawn ──────────────────────────────────────────────────────────
        grp_spawn = QGroupBox("Spawn Robot")
        form = QFormLayout(grp_spawn)

        self.spawn_name_edit = QLineEdit("robot_new")

        self.spawn_type_combo = QComboBox()
        self.spawn_type_combo.addItems(["ugv", "uav", "robot"])

        self.spawn_kin_combo = QComboBox()
        self.spawn_kin_combo.addItems(["diff", "omni", "acker"])

        self.spawn_x_spin = QDoubleSpinBox()
        self.spawn_x_spin.setRange(-500, 500)
        self.spawn_x_spin.setValue(5.0)

        self.spawn_y_spin = QDoubleSpinBox()
        self.spawn_y_spin.setRange(-500, 500)
        self.spawn_y_spin.setValue(5.0)

        self.spawn_theta_spin = QDoubleSpinBox()
        self.spawn_theta_spin.setRange(-3.15, 3.15)
        self.spawn_theta_spin.setValue(0.0)
        self.spawn_theta_spin.setSingleStep(0.1)

        self.spawn_goal_x_spin = QDoubleSpinBox()
        self.spawn_goal_x_spin.setRange(-500, 500)
        self.spawn_goal_x_spin.setValue(10.0)

        self.spawn_goal_y_spin = QDoubleSpinBox()
        self.spawn_goal_y_spin.setRange(-500, 500)
        self.spawn_goal_y_spin.setValue(10.0)

        form.addRow("Name:", self.spawn_name_edit)
        form.addRow("Type:", self.spawn_type_combo)
        form.addRow("Kinematics:", self.spawn_kin_combo)
        form.addRow("X [m]:", self.spawn_x_spin)
        form.addRow("Y [m]:", self.spawn_y_spin)
        form.addRow("θ [rad]:", self.spawn_theta_spin)
        form.addRow("Goal X:", self.spawn_goal_x_spin)
        form.addRow("Goal Y:", self.spawn_goal_y_spin)

        btn_spawn = QPushButton("🚀  Spawn Robot")
        btn_spawn.clicked.connect(self._spawn_robot)

        # ── Robot Actions ─────────────────────────────────────────────────────────
        grp_act = QGroupBox("Robot Actions")
        act_vbox = QVBoxLayout(grp_act)

        self.act_combo = QComboBox()
        self._refresh_delete_combo()

        act_vbox.addWidget(self.act_combo)

        btn_save_run = QPushButton("💾  Save run (path + figures)")
        btn_robot_reset = QPushButton("⏮  Reset Selected")
        btn_del = QPushButton("🗑  Delete Selected")

        btn_save_run.clicked.connect(self._save_robot_run)
        btn_robot_reset.clicked.connect(self._reset_single_robot)
        btn_del.clicked.connect(self._delete_robot)

        act_vbox.addWidget(btn_save_run)
        act_vbox.addWidget(btn_robot_reset)
        act_vbox.addWidget(btn_del)

        layout.addWidget(grp_spawn)
        layout.addWidget(btn_spawn)
        layout.addWidget(grp_act)
        layout.addStretch()
        return w

    # ── Mission tab ────────────────────────────────────────────────────────

    def _build_mission_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        grp = QGroupBox("New Mission")
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

        form.addRow("Mission ID:", self.mission_id_edit)
        form.addRow("Type:", self.mission_type_combo)
        form.addRow("Goal X [m]:", self.goal_x_spin)
        form.addRow("Goal Y [m]:", self.goal_y_spin)
        form.addRow("Posture:", self.posture_combo)

        btn_add = QPushButton("➕  Add Mission")
        btn_add.clicked.connect(self._add_mission)

        grp_active = QGroupBox("Active Missions")
        act_vbox = QVBoxLayout(grp_active)
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
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)

        # ── Layer selector ─────────────────────────────────────────────────
        grp_sel = QGroupBox("Layer")
        sel_vbox = QVBoxLayout(grp_sel)

        self.map_layer_combo = QComboBox()
        self._rebuild_map_layer_combo()  # populate from current adt state
        sel_vbox.addWidget(self.map_layer_combo)

        btn_refresh_list = QPushButton("🔄  Refresh layer list")
        btn_refresh_list.clicked.connect(self._rebuild_map_layer_combo)
        sel_vbox.addWidget(btn_refresh_list)
        layout.addWidget(grp_sel)

        # ── Colormap ───────────────────────────────────────────────────────
        grp_cmap = QGroupBox("Colormap")
        cmap_vbox = QVBoxLayout(grp_cmap)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["YlOrRd", "viridis", "plasma", "gray", "RdYlGn_r"])
        cmap_vbox.addWidget(self.cmap_combo)
        layout.addWidget(grp_cmap)

        # ── Overlay on sim ─────────────────────────────────────────────────

        grp_ovrl = QGroupBox("Overlay on simulation view")
        ovrl_vbox = QFormLayout(grp_ovrl)
        self.chk_overlay = QCheckBox("Show overlay on simulation view")
        self.chk_overlay.setChecked(False)
        self.chk_overlay.toggled.connect(self._on_overlay_toggle)
        self.alpha_overlay = QDoubleSpinBox()
        self.alpha_overlay.setRange(0.1, 1)
        self.alpha_overlay.setValue(0.3)
        self.alpha_overlay.setSingleStep(0.05)
        ovrl_vbox.addRow(self.chk_overlay)
        ovrl_vbox.addRow("Overlay alpha:", self.alpha_overlay)
        layout.addWidget(grp_ovrl)

        # ── Auto-refresh on sim step ───────────────────────────────────────
        grp_rfrs = QGroupBox("Auto-refresh")
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
        w = QWidget()
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
        if event.inaxes is None or event.button != 3:
            return

        #For the polySelector tool
        if hasattr(self, 'poly_selector') and self.poly_selector.active:
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

        if hit:
            # Build context menu at cursor For object hit
            menu = QMenu(self)
            title = QAction(f"{type(hit).__name__}  id={hit.id}", self)
            title.setEnabled(False)
            menu.addAction(title)
            menu.addSeparator()
            menu.addAction("🗑  Delete", lambda: self._delete_by_obj(hit))
            menu.addAction("👁  Toggle visible", lambda: self._toggle_visible_by_obj(hit))
            menu.addAction("🚫 Toggle UAV Fault", lambda: self._fault_inject_uav(hit))
            menu.exec(self.canvas.mapToGlobal(
                self.canvas.mapFromGlobal(
                    self.canvas.cursor().pos()
                )
            ))
        else:
            # Build context menu for Empty Space
            menu = QMenu(self)
            title = QAction(f"📍 Location: ({mx:.2f}, {my:.2f})", self)
            title.setEnabled(False)
            menu.addAction(title)
            menu.addSeparator()

            # Submenu: Add Object
            add_menu = menu.addMenu("➕ Add Object...")
            add_menu.addAction("⚪ Add Circle", lambda: self._dialog_spawn(mx, my, "circle"))
            add_menu.addAction("🧱 Add Rectangle", lambda: self._dialog_spawn(mx, my, "rectangle"))
            add_menu.addSeparator()
            add_menu.addAction("🤖 Add Robot", lambda: self._dialog_spawn(mx, my, "robot"))

            menu.addSeparator()

            # Interactive drawing mode
            menu.addAction("✏️ Draw Custom Polygon", lambda: self._activate_polygon_mode())

            menu.exec(self.canvas.mapToGlobal(self.canvas.mapFromGlobal(self.canvas.cursor().pos())))

    def _dialog_spawn(self, mx, my, spawn_type):
        """Opens the custom dialog and spawns the object if accepted."""
        dialog = ObjectSpawnDialog(spawn_type, self)
        if not dialog.exec():
            return  # User canceled

        vals = dialog.get_values()

        try:
            if spawn_type in ["circle", "rectangle"]:
                # Shape dict directly maps to GeometryFactory requirements
                shape_dict = {"name": spawn_type}
                shape_dict.update({k: v for k, v in vals.items() if k != "theta"})
                theta = vals.get("theta", 0.0)

                obs = self.env.object_factory.create_obstacle(
                    shape=shape_dict, 
                    state=[mx, my, theta],
                    name=f"obs_{spawn_type}_{int(mx)}_{int(my)}"
                )

                self._add_by_obj(obs)

            elif spawn_type == "robot":
                # Create the robot
                robot = self.env.object_factory.create_robot(
                    type=vals["type"],
                    kinematics={"name": vals["kin"]},
                    state=[mx, my, vals["theta"]],
                    goal=[mx + 5.0, my + 5.0, 0.0],  # Default goal slightly ahead
                    name=vals["name"],
                )

                self._add_by_obj(robot)


        except Exception as e:
            self._log(f"❌ [Spawn Error] {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Interactive Drawing (PolygonSelector)
    # ══════════════════════════════════════════════════════════════════════

    def _activate_polygon_mode(self):
        """Activates the interactive polygon drawing tool."""
        self.poly_selector = PolygonSelector(
            self.env._env_plot.ax,
            self._on_polygon_complete,
            useblit=True
        )
        self._log("✏️ Polygon mode active: Click to add corners. Press 'Enter' to finish, or 'Esc' to cancel.")

    def _on_polygon_complete(self, vertices):
        """Callback triggered when the user presses 'Enter' after drawing."""
        if len(vertices) < 3:
            self._log("⚠️ A polygon needs at least 3 points! Canceled.")
            self.poly_selector.set_active(False)
            return

        verts_array = np.array(vertices)

        # Calculate the centroid (the physical x, y center of the object)
        cx = np.mean(verts_array[:, 0])
        cy = np.mean(verts_array[:, 1])

        # Convert to local coordinates (ir-sim expects vertices relative to center)
        local_vertices = verts_array - [cx, cy]

        shape_dict = {
            "name": "polygon",
            "vertices": local_vertices.tolist()
        }
        try:
            # Spawn the object
            obs = self.env.object_factory.create_obstacle(
                shape=shape_dict,
                state=[cx, cy, 0.0],
                name=f"poly_obs_{int(cx)}_{int(cy)}"
            )
            self._add_by_obj(obs)

            self._log(f"📐 Custom polygon spawned at ({cx:.1f}, {cy:.1f})")

        except Exception as e:
            self._log(f"❌ [Spawn Error] {e}")

        # Deactivate and remove the drawing tool from the canvas
        self.poly_selector.set_active(False)
        self.poly_selector.disconnect_events()
        self.canvas.draw_idle()

    def _add_by_obj(self,obj):
        self.env.add_object(obj)

        if getattr(obj, 'role', None) == "robot":
            if isinstance(obj, UGVTwin):
                self.adt.add_ugv(obj)
                if self.controllers:
                    self.controllers[obj.id] = CollisionConeCBFController(
                        robot_type=obj.kinematics, safety_margin=0.05, goal_gain=0.8
                    )
            elif isinstance(obj, UAVTwin):
                self.adt.uav_fleet.add_uav(obj)
            else:
                print("Regular robot type not supported yet")
                return

            self._visibility_objects[obj.id] = True

            # Update the side panel GUI
            self._refresh_robot_panel(obj, add=True)
            self._refresh_delete_combo()
        else:
            self.adt.add_perceived_obstacle(obj)
            self._visibility_objects[obj.id] = True

        self._log(f"🚀 Spawnd {obj.type} '{obj.name}' at ({float(obj.state.flat[0]):.2f}, {float(obj.state.flat[1]):.2f})")

        self._draw_perception_highlights()
        self.canvas.draw_idle()


    def _delete_by_obj(self, obj):
        if getattr(obj, 'role', None) == "robot":
            # Select it in combo then reuse _delete_robot
            idx = self.act_combo.findData(obj.id)
            if idx >= 0:
                self.act_combo.setCurrentIndex(idx)
            self._delete_robot()
        else:
            self._delete_obstacle(obj)


    def _toggle_visible_by_obj(self, obj):

        # Get object visibility status
        if obj.id not in self._visibility_objects:
            self.legger.warning("Object selected not in visibility dict")
            return
        new_state = not self._visibility_objects.get(obj.id)
        self._visibility_objects[obj.id] = new_state

        # For the robot toggles on the side
        if getattr(obj, 'role', None) == "robot":
            self._visibility_robots[obj.id] = new_state
            chk = self._robot_checks.get(obj.id)
            if chk:
                chk.setChecked(new_state)  # triggers _on_robot_toggle via signal
            return

        # Hide/Show the base irsim visual patch
        if hasattr(obj, 'plot_patch_list'):
            for patch in obj.plot_patch_list:
                try:
                    patch.set_visible(new_state)
                except Exception:
                    pass

        # Set object unobstructed
        obj.unobstructed = not new_state

        # Update perceived obstacles in adt
        if new_state:
            self.adt.add_perceived_obstacle(obj)
            self._log(f"Unhiding obj id={obj.id} (unobstructed)")

        else:
            self.adt.remove_perceived_obstacle(obj)
            self._log(f"Hiding obj id={obj.id} (unobstructed)")

        # force redraw
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
        # force redraw
        self._draw_perception_highlights()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

        self._log(f"🚫 UAV fault: hiding obj id={obj.id} from UAV perception")

    # ══════════════════════════════════════════════════════════════════════
    # Figure Export (high res)
    # ══════════════════════════════════════════════════════════════════════

    def export_active_view(self):

        """Exports the currently visible tab as a High-Res PNG or PDF."""
        # Determine which tab is active
        current_tab = self._canvas_tabs.currentIndex()

        if current_tab == 0:
            fig_to_save = self.env._env_plot.fig
            default_name = "simulation_view.pdf"
        elif current_tab == 1:
            fig_to_save = self._map_fig
            default_name = "map_view.pdf"
        else:
            return  # Safety catch

        # Open the Save Dialog
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            None,
            "Export Current View",
            default_name,
            "Vector PDF (*.pdf);;High-Res PNG (*.png)"
        )

        if not file_path:
            return # User canceled file dialog

        # Open our Custom Dialog for settings
        dialog = ExportSettingsDialog(self)
        if not dialog.exec():
            return  # User clicked 'Cancel' on the settings prompt

        # Extract the user's choices
        custom_title, _keep_ax_title = dialog.get_values()

        # Apply the Super Title if the user typed one
        if custom_title.strip():
            fig_to_save.suptitle(custom_title.strip(), fontsize=18, fontweight='bold')

        # Hide existing axis titles if the user unchecked the box
        original_ax_titles = []
        if not _keep_ax_title:
            # Iterate through all axes in the figure (safely handles subplots)
            for ax in fig_to_save.axes:
                original_ax_titles.append((ax, ax.get_title()))
                ax.set_title("") # Clear it temporarily

        # Adjust layout before saving to account for text additions/removals
        fig_to_save.tight_layout()

        if file_path.endswith('.png'):
            # PNG is a raster format, so it needs the high DPI multiplier
            fig_to_save.savefig(file_path, dpi=300, bbox_inches='tight')
        else:
            # PDF is a vector format. DPI doesn't matter for the drawn lines/shapes!
            fig_to_save.savefig(file_path, bbox_inches='tight')

        print(f"Successfully exported to: {file_path}")


        # Remove the custom super title from the live GUI
        if custom_title.strip():
            fig_to_save.suptitle("") # Clear the text

        # Restore normal layout
        fig_to_save.tight_layout()

        # Hard sync the canvas to prevent the Qt event loop from hanging
        if current_tab == 0 and hasattr(self, 'canvas'):
            self.canvas.draw()         # Force immediate Matplotlib redraw
            self.canvas.flush_events() # Force Qt to process the GUI update
        elif current_tab == 1 and hasattr(self, 'map_canvas'):
            self.map_canvas.draw()
            self.map_canvas.flush_events()


    # ══════════════════════════════════════════════════════════════════════
    # Robot selection
    # ══════════════════════════════════════════════════════════════════════

    def _get_selected_robot(self):
        rid = self.act_combo.currentData()
        return next((r for r in self.env.robot_list if r.id == rid), None)

    def _toggle_selected_robot_vis(self):
        robot = self._get_selected_robot()
        if robot:
            chk = self._robot_checks.get(robot.id)
            if chk: chk.setChecked(not chk.isChecked())

    def _update_robot_arrived(self, ugv):
        """Handle mission completion and reset UGV state for the next mission."""

        #Update gui arrived status
        self._last_robot_status[ugv.id] = True
        lbl = self._robot_status_labels.get(ugv.id)
        if lbl is not None:
                    lbl.setText("✅ arrived")
                    lbl.setStyleSheet(
                        "color: green; font-size: 9px;"
                    )

        mission = getattr(ugv, 'assigned_mission', None)

        if mission is not None:
            #  Mark the mission as complete so the planner knows it's done
            mission.status = "complete"
            self._log(f"✅ Mission '{mission.mission_id}' completed by UGV {ugv.id}!")

            # Free the UGV from the overarching twin's perspective
            ugv.assigned_mission = None

        #  Nuke internal goals and stop the robot from wandering (Fallback fix)
        ugv.set_goal(None, init=True)
        ugv.set_velocity([0, 0])

        ugv.goal_threshold = ugv.info.goal_threshold

        #  Refresh the Mission UI panel to show the "complete" status
        self._refresh_mission_list()

    def _update_robot_status(self):
        for robot in self.env.robot_list:
            arrived = getattr(robot, 'arrive_flag', False)

            # Check if the state has changed since the last frame
            if self._last_robot_status.get(robot.id) != arrived:

                # Update our cache
                self._last_robot_status[robot.id] = arrived

                # Perform the expensive UI update ONLY when the state flips
                lbl = self._robot_status_labels.get(robot.id)
                if lbl is not None:
                    lbl.setText("✅ arrived" if arrived else "⏳ moving")
                    lbl.setStyleSheet(
                        "color: green; font-size: 9px;" if arrived
                        else "color: gray; font-size: 9px;"
                    )

                # Handle the text clearing logic if it just arrived
                if arrived:
                    robot.set_text("")

    def _save_robot_run(self):
        robot = self._get_selected_robot()
        if robot is None: return

        # Pop up a dialog asking for a custom name
        custom_name, ok = QInputDialog.getText(
            self,
            "Save Run",
            "Enter custom name (leave blank for default):"
        )

        rid = robot.id

        # Check if the user pressed OK and actually typed something
        if ok and custom_name.strip():
            label = f"{type(robot).__name__}_{rid}_{custom_name.strip()}"
        else:
            ts = datetime.datetime.now().strftime("%H%M%S")
            label = f"{type(robot).__name__}_{rid}_{ts}"

        save_dir = f"./runs/{label}"
        os.makedirs(save_dir, exist_ok=True)

        # Save path snapshot
        xs, ys = (list(v) for v in self._ugv_traces.get(rid, ([], [])))
        self._saved_runs[label] = {"rid": rid, "xs": xs, "ys": ys, "artist": None}

        # Save figures
        ml = getattr(self.adt, '_loggers', {}).get(rid)
        if ml:
            ml.plot_figures(save_dir=save_dir, show=False, file_fmt="pdf")

        # Save path as numpy
        np.save(os.path.join(save_dir, "path.npy"), np.array([xs, ys]))

        # Add to list widget
        from PyQt6.QtCore import Qt
        item = QListWidgetItem(label)
        item.setCheckState(Qt.CheckState.Checked)
        self.saved_runs_list.addItem(item)
        self._log(f"💾 Saved run '{label}' → {save_dir}")
        self._on_saved_run_toggle(item)  # draw immediately

    def _reset_single_robot(self):
        robot = self._get_selected_robot()
        if robot is None: return

        robot.reset()

        robot.set_text(None)  # Restore original name
        robot.goal_threshold = robot.info.goal_threshold # Restore original threshold

        # Reinit just this robot's matplotlib patches
        try:
            self.env.step(action=[np.zeros((2, 1))], action_id=[robot.id])
            robot.plot_clear(all=True)
            robot._init_plot(self.env._env_plot.ax)
        except Exception as e:
            self._log(f"[Reset plot warn] {e}")

        # Clear its trace
        rid = robot.id
        self._ugv_traces[rid] = ([], [])
        self.canvas.draw_idle()
        self._log(f"⏮ Reset robot id={rid}")

    def _on_saved_run_toggle(self, item):
        from PyQt6.QtCore import Qt
        label = item.text()
        run = self._saved_runs.get(label)
        if run is None: return

        # Remove old artist
        old = run.get("artist")
        if old:
            try:
                old.remove()
            except:
                pass
            run["artist"] = None

        if item.checkState() == Qt.CheckState.Checked:
            ax = self.env._env_plot.ax
            xs, ys = run["xs"], run["ys"]
            if xs:
                color = self._run_colors[
                    list(self._saved_runs).index(label) % len(self._run_colors)
                    ]
                line, = ax.plot(xs, ys, '--', color=color, lw=1.5,
                                alpha=0.8, label=label, zorder=6)
                run["artist"] = line
                ax.legend(loc='upper right',  framealpha=0.5)

        self.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════
    # Simulation state
    # ══════════════════════════════════════════════════════════════════════

    def _reset_sim(self):
        self.timer.stop()
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶  Play")
        self._step = 0
        self._use_global_plan = False
        self.lbl_step.setText("Step: 0")

        # Clear per-frame artists
        for a in self._perception_artists:
            try:
                a.remove()
            except:
                pass
        self._perception_artists.clear()

        # Reset UGV position tracking
        self._ugv_traces = {ugv.id: ([], []) for ugv in self.adt.all_ugvs}
        
        for robot in self.env.robot_list:
            robot.set_text(None) # Restore original name
            robot.goal_threshold = robot.info.goal_threshold # Restore original threshold
            if robot.info.goal is not None:
                robot.set_goal(robot.info.goal, init=True)

        # Reset other components
        self.env.reset()
        self.adt.reset()


        self.canvas.draw_idle()
        self._log("⏮ Simulation reset.")

    # ══════════════════════════════════════════════════════════════════════
    # Simulation loop
    # ══════════════════════════════════════════════════════════════════════

    def _sim_step(self):

        # 1. OverArchingTwin tick
        self.adt.step()


        # 2. Local controllers → actions
        actions, ids = [], []
        for ugv in self.adt.ugvs:

            if self._use_global_plan and ugv.assigned_mission is None:
                continue

            waypoints_remaining = len(ugv._goal) if ugv._goal else 0
            if waypoints_remaining <= 1:
                ugv.goal_threshold = 0.1

            if ugv.arrive_flag:
                self.timer.stop()
                self.btn_play.setChecked(False)
                self.btn_play.setText("▶  Play")
                self._log(f"🏁 Robot {ugv.id} at step {self._step}")
                self._update_robot_arrived(ugv)

            obstacles = ugv.get_ugv_view()
            action = self.controllers[ugv.id].get_action(ugv, obstacles)
            actions.append(action)
            ids.append(ugv.id)

        # 3. Physics
        self.env.step(action=actions, action_id=ids)

        for ugv in self.adt.ugvs:
            xs, ys = self._ugv_traces[ugv.id]
            xs.append(float(ugv.state.flat[0]))
            ys.append(float(ugv.state.flat[1]))

        self._env_plot_step(self.env.robot_list)
        self._draw_perception_highlights()
        self.canvas.draw_idle()

        self._update_robot_status()

        if self._step % 15 == 0:
            self._refresh_mission_list()

        # 5. Bookkeeping
        self._step += 1
        self.lbl_step.setText(f"Step: {self._step}")


        if self.env.done():
            self.timer.stop()
            self.btn_play.setChecked(False)
            self.btn_play.setText("▶  Play")
            self._log(f"🏁 Done at step {self._step}")

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

            artists = []

            # Add patches (body, goal marker, arrow, fov wedge)
            artists.extend(getattr(robot, 'plot_patch_list', []))

            # Add text (robot label, goal label)
            artists.extend(getattr(robot, 'plot_text_list', []))

            # Add trails
            artists.extend(getattr(robot, 'plot_trail_list', []))

            # Add lines (trajectory paths are stored as lists inside the list)
            for line_item in getattr(robot, 'plot_line_list', []):
                if isinstance(line_item, list):
                    artists.extend(line_item)
                else:
                    artists.append(line_item)

            # Apply visibility toggle to the robot's own graphics
            for artist in artists:
                try:
                    artist.set_visible(checked)
                except Exception:
                    pass

            # Toggle Sensors (respecting the global sensor toggle state)
            sensor_objs = list(getattr(robot, 'sensors', None) or [])
            single = getattr(robot, 'sensor', None)
            if single is not None and single not in sensor_objs:
                sensor_objs.append(single)

            for sensor in sensor_objs:
                for patch in getattr(sensor, 'plot_patch_list', []):
                    try:
                        patch.set_visible(checked and self._show_sensors)
                    except Exception:
                        pass

            robot.unobstructed = not checked

        self.canvas.draw_idle()

    def _set_all_robots(self, state: bool):
        for chk in self._robot_checks.values():
            chk.setChecked(state)

    def _on_sensor_toggle(self, checked):
        """Toggle sensor footprint patches independently of robot body visibility."""
        self._show_sensors = checked
        for robot in self.env.robot_list:
            robot.plot_kwargs["show_sensor"] = checked
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
            self.map_layer_combo.addItem("Occupancy Grid", ("occ", None, None))
            self.map_layer_combo.addItem("Risk Layer", ("risk", None, None))

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
        W = getattr(gm, '_W', getattr(gm, 'width', 40))
        H = getattr(gm, '_H', getattr(gm, 'height', 40))
        return [ox, ox + W, oy, oy + H]

    def _render_map_layer(self):
        """Render the currently selected map layer onto the Map View canvas."""
        data = self.map_layer_combo.currentData()
        if data is None:
            return

        kind, mission_id, ugv_id = data
        gm = getattr(self.adt, 'grid_map', None)
        mp = getattr(self.adt, 'mission_planner', None)
        if mp is not None:
            ml = getattr(mp, 'mission_logger', None)
        cmap = self.cmap_combo.currentText()

        self._map_fig.clear()  # Clear the whole figure
        self._map_ax = self._map_fig.add_subplot(111)  # Recreate a fresh, full-size axes

        try:
            if kind == "occ" and gm is not None:
                grid = gm._occ
                extent = self._get_grid_extent(gm)
                im = self._map_ax.imshow(
                    grid.T, origin='lower', extent=extent,
                    cmap='gray_r', vmin=0, vmax=100,
                )
                self._map_ax.set_title("Occupancy Grid")
                self._add_map_colorbar(im, "Occupancy (0=free, 100=occ)")

            elif kind == "risk" and gm is not None:
                grid = gm._risk
                extent = self._get_grid_extent(gm)
                im = self._map_ax.imshow(
                    grid.T, origin='lower', extent=extent, cmap=cmap,
                )
                self._map_ax.set_title("Risk Layer")
                self._add_map_colorbar(im, "Risk score")

            elif kind == "cost" and gm is not None and ml is not None:
                from overarching_twin.mission import POSTURE_WEIGHTS

                # Get mission from adt 
                mission = next(
                    (m for m in self.adt.missions if m.mission_id == mission_id), None
                )
                # Get posture from this mission
                posture = (mission.mission_posture)
                print(posture)
                weights = POSTURE_WEIGHTS[posture]
                print(weights)

                # Safely find the corresponding UGV object in the OverArchingTwin
                ugv = next((u for u in self.adt.ugvs if getattr(u, 'id', str(id(u))) == ugv_id), None)

                cost_img = gm.get_cost_image(
                    weights=weights,
                    robot_mass=ugv.mass,
                    v_avg=ugv.avg_speed,
                    Ka=ugv.ancillary_drain)

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
                                self._map_ax.plot(path[0][0], path[1][0], 'rx', ms=7)
                            break
                if (ml._per_mission_assignement_log[mission_id][0]["ugv_id"] == ugv_id):
                    assigned_str = ": [Winner]"
                else:
                    assigned_str = " - [Loser]"

                self._map_ax.set_title(
                    f"Cost Map - Mission: {mission_id} - UGV: {ugv_id} - Cost:{entry['cost']:.2f} \n[{posture}]" + assigned_str)
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
        ax = self.env._env_plot.ax  # Simulation axes

        # Clear previous elements
        if hasattr(self, '_overlay_elements'):
            for item in self._overlay_elements:
                try:
                    item.remove()
                except:
                    pass
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
                posture = mission.mission_posture

                weights = POSTURE_WEIGHTS[posture]
                ugv = next((u for u in self.adt.ugvs if getattr(u, 'id', None) == ugv_id), None)

                if ugv:
                    img_data = gm.get_cost_image(weights=weights, robot_mass=ugv.mass,
                                                 v_avg=ugv.avg_speed, Ka=ugv.ancillary_drain).T

            if img_data is not None:
                im = ax.imshow(img_data, origin='lower', extent=extent,
                               cmap=self.cmap_combo.currentText(),
                               alpha=alpha_val, zorder=2)  # Base layer
                self._overlay_elements.append(im)

            # draw path
            if ml is not None and mission_id in ml._per_mission_log:
                logs = ml._per_mission_log[mission_id]
                for entry in logs:

                    if str(entry['id']) == str(ugv_id):  # Force string comparison to be safe
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

    def _on_arrows_toggle(self, checked):
        """
        Toggles the 'show_arrow' property for all obstacles and refreshes the plot.
        """
        # Loop through all obstacles in the environment
        for obj in self.env.obstacle_list:
            # Update the internal plotting kwargs
            obj.plot_kwargs["show_arrow"] = checked

        # Re-initialize the plot to apply the changes
        # This clears old components and rebuilds them with the new kwargs
        self.env.reset_plot()

        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _on_border_toggle(self, checked):
        self._show_borders = checked
        # force redraw
        self._draw_perception_highlights()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _draw_perception_highlights(self):
        # Fast cleanup
        for a in self._perception_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._perception_artists.clear()

        ax = self.env._env_plot.ax
        gm = getattr(self.adt, 'grid_map', None)
        if gm is None: return

        # O(1) Lookups: Pre-compute sets outside the loop to avoid redundant calculations
        try:
            uav_obs_ids = {o.id for o in self.adt.get_uavs_view()}
        except Exception:
            uav_obs_ids = set()

        try:
            ugv_obs_ids = {o.id for o in self.adt.get_ugvs_view()}
        except Exception:
            ugv_obs_ids = set()


        fleet = getattr(self.adt, 'uav_fleet', None)
        fleet_hidden_ids = {getattr(o, 'id', o) for o in getattr(fleet, 'hidden_objects', [])}

        # Convert list to a set of IDs for O(1) lookup speed
        adt_perceived = getattr(self.adt, 'perceived_obstacles', [])
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
                color, lw, facecolor, alpha, ls = "#A3A0A0", 1.5, "#9F9A9A", 0.2, ':'
            elif is_fleet_hidden:
                color, lw, facecolor, alpha, ls = 'red', 2.0, 'red', 0.4, '-.'

            elif self._show_borders:
                if in_uav and in_ugv:
                    color, lw, facecolor, alpha, ls = 'white', 1.5, 'none', 1.0, '--'
                elif in_uav:
                    color, lw, facecolor, alpha, ls = 'cyan', 2.0, 'none', 1.0, '--'
                elif in_ugv:
                    color, lw, facecolor, alpha, ls = 'orange', 2.0, 'none', 1.0, '--'
                elif in_adt:
                    color, lw, facecolor, alpha, ls = 'yellow', 1.5, 'none', 1.0, '--'
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

        # Batch Render: Draw everything at once via PatchCollection
        if patches:
            collection = PatchCollection(
                patches,
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                linestyles=linestyles,
                zorder=6,
                match_original=False  # Forces collection to use our explicit arrays
            )
            ax.add_collection(collection)
            self._perception_artists.append(
                collection) 

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
            name = self.spawn_name_edit.text().strip() or "robot_new"
            rtype = self.spawn_type_combo.currentText()  # "ugv" | "uav" | "robot"
            kin = self.spawn_kin_combo.currentText()  # "diff" | "omni" | "acker"
            x = self.spawn_x_spin.value()
            y = self.spawn_y_spin.value()
            theta = self.spawn_theta_spin.value()
            gx = self.spawn_goal_x_spin.value()
            gy = self.spawn_goal_y_spin.value()

            robot = self.env.object_factory.create_robot(
                type=rtype,
                kinematics={"name": kin},
                state=[x, y, theta],
                goal=[gx, gy, 0.0],
                name=name,
            )

            # add_object: sets _env, calls _init_plot + _step_plot, rebuilds tree
            self.env.add_object(robot)

            # Wire into twin lists so adt / controllers can pick it up
            from irsim.world.robots.uav_twin import UAVTwin
            from irsim.world.robots.ugv_twin import UGVTwin

            if isinstance(robot, UGVTwin):
                self.adt.add_ugv(robot)
                # Add a default controller (same type as first UGV if available)
                if self.controllers:
                    self.controllers[robot.id] = CollisionConeCBFController(
                        robot_type=robot.kinematics, safety_margin=0.05, goal_gain=0.8
                    )
            elif isinstance(robot, UAVTwin):
                self.adt.uav_fleet.add_uav(robot)

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
        robot = self._get_selected_robot()
        if robot is None:
            self._log(f"[Delete] Robot id={robot.id} not found")
            return

        rid = robot.id

        # env.delete_object handles plot_clear + _objects removal + build_tree
        self.env.delete_object(rid)
        self.adt.remove_ugv(robot)
        self.adt.uav_fleet.remove_uav(robot)
        self.controllers.pop(rid, None)
        self._visibility_robots.pop(rid, None)

        # Remove checkbox row
        chk = self._robot_checks.pop(rid, None)
        if chk:
            # chk is the QCheckBox; its parent is the row QWidget
            row = chk.parent()
            if row:
                row.deleteLater()
            else:
                chk.deleteLater()

        self._robot_status_labels.pop(rid, None)
        self._refresh_delete_combo()
        self._draw_perception_highlights()
        self.canvas.draw_idle()
        self._log(f"🗑 Deleted robot id={rid}")


    def _delete_obstacle(self, obs):
        self._visibility_objects.pop(obs.id)
        self.adt.remove_perceived_obstacle(obs)
        self.env.delete_object(obs.id)

        self._draw_perception_highlights()
        self.canvas.draw_idle()
        self._log(f"🗑 Deleted object id={obs.id}")



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
        self.act_combo.clear()
        for r in self.env.robot_list:
            label = f"{type(r).__name__}  [id={r.id}]  '{getattr(r, 'name', '')}'"
            self.act_combo.addItem(label, r.id)

    # ══════════════════════════════════════════════════════════════════════
    # Mission controls
    # ══════════════════════════════════════════════════════════════════════

    def _add_mission(self):
        try:
            mid = self.mission_id_edit.text().strip() or "mission_dyn"
            mtype = MissionType[self.mission_type_combo.currentText()]
            gx = self.goal_x_spin.value()
            gy = self.goal_y_spin.value()
            posture = self.posture_combo.currentText()

            mission = Mission(
                mission_id=mid,
                mission_type=mtype,
                goal_xy=(gx, gy),
                mission_posture=posture,
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
# Custom Classes
# ══════════════════════════════════════════════════════════════════════════════


class CustomNavigationToolbar(NavigationToolbar2QT):

    def __init__(self, canvas, parent, custom_save_callback=None):
        super().__init__(canvas, parent)
        self.custom_save_callback = custom_save_callback

    # toolitems needed to display
    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),  # <-- Uncommented!
    )

    # called when save is clicked
    def save_figure(self, *args):
        if self.custom_save_callback:
            self.custom_save_callback()
        else:
            # Fallback to default Matplotlib save dialog if no callback was provided
            super().save_figure(*args)

class ExportSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Settings")

        layout = QVBoxLayout(self)

        # 1. Custom Super Title Input
        layout.addWidget(QLabel("Enter a title for the figure (leave blank for no title):"))
        self.title_input = QLineEdit(self)
        layout.addWidget(self.title_input)

        # Checkbox to toggle the existing axis title
        self.show_ax_title_cb = QCheckBox("Include simulation time/status title", self)
        self.show_ax_title_cb.setChecked(True)  # Default to keeping it
        layout.addWidget(self.show_ax_title_cb)

        # Ok / Cancel Buttons (PyQt6 Strict Enum Syntax)
        buttons = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.button_box = QDialogButtonBox(buttons, self)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_values(self):
        """Returns a tuple: (custom_title_text, keep_axis_title_boolean)"""
        return self.title_input.text(), self.show_ax_title_cb.isChecked()
    
class ObjectSpawnDialog(QDialog):
    """A flexible dialog for spawning circles, rectangles, or robots."""
    def __init__(self, spawn_type, parent=None):
        super().__init__(parent)
        self.spawn_type = spawn_type
        self.setWindowTitle(f"Spawn {spawn_type.capitalize()}")
        
        self.layout = QFormLayout(self)
        self.inputs = {}

        if spawn_type == "circle":
            self._add_spinbox("Radius [m]:", "radius", 1.0, 0.1, 20.0, 0.1)
        
        elif spawn_type == "rectangle":
            self._add_spinbox("Length (X) [m]:", "length", 2.0, 0.1, 50.0, 0.5)
            self._add_spinbox("Width (Y) [m]:", "width", 2.0, 0.1, 50.0, 0.5)
            self._add_spinbox("Orientation θ [rad]:", "theta", 0.0, -3.15, 3.15, 0.1)
            
        elif spawn_type == "robot":
            # Name
            self.inputs["name"] = QLineEdit(f"robot_{np.random.randint(100,999)}")
            self.layout.addRow("Name:", self.inputs["name"])
            # Type
            self.inputs["type"] = QComboBox()
            self.inputs["type"].addItems(["ugv", "uav", "robot"])
            self.layout.addRow("Type:", self.inputs["type"])
            # Kinematics
            self.inputs["kin"] = QComboBox()
            self.inputs["kin"].addItems(["diff", "omni", "acker"])
            self.layout.addRow("Kinematics:", self.inputs["kin"])
            # Orientation
            self._add_spinbox("Orientation θ [rad]:", "theta", 0.0, -3.15, 3.15, 0.1)

        # Ok / Cancel Buttons
        buttons = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.button_box = QDialogButtonBox(buttons, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def _add_spinbox(self, label, key, default, min_v, max_v, step):
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        spin.setSingleStep(step)
        self.inputs[key] = spin
        self.layout.addRow(label, spin)

    def get_values(self):
        """Returns a dict of all inputted values."""
        vals = {}
        for k, widget in self.inputs.items():
            if isinstance(widget, QDoubleSpinBox):
                vals[k] = widget.value()
            elif isinstance(widget, QComboBox):
                vals[k] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                vals[k] = widget.text().strip()
        return vals

# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def launch(env, adt, controllers,
           max_steps: int = 800, step_ms: int = 100,
           perception_mode: str = "ugv"):
    """
    Drop-in replacement for the manual for-loop in custom_world.py.
    Blocks until the Qt window is closed, then returns so post-run
    logging (mission_logger, metric_logger) can execute normally.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    gui = SimulationGUI(
        env=env,
        adt=adt,
        controllers=controllers,
        max_steps=max_steps,
        step_ms=step_ms,
        perception_mode=perception_mode,
    )
    gui.show()
    app.exec()  # blocks here; returns when window is closed
