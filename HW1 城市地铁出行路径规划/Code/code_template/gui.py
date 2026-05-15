# Copyright 2026, Yumeng Liu @ USTC
"""
地铁路径规划 —— 交互式 GUI 模块

基于 Tkinter + Matplotlib，提供：
  - 城市选择 → 绘制全网络
  - 起终站下拉选择 → 即时高亮
  - 求解按钮 → Dijkstra 最短路径高亮 + 文本输出
"""


import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from metro_algorithm import Graph, MetroSystem, detect_cities

matplotlib.use("TkAgg")


# ============================================================
# 布局算法（Fruchterman-Reingold 弹簧模型）
# ============================================================

def spring_layout(graph: Graph, seed: int = 42, iterations: int = 80) -> dict[int, tuple[float, float]]:
    """
    计算图的 Fruchterman-Reingold 弹簧布局。

    返回 {node_id: (x, y)} 坐标字典，坐标归一化到 [0.05, 0.95]。
    """
    node_ids = list(graph.nodes.keys())
    n = len(node_ids)
    if n == 0:
        return {}

    rng = np.random.RandomState(seed)
    pos = rng.rand(n, 2)
    idx = {nid: i for i, nid in enumerate(node_ids)}

    k = 1.0 / np.sqrt(n)
    temp = 1.0
    edge_list = graph.edges()

    for _ in range(iterations):
        disp = np.zeros((n, 2))

        # 斥力：所有节点对
        for i in range(n):
            diff = pos[i] - pos
            dist = np.sqrt((diff ** 2).sum(axis=1))
            dist = np.clip(dist, 0.001, None)
            force = k * k / dist
            force[i] = 0.0
            disp[i] += (diff * force[:, np.newaxis]).sum(axis=0)

        # 引力：沿边
        for u, v, _w in edge_list:
            i, j = idx[u], idx[v]
            diff = pos[i] - pos[j]
            dist = max(np.sqrt((diff ** 2).sum()), 0.001)
            f = dist / k
            disp[i] -= diff * f / dist
            disp[j] += diff * f / dist

        # 更新
        mag = np.sqrt((disp ** 2).sum(axis=1))
        mag = np.clip(mag, 0.001, None)
        pos += disp * np.minimum(temp, mag)[:, np.newaxis] / mag[:, np.newaxis]
        temp *= 0.95

    # 归一化到 [0.05, 0.95]
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    span = hi - lo
    span = np.where(span < 1e-6, 1.0, span)
    pos = (pos - lo) / span * 0.9 + 0.05

    return {nid: (pos[i, 0], pos[i, 1]) for i, nid in enumerate(node_ids)}


# ============================================================
# GUI 主类
# ============================================================

class MetroApp:
    """基于 Tkinter 的交互式地铁路径规划界面。"""

    BG_COLOR = "#f5f5f5"
    EDGE_COLOR = "#bdbdbd"
    NODE_COLOR = "#90caf9"
    NODE_EDGE_COLOR = "#1565c0"
    SRC_COLOR = "#2e7d32"
    DST_COLOR = "#e65100"
    PATH_EDGE_COLOR = "#e53935"
    PATH_NODE_COLOR = "#ffcdd2"
    PATH_NODE_EDGE = "#b71c1c"
    SIDEBAR_WIDTH = 320

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.cities = detect_cities(self.data_root)

        self.metro: MetroSystem | None = None
        self.pos: dict[int, tuple[float, float]] = {}

        self._build_ui()

    # ================================================================
    # UI 构建
    # ================================================================

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Metro Shortest Path Finder")
        self.root.configure(bg=self.BG_COLOR)
        self.root.geometry("1400x850")

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0, minsize=self.SIDEBAR_WIDTH)
        self.root.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor=self.BG_COLOR)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(self.root, width=self.SIDEBAR_WIDTH)
        sidebar.grid(row=0, column=1, sticky="nsew", padx=(0, 5), pady=5)
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

    def _build_sidebar(self, sidebar: ttk.Frame):
        px, py = 10, 6

        sec = ttk.LabelFrame(sidebar, text="City", padding=8)
        sec.pack(fill=tk.X, padx=px, pady=py)
        self.city_var = tk.StringVar()
        cb = ttk.Combobox(sec, textvariable=self.city_var,
                          values=self.cities, state="readonly")
        cb.pack(fill=tk.X)
        cb.bind("<<ComboboxSelected>>", self._on_city_selected)

        sec = ttk.LabelFrame(sidebar, text="Station Selection", padding=8)
        sec.pack(fill=tk.X, padx=px, pady=py)

        ttk.Label(sec, text="From:").pack(anchor=tk.W)
        self.src_var = tk.StringVar()
        self.src_cb = ttk.Combobox(sec, textvariable=self.src_var, state="readonly")
        self.src_cb.pack(fill=tk.X, pady=(0, 8))
        self.src_cb.bind("<<ComboboxSelected>>", self._on_station_selected)

        ttk.Label(sec, text="To:").pack(anchor=tk.W)
        self.dst_var = tk.StringVar()
        self.dst_cb = ttk.Combobox(sec, textvariable=self.dst_var, state="readonly")
        self.dst_cb.pack(fill=tk.X, pady=(0, 8))
        self.dst_cb.bind("<<ComboboxSelected>>", self._on_station_selected)

        btn_frame = ttk.Frame(sec)
        btn_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_frame, text="Find Shortest Path",
                   command=self._on_solve).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(btn_frame, text="Reset",
                   command=self._on_reset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        out = ttk.LabelFrame(sidebar, text="Route Output", padding=8)
        out.pack(fill=tk.BOTH, expand=True, padx=px, pady=py)
        self.output_text = tk.Text(out, wrap=tk.WORD, font=("Consolas", 11),
                                   bg="#fafafa", relief=tk.FLAT, padx=8, pady=6)
        scrollbar = ttk.Scrollbar(out, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    # ================================================================
    # 事件处理
    # ================================================================

    def _on_city_selected(self, _event=None):
        city = self.city_var.get()
        self.metro = MetroSystem(self.data_root / city)

        names = self.metro.sorted_station_names()
        self.src_cb["values"] = names
        self.dst_cb["values"] = names
        self.src_var.set("")
        self.dst_var.set("")

        n_nodes = self.metro.graph.number_of_nodes()
        if n_nodes > 0:
            self.pos = spring_layout(self.metro.graph)
        else:
            self.pos = {}

        self._draw_base()
        self._log("Loaded {}: {} stations, {} edges".format(
            city, len(self.metro.stations), self.metro.graph.number_of_edges()))
        if n_nodes == 0:
            self._log("  [Note] Graph is empty — build_graph() not yet implemented?")

    def _on_station_selected(self, _event=None):
        if self.metro is None:
            return
        self._draw_base()
        self._highlight_endpoints()
        self.canvas.draw_idle()

    def _on_solve(self):
        if self.metro is None:
            return
        src_name = self.src_var.get()
        dst_name = self.dst_var.get()
        if not src_name or not dst_name:
            self._log("Please select both a source and a destination station.")
            return
        if src_name == dst_name:
            self._log("Source and destination are the same station.")
            return

        cost, path = self.metro.shortest_path(src_name, dst_name)
        if not path:
            self._log("No path found from {} to {}.".format(src_name, dst_name))
            if self.metro.graph.number_of_nodes() == 0:
                self._log("  [Hint] build_graph() or dijkstra() not yet implemented?")
            return

        self._draw_base()
        self._draw_path(path, cost)
        self.canvas.draw_idle()

        names = [self.metro.stations[nid] for nid in path]
        self._log(
            "{line}\n"
            "  {src}  ->  {dst}\n"
            "  Distance : {cost:.3f} km\n"
            "  Stops    : {stops}\n"
            "  Route    : {route}\n"
            "{line}".format(
                line="-" * 40, src=src_name, dst=dst_name,
                cost=cost, stops=len(path), route=" -> ".join(names),
            )
        )

    def _on_reset(self):
        self.src_var.set("")
        self.dst_var.set("")
        if self.metro is not None:
            self._draw_base()
            self.canvas.draw_idle()
        self.output_text.delete("1.0", tk.END)

    def _draw_base(self):
        """绘制底层地铁网络。"""
        self.ax.clear()
        self.ax.set_facecolor(self.BG_COLOR)

        if not self.pos:
            self.ax.set_title("{} Metro Network".format(
                self.metro.city if self.metro else ""),
                fontsize=14, fontweight="bold")
            self.ax.axis("off")
            self.fig.tight_layout()
            self.canvas.draw_idle()
            return

        G = self.metro.graph

        # 边
        segments = []
        for u, v, _w in G.edges():
            if u in self.pos and v in self.pos:
                segments.append([self.pos[u], self.pos[v]])
        if segments:
            lc = LineCollection(segments, colors=self.EDGE_COLOR,
                                linewidths=0.8, alpha=0.6)
            self.ax.add_collection(lc)

        # 节点
        xs = [self.pos[nid][0] for nid in G.nodes if nid in self.pos]
        ys = [self.pos[nid][1] for nid in G.nodes if nid in self.pos]
        self.ax.scatter(xs, ys, s=25, c=self.NODE_COLOR,
                        edgecolors=self.NODE_EDGE_COLOR, linewidths=0.4, zorder=3)

        self.ax.set_title("{} Metro Network".format(self.metro.city),
                          fontsize=14, fontweight="bold")
        self.ax.axis("off")
        self.ax.set_xlim(-0.02, 1.02)
        self.ax.set_ylim(-0.02, 1.02)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _annotate_station(self, nid, color, marker, label, size=180):
        if nid not in self.pos:
            return
        x, y = self.pos[nid]
        self.ax.scatter(x, y, s=size, c=color, marker=marker,
                        zorder=5, edgecolors="white", linewidths=2)
        self.ax.annotate(
            "[{}] {}".format(label, self.metro.stations[nid]), (x, y),
            textcoords="offset points", xytext=(8, 8),
            fontsize=8, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      alpha=0.85, ec=color, lw=0.8),
        )

    def _highlight_endpoints(self):
        for var, color, marker, label in [
            (self.src_var, self.SRC_COLOR, "o", "From"),
            (self.dst_var, self.DST_COLOR, "s", "To"),
        ]:
            name = var.get()
            if name:
                nid = self.metro.name_to_id.get(name)
                if nid is not None and nid in self.pos:
                    self._annotate_station(nid, color, marker, label)

    def _draw_path(self, path: list[int], cost: float):
        """在底层网络上绘制最短路径高亮。"""
        stations = self.metro.stations

        # 高亮边
        segments = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in self.pos and v in self.pos:
                segments.append([self.pos[u], self.pos[v]])
        if segments:
            lc = LineCollection(segments, colors=self.PATH_EDGE_COLOR,
                                linewidths=3.0, alpha=0.9, zorder=4)
            self.ax.add_collection(lc)

        # 高亮节点
        pxs = [self.pos[nid][0] for nid in path if nid in self.pos]
        pys = [self.pos[nid][1] for nid in path if nid in self.pos]
        self.ax.scatter(pxs, pys, s=70, c=self.PATH_NODE_COLOR,
                        edgecolors=self.PATH_NODE_EDGE, linewidths=1.5, zorder=4)

        # 路径站名标签
        for nid in path:
            if nid in self.pos:
                x, y = self.pos[nid]
                self.ax.text(x, y, stations[nid], fontsize=6, fontweight="bold",
                             color=self.PATH_NODE_EDGE, ha="center", va="bottom",
                             transform=self.ax.transData)

        # 起终点标记
        self._annotate_station(path[0], self.SRC_COLOR, "o", "From", size=220)
        self._annotate_station(path[-1], self.DST_COLOR, "s", "To", size=220)

        self.ax.set_title(
            "{city} Metro Network\n"
            "Shortest Path: {src} -> {dst}  (distance = {cost:.3f} km)".format(
                city=self.metro.city,
                src=stations[path[0]], dst=stations[path[-1]], cost=cost),
            fontsize=13, fontweight="bold",
        )

    # ================================================================
    # 工具
    # ================================================================

    def _log(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def run(self):
        self.root.mainloop()
