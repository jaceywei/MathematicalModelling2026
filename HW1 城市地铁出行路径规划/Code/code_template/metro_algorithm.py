# Copyright 2026, Yumeng Liu @ USTC

"""
地铁网络算法模块 —— 数据加载、图构建、Dijkstra 求解
"""

import csv
import heapq
from pathlib import Path

import numpy as np


# ============================================================
# Graph 数据结构
# ============================================================

class Graph:
    """
    简单的无向加权图。

    需要实现的接口
    -------------
    - add_node(node_id, **attrs) : 添加节点
    - add_edge(u, v, weight)     : 添加无向边
    - neighbors(node_id)         : 返回邻居字典 {neighbor_id: weight}
    - number_of_nodes()          : 返回节点数
    - number_of_edges()          : 返回边数
    - edges()                    : 返回所有边列表 [(u, v, weight), ...]

    属性
    ----
    nodes : dict[int, dict]
        节点字典，{node_id: {"name": str, ...}}。
        GUI 会读取此属性来获取节点信息，请确保 add_node 时正确填充。

    提示
    ----
    你可以自由选择底层数据结构（邻接表、邻接矩阵、边列表等）。
    """

    def __init__(self):
        self.nodes = {}
        self.adj = {}  # 邻居表
        self.totaledge = 0  # 边数统计

    def add_node(self, node_id, **attrs):
        """
        添加节点。

        Parameters
        ----------
        node_id : int
            节点编号。
        **attrs
            节点属性，例如 name="StationA"。
        """

        self.nodes[node_id] = attrs
        self.adj[node_id] = {}  # 初始化邻居字典

    def add_edge(self, u, v, weight=1.0):
        """
        添加无向边 (u, v)，权重为 weight。
        """

        if u not in self.nodes or v not in self.nodes:
            raise ValueError("Both nodes must exist before adding an edge.")
        if v in self.adj[u]:
            raise ValueError("Edge already exists.")
        
        self.adj[u][v] = weight
        self.adj[v][u] = weight
        self.totaledge += 1  # 统计边数

    def neighbors(self, node_id):
        """
        返回 node_id 的邻居字典 {neighbor_id: weight}。

        若节点不存在或无邻居，返回空字典。
        """

        return self.adj[node_id] if node_id in self.adj else {}

    def number_of_nodes(self):
        """返回图中节点数量。"""

        return len(self.nodes)

    def number_of_edges(self):
        """返回图中边的数量（每条无向边只计一次）。"""

        return self.totaledge

    def edges(self):
        """
        返回所有边的列表 [(u, v, weight), ...]，每条边只出现一次。

        GUI 的绘图函数会调用此方法来绘制网络边。
        """
        # 只需返回 u < v 的边，避免重复

        edge_list = []
        for u in self.adj:
            for v, w in self.adj[u].items():
                if u < v:
                    edge_list.append((u, v, w))
        return edge_list


# ============================================================
# 数据加载
# ============================================================

def load_station_map(tsv_path: str) -> dict[int, str]:
    """读取 station-id-map.tsv，返回 {id: name} 映射。"""
    stations: dict[int, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stations[int(row["id"])] = row["name"]
    return stations


def load_adjacency_matrix(csv_path: str) -> np.ndarray:
    """读取 adjacency-distance.csv，返回 N×N numpy 矩阵。"""
    return np.loadtxt(csv_path, delimiter=",")


def build_graph(stations: dict[int, str], adj: np.ndarray) -> Graph:
    """
    根据站点映射和邻接矩阵构建加权图。

    Parameters
    ----------
    stations : dict[int, str]
        站点 id → 名称映射（id 从 1 开始）。
    adj : np.ndarray
        N×N 邻接距离矩阵，adj[i,j] > 0 表示站点 i+1 与 j+1 之间有边。

    Returns
    -------
    Graph
        带权无向图，节点属性 name 为站名，边权 weight 为距离。

    提示
    ----
    - 使用 Graph.add_node(node_id, name=...) 添加节点
    - 使用 Graph.add_edge(u, v, weight=...) 添加边
    - 矩阵下标从 0 开始，站点 id 从 1 开始
    """
    
    g = Graph()
    for stationid, name in stations.items():
        g.add_node(stationid, name=name)
    
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                g.add_edge(i+1, j+1, weight=adj[i, j])

    return g


# ============================================================
# Dijkstra 最短路径
# ============================================================

def dijkstra(G: Graph, src: int, dst: int) -> tuple[float, list[int]]:
    """
    实现 Dijkstra 求 src → dst 最短路径。

    Parameters
    ----------
    G : Graph
        带权图。
    src : int
        起点站点 id。
    dst : int
        终点站点 id。

    Returns
    -------
    (cost, path) : (float, list[int])
        cost 为最短距离，path 为站点 id 序列（含起终点）。
        若不可达，返回 (float("inf"), [])。

    提示
    ----
    - 使用 G.neighbors(u) 获取邻居字典 {neighbor_id: weight}
    - 使用 heapq 实现最小堆
    - 使用前驱字典 prev 回溯路径
    """

    # 初始化
    dist = {src: 0.0}  # src 到各节点的距离
    prev = {src: None}  # 前驱节点字典
    heap = [(0.0, src)]  # 最小堆，元素为 (distance, node_id)

    while heap:
        # 1. 取出并删除距离最小的节点
        d, u = heapq.heappop(heap)
        
        # 2. 如果这个距离比已知最短距离大，说明是旧记录，跳过
        if d > dist.get(u, float('inf')):
            continue
        
        # 3. 如果到达终点，结束
        if u == dst:
            break
        
        # 4. 遍历 u 的所有邻居
        for v, weight in G.neighbors(u).items():
            new_dist = d + weight
            
            # 5. 如果找到更短的路径或原本没有进入heap的节点，更新并入堆
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    # 检查是否可达
    if dst not in dist:
        return (float('inf'), [])
    
    # 回溯路径
    path = []
    current = dst
    while current is not None:
        path.append(current)
        current = prev.get(current)
    path.reverse()
    
    return (dist[dst], path)


# ============================================================
# MetroSystem 高层封装
# ============================================================

class MetroSystem:
    """封装单个城市的地铁系统：加载数据、构建图、求解路径。"""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.city = data_dir.name

        tsv = next(data_dir.glob("*station-id-map.tsv"))
        csv_f = next(data_dir.glob("*adjacency-distance.csv"))

        self.stations = load_station_map(str(tsv))
        adj = load_adjacency_matrix(str(csv_f))
        self.graph = build_graph(self.stations, adj)

        self.name_to_id: dict[str, int] = {
            name: sid for sid, name in self.stations.items()
        }

    def sorted_station_names(self) -> list[str]:
        """返回按字母排序的站名列表。"""
        return sorted(self.stations.values())

    def shortest_path(self, src_name: str, dst_name: str) -> tuple[float, list[int]]:
        """
        求两站之间的最短路径。

        Parameters
        ----------
        src_name : str
            起点站名。
        dst_name : str
            终点站名。

        Returns
        -------
        (cost, path) : (float, list[int])
            cost 为最短距离 (km)，path 为站点 id 序列。

        提示
        ----
        - 使用 self.name_to_id 将站名转为 id
        - 调用 dijkstra(self.graph, src_id, dst_id)
        """
        
        src = self.name_to_id.get(src_name)
        dst = self.name_to_id.get(dst_name)
        if src is None or dst is None:
            raise ValueError("Invalid station name.")
        return dijkstra(self.graph, src, dst)


def detect_cities(data_root: str | Path) -> list[str]:
    """扫描 data_root 下所有包含数据文件的城市子目录。"""
    data_root = Path(data_root)
    cities: list[str] = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and list(d.glob("*adjacency-distance.csv")):
            cities.append(d.name)
    return cities
