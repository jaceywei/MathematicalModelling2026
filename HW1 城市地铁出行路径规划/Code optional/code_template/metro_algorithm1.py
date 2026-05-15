# Copyright 2026, Yumeng Liu @ USTC

"""
地铁网络算法模块 —— 数据加载、图构建、Dijkstra 求解 (支持换乘)
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
    简单的无向加权图。实现同之前。
    """
    def __init__(self):
        self.nodes = {}
        self.adj = {}
        self.totaledge = 0

    def add_node(self, node_id, **attrs):
        self.nodes[node_id] = attrs
        self.adj[node_id] = {}

    def add_edge(self, u, v, weight=1.0):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError("Both nodes must exist before adding an edge.")
        self.adj[u][v] = weight
        self.adj[v][u] = weight
        self.totaledge += 1

    def neighbors(self, node_id):
        return self.adj[node_id] if node_id in self.adj else {}

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return self.totaledge

    def edges(self):
        edge_list = []
        for u in self.adj:
            for v, w in self.adj[u].items():
                if u < v:
                    edge_list.append((u, v, w))
        return edge_list


# ============================================================
# 数据加载 (新增线路解析)
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


def load_station_lines(txt_path: str) -> dict[str, list[str]]:
    """
    读取 station-lines.txt，返回 {station_name: [line_names]} 映射。
    
    格式示例：
    AgriculturalExhibitionCenter	Line10
    BeijingSouthRailwayStation	Line4,Line14
    """
    station_lines: dict[str, list[str]] = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            name, lines_str = parts
            lines = [l.strip() for l in lines_str.split(",")]
            station_lines[name] = lines
    return station_lines

def build_graphs(
    stations: dict[int, str], 
    adj: np.ndarray, 
    station_lines: dict[str, list[str]]
) -> tuple[Graph, Graph]:
    """
    同时构建物理图(仅含物理站点)和换乘图(拆点后的图)。
    
    策略：
    1. 严格检查 stations 和 station_lines 的一致性。
    2. 物理图节点属性中新增 'transfer_nodes'，存储其对应的所有逻辑节点 ID，方便后期多源/多目标 Dijkstra。
    3. 遍历邻接矩阵时，同时为物理图和换乘图添加行车边。
    
    Returns
    -------
    tuple[Graph, Graph]
        (物理图 phys_graph, 换乘图 trans_graph)
    """
    phys_graph = Graph()
    trans_graph = Graph()
    
    # 1. 严格的数据一致性检查

    station_id_lines = {}
    for sid, name in stations.items():
        lines = station_lines.get(name, [])
        if not lines:
            raise ValueError(f"Data inconsistency: Station '{name}' (ID {sid}) is in stations list but has no line information.")
        station_id_lines[sid] = lines
        
    # 反向检查：station_lines 里有没有 stations 字典里不存在的废弃站点
    phys_station_names = set(stations.values())

    # 注意读取的时候会读到列标题，第一次运行时报错了
    station_lines.pop('station', None)
    station_lines.pop('name', None)

    for name in station_lines.keys():
        if name not in phys_station_names:
            raise ValueError(f"Data inconsistency: Station '{name}' is in lines data but missing from stations list.")

    # 2. 创建节点 & 建立 Link (物理节点 -> 逻辑节点)

    id_map = {} # (sid, line) -> trans_node_id
    new_id_counter = 1
    
    for sid, name in stations.items():
        lines = station_id_lines[sid]
        
        # 收集当前物理站点对应的所有逻辑节点 ID
        linked_trans_nodes = [] 
        
        for line in lines:
            t_node_id = new_id_counter
            new_id_counter += 1
            
            id_map[(sid, line)] = t_node_id
            linked_trans_nodes.append(t_node_id)
            
            # 换乘图添加拆分后的逻辑节点
            display_name = f"{name} {line}"
            trans_graph.add_node(t_node_id, name=display_name, original_id=sid, line=line)
            
        # 物理图添加节点，并把 linked_trans_nodes 存进去
        phys_graph.add_node(sid, name=name, transfer_nodes=linked_trans_nodes)

    # 3. 为换乘图添加“换乘边” (同一站点的不同线路间)

    TRANSFER_WEIGHT = 2.917
    for sid, lines in station_id_lines.items():
        if len(lines) > 1:
            # 取出该物理站对应的所有逻辑节点，构造完全图
            node_ids = [id_map[(sid, line)] for line in lines]
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    trans_graph.add_edge(node_ids[i], node_ids[j], weight=TRANSFER_WEIGHT)

    # 4. 同时为两个图添加“行车边” (基于邻接矩阵)

    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dist = adj[i, j]
            if dist > 0:
                sid_u = i + 1
                sid_v = j + 1
                
                # 给物理图加上行车边
                phys_graph.add_edge(sid_u, sid_v, weight=dist)
                
                # 给换乘图加上行车边
                lines_u = set(station_id_lines[sid_u])
                lines_v = set(station_id_lines[sid_v])
                common_lines = lines_u & lines_v
                
                if not common_lines:
                    raise ValueError(f"Data inconsistency: Stations ID {sid_u} and {sid_v} have distance {dist} but no common line.")
                
                for line in common_lines:
                    u_node = id_map[(sid_u, line)]
                    v_node = id_map[(sid_v, line)]
                    trans_graph.add_edge(u_node, v_node, weight=dist)

    return phys_graph, trans_graph

# ============================================================
# Dijkstra 最短路径 (终点变成数组)
# ============================================================

def dijkstra(G, src: int, targets: set[int]) -> tuple[float, list[int]]:
    """
    实现 Dijkstra 求 src → targets 中任意一个节点的最短路径。
    一旦触碰 targets 中的任意一个节点，搜索即刻停止。
    """
    
    dist = {src: 0.0}
    prev = {src: None}
    heap = [(0.0, src)]
    
    # 用于记录最终实际到达的是哪一个目标节点
    reached_dst = None 

    while heap:
        d, u = heapq.heappop(heap)
        
        if d > dist.get(u, float('inf')):
            continue
            
        if u in targets:
            reached_dst = u
            break
            
        for v, weight in G.neighbors(u).items():
            new_dist = d + weight
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    # 如果队列空了还没找到任何一个目标节点，说明不连通
    if reached_dst is None:
        return (float('inf'), [])
    
    # 路径回溯（从实际到达的那个目标节点开始往回找）
    path = []
    current = reached_dst
    while current is not None:
        path.append(current)
        current = prev.get(current)
    path.reverse()
    
    return (dist[reached_dst], path)

# ============================================================
# MetroSystem 高层封装 (支持换乘)
# ============================================================

class MetroSystem:
    """封装单个城市的地铁系统：加载数据、构建图、求解路径。"""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.city = data_dir.name

        # 加载基础数据
        tsv = next(data_dir.glob("*station-id-map.tsv"))
        csv_f = next(data_dir.glob("*adjacency-distance.csv"))
        
        self.stations = load_station_map(str(tsv))
        adj = load_adjacency_matrix(str(csv_f))
        
        # 尝试加载线路文件
        line_txt = next(data_dir.glob("*station-lines.txt"), None)
        
        if line_txt:
            station_lines = load_station_lines(str(line_txt))
            self.graph, self.trans_graph = build_graphs(self.stations, adj, station_lines)
        else:
            raise ValueError (f"Missing station-lines.txt for city {self.city}. Cannot build transfer graph.")

        # 构建名称到 ID 的映射 (使用物理图生成的节点名称)
        # 新图的 nodes 中存储了 name 属性
        self.name_to_id: dict[str, int] = {
            attrs["name"]: node_id 
            for node_id, attrs in self.graph.nodes.items()
        }

    def sorted_station_names(self) -> list[str]:
        """返回按字母排序的站名列表 (包含线路后缀)。"""
        return sorted(self.name_to_id.keys())

    def shortest_path(self, src_name: str, dst_name: str) -> tuple[float, list[int]]:
            # 1. 将物理站名转换为物理节点的 ID
            src_phys_id = self.name_to_id.get(src_name)
            dst_phys_id = self.name_to_id.get(dst_name)
            
            if src_phys_id is None or dst_phys_id is None:
                raise ValueError(f"Invalid station name: {src_name} or {dst_name}")

            # 2. 从物理图的节点属性中，提取出它们对应的所有逻辑节点
            src_logic_nodes = self.graph.nodes[src_phys_id]['transfer_nodes']
            dst_logic_nodes = set(self.graph.nodes[dst_phys_id]['transfer_nodes'])

            # 3. 运行 Dijkstra (在换乘图 self.trans_graph 上求解)
            best_cost = float('inf')
            best_logic_path = []

            for start_node in src_logic_nodes:
                # 这里的 dijkstra 接收的是单个起点和一组终点 (tuple)
                cost, path = dijkstra(self.trans_graph, start_node, dst_logic_nodes)
                if cost < best_cost:
                    best_cost = cost
                    best_logic_path = path

            # 如果没有找到路径（比如网络不连通）
            if best_cost == float('inf'):
                return float('inf'), []

            # 4. 将逻辑路径翻译回物理路径，并去除换乘时产生的相邻重复节点
            phys_path = []
            for logic_id in best_logic_path:
                # 查出这个逻辑节点对应的原物理 ID
                p_id = self.trans_graph.nodes[logic_id]['original_id']
                
                # 如果 phys_path 是空的，或者当前 p_id 和上一个记录的 p_id 不一样，才加入
                if not phys_path or phys_path[-1] != p_id:
                    phys_path.append(p_id)

            # 最终返回物理路径和两种路径，供后续分析使用        
            return best_cost, phys_path, best_logic_path


def detect_cities(data_root: str | Path) -> list[str]:
    """扫描 data_root 下所有包含数据文件的城市子目录。"""
    data_root = Path(data_root)
    cities: list[str] = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and list(d.glob("*adjacency-distance.csv")):
            cities.append(d.name)
    return cities