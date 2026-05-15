import os
import numpy as np
from svd import svd
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
import matplotlib.pyplot as plt

class SVDImage:
    def __init__(self, image_path):
        """初始化：加载图片并提取元数据"""
        self.input_path = image_path
        self.matrix, self.max_val, self.info = self._load_image(image_path)
        self.m, self.n, self.c = self.matrix.shape
        self.U_list = []  # 用于存储每个通道的 U 矩阵
        self.S_list = []  # 用于存储每个通道的 \Sigma
        self.Vt_list = [] # 用于存储每个通道的 V^T 矩阵
        
        # 用于存储最近一次压缩的结果
        self.reconstructed_matrix = None

    def _load_image(self, path):
            """私有方法：自适应加载并强制对齐三通道结构"""
            with Image.open(path) as img:
                mode = img.mode
                # 1. 识别位深
                max_val = 65535.0 if mode in ['I;16', 'I;16B', 'I;16L'] else 255.0
                
                # 2. 基础转换（处理 RGBA, P, CMYK 等模式）
                # 注意：16位图不能直接 convert('RGB')，否则会降级成 8位
                if mode not in ['RGB', 'I;16', 'I;16B', 'I;16L']:
                    img = img.convert('RGB')
                
                img_array = np.array(img).astype(np.float64)
                
                # 3. 【关键修正】维度补齐
                # 情况 A：如果是灰度图 (M, N)，将其堆叠为 (M, N, 3)
                if img_array.ndim == 2:
                    # np.stack 会在最后一个轴上重复三次
                    img_array = np.stack([img_array] * 3, axis=-1)
                    # print("检测到单通道图，已自动扩展为三通道矩阵以适配算法")
                
                # 情况 B：如果是带有透明通道的浮点数组 (M, N, 4)，切掉它
                elif img_array.ndim == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                    # print("检测到 Alpha 通道，已自动移除")

                info = {"mode": mode, "shape": img_array.shape}
                return img_array, max_val, info

    def precompute_svd(self):
            """一次性完成全量 SVD 分解并缓存矩阵，后续不再重复计算"""
            self.U_list.clear()
            self.S_list.clear()
            self.Vt_list.clear()
            
            # 对每一个通道独立进行 SVD 分解
            for i in range(self.c):
                channel_data = self.matrix[:, :, i]
                # 调用 svd 算法计算全量特征（k=0 代表不截断）
                U, S, Vt = svd(channel_data, k=0, epsilon = 1e-6 * self.max_val)
                self.U_list.append(U)
                self.S_list.append(S)
                self.Vt_list.append(Vt)

    def compress(self, k):
        """仅利用已缓存的矩阵进行快速截断和重构 (毫秒级)"""
        reconstructed_matrix = np.zeros_like(self.matrix)
        
        for i in range(self.c):
            # 取出预计算好的矩阵
            U = self.U_list[i]
            S = self.S_list[i]
            Vt = self.Vt_list[i]
            
            # 确保截断的 k 不超过实际奇异值数量
            k_eff = min(k, len(S))
            
            # 矩阵切片截断
            U_k = U[:, :k_eff]
            S_k = np.diag(S[:k_eff])
            Vt_k = Vt[:k_eff, :]
            
            # 快速重构该通道
            reconstructed_matrix[:, :, i] = U_k @ S_k @ Vt_k

        return reconstructed_matrix, self._update_metrics(reconstructed_matrix, k)

    def _update_metrics(self, reconstructed_matrix, k):
        """计算性能评价指标"""
        # 1. 压缩率 rho
        rho = (k * (self.m + self.n + 1)) / (self.m * self.n)
        
        # 2. PSNR
        mse = np.mean((self.matrix - reconstructed_matrix) ** 2)
        psnr = 10 * np.log10(self.max_val**2 / mse) if mse > 0 else 100
        
        # 3. SSIM (平均结构相似性)
        # 注意：ssim_func 期望 uint8 或 uint16 输入，所以这里做个临时转换
        data_range = self.max_val
        orig_img = self.matrix.astype(np.uint16 if data_range > 255 else np.uint8)
        rec_img = reconstructed_matrix.astype(np.uint16 if data_range > 255 else np.uint8)
        
        score_ssim = ssim_func(orig_img, rec_img, channel_axis=2, data_range=data_range)
        
        metrics = {"k": k, "rho": rho, "psnr": psnr, "ssim": score_ssim}

        return metrics
    
    def plot_singular_values(self):
        """
        绘制奇异值散点图（对数坐标），观察衰减特性
        """
        # 1. 提取预计算好的奇异值 (R通道)
        sigmas = self.S_list[0] 
        
        # 2. 处理奇异值为 0 的情况
        epsilon = 1e-16
        sigmas_log = np.maximum(sigmas, epsilon)
        indices = range(1, len(sigmas_log) + 1)
        
        # 3. 开始绘图
        plt.figure(figsize=(9, 6))
        
        # --- 核心修改：使用 scatter 绘制小散点 ---
        # s=1 表示点的大小，marker='o' 表示圆点
        plt.scatter(indices, sigmas_log, s=2, color='royalblue', 
                    marker='o', alpha=0.7, label='Singular Values')
        
        # 4. 强制开启对数轴 (因为 scatter 本身不带 log 逻辑)
        plt.yscale('log')
        
        # 5. 设置坐标轴和格式
        plt.title(f"Singular Value Distribution (Log Scale)\n{os.path.basename(self.input_path)}", fontsize=14)
        plt.xlabel("Index $n$", fontsize=12)
        plt.ylabel("Value $\sigma_n$ (log10)", fontsize=12)
        
        # 增加主网格和次网格，方便对齐量级
        plt.grid(True, which="major", ls="-", alpha=0.6)
        plt.grid(True, which="minor", ls=":", alpha=0.3)
        
        plt.legend()
        
        # 6. 保存到文件夹
        code_folder = os.getcwd() 
        name, _ = os.path.splitext(os.path.basename(self.input_path))
        plot_path = os.path.join(code_folder, f"{name}SVDecay.png")
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"奇异值散点图已更新至: {plot_path}")

    def save(self, rec_matrix, k):
        """接收外部传入的矩阵和 k 值进行保存"""
        folder = os.path.dirname(self.input_path)
        name, _ = os.path.splitext(os.path.basename(self.input_path))
        
        output_path = os.path.join(folder, f"{name}-k{k}.png")

        # 转换位深并保存
        dtype = np.uint16 if self.max_val == 65535.0 else np.uint8
        # 添加 np.clip 防止溢出
        clipped_matrix = np.clip(rec_matrix, 0, self.max_val)
        img_to_save = Image.fromarray(clipped_matrix.astype(dtype))
        img_to_save.save(output_path)
        print(f"结果已保存至: {output_path}")
        return output_path  # 添加这行，将路径返回给 app.py

    def show_metrics(metrics):
        """打印当前的评价结果"""
        print(f"\n--- 实验结果 (k={metrics['k']}) ---")
        print(f"压缩率 (rho): {metrics['rho']*100:.2f}%")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")