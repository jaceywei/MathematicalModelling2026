import numpy as np
import numba as nb

@nb.njit(fastmath=True) # 开启 fastmath 进一步压榨浮点运算速度
def _eigensystem_numba(C, epsilon, max_sweeps=20):
    # Numba 中尽量保持原始数组传入，或者在外部 copy 好
    C = C.copy().astype(np.float64) 
    n = C.shape[0]
    V = np.eye(n)
    
    for sweep in range(max_sweeps):
        # Numba 不太擅长 np.triu，直接手写找最大值更快
        max_off_val = 0.0
        for i in range(n):
            for j in range(i+1, n):
                val = abs(C[i, j])
                if val > max_off_val:
                    max_off_val = val
                    
        if max_off_val < epsilon:
            break
            
        thresh = max(epsilon, max_off_val * 0.2)
        
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(C[p, q]) > thresh:
                    theta = (C[q, q] - C[p, p]) / (2.0 * C[p, q])
                    # Numba 中 np.sign 对 0 的处理有点不同，手动写 if 最稳妥
                    if theta == 0.0:
                        t = 1.0
                    else:
                        sign_theta = 1.0 if theta > 0 else -1.0
                        t = sign_theta / (abs(theta) + np.sqrt(1.0 + theta**2))
                        
                    c = 1.0 / np.sqrt(1.0 + t**2)
                    s = c * t
                    
                    cpp, cqq, cpq = C[p, p], C[q, q], C[p, q]
                    
                    # Numba 里的终极奥义：纯标量循环极其快，绝对不要用切片(C[:, p])！
                    for i in range(n):
                        # 更新 C 的行列
                        cip = C[i, p]
                        ciq = C[i, q]
                        new_cip = c * cip - s * ciq
                        new_ciq = s * cip + c * ciq
                        C[i, p] = C[p, i] = new_cip
                        C[i, q] = C[q, i] = new_ciq
                        
                        # 更新 V
                        vip = V[i, p]
                        viq = V[i, q]
                        V[i, p] = c * vip - s * viq
                        V[i, q] = s * vip + c * viq
                    
                    # 修正由于浮点误差导致的交叉点异常
                    C[p, p] = c**2 * cpp - 2*s*c*cpq + s**2 * cqq
                    C[q, q] = s**2 * cpp + 2*s*c*cpq + c**2 * cqq
                    C[p, q] = C[q, p] = 0.0

    # Numba 里不能直接 np.argsort(np.diag(C))[::-1]，提取到外面做
    return C, V

# 包装函数
def _eigensystem(C, epsilon, max_sweeps=20):
    C_out, V_out = _eigensystem_numba(C, epsilon, max_sweeps)
    lambdas = np.diag(C_out)
    idx = np.argsort(lambdas)[::-1]
    return lambdas[idx], V_out[:, idx]

def svd(A, k, epsilon):
    """
    长方形 SVD 优化版：判断形状，拒绝无效算力
    """
    A = A.astype(np.float64)
    m, n = A.shape
    
    # --- 核心优化：小矩阵优先法则 ---
    if m >= n:
        # 高瘦型 (如 1920 x 1024) -> 构造 n x n 矩阵
        C = A.T @ A  
        lambdas, V = _eigensystem(C, epsilon=epsilon)
        
        # 计算有效秩与奇异值
        rank_A = np.sum(lambdas > 1e-9)
        k_eff = rank_A if (k <= 0 or k > rank_A) else k
        
        sigmas = np.sqrt(np.maximum(lambdas, 0))
        sigmas_k = sigmas[:k_eff]
        
        # 向量化求解 U_k = A * V * Sigma^-1
        U_k = (A @ V[:, :k_eff]) / sigmas_k
        
        # 注意：对于图像压缩和重构，无需补全 U！零奇异值对应的基底对图像毫无贡献。
        return U_k, sigmas_k, V[:, :k_eff].T
        
    else:
        # 矮胖型 (如 1024 x 1920) -> 构造 m x m 矩阵
        C = A @ A.T  
        lambdas, U = _eigensystem(C, epsilon=epsilon)
        
        rank_A = np.sum(lambdas > 1e-9)
        k_eff = rank_A if (k <= 0 or k > rank_A) else k
        
        sigmas = np.sqrt(np.maximum(lambdas, 0))
        sigmas_k = sigmas[:k_eff]
        
        # 向量化求解 V_k (利用 A = U \Sigma V^T  =>  V = A^T * U * \Sigma^-1)
        V_k = (A.T @ U[:, :k_eff]) / sigmas_k
        
        return U[:, :k_eff], sigmas_k, V_k.T