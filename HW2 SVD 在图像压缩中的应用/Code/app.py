import streamlit as st
import numpy as np
from PIL import Image
import os
from image_algorithm import SVDImage

# --- 页面配置 ---
st.set_page_config(page_title="SVD 图像压缩工具", layout="wide")

st.title("🖼️ 基于 SVD 的图像压缩实验平台")
st.markdown("通过奇异值分解（SVD）实现图像的低秩近似，探索 $k$ 值对图像质量的影响。")

# --- 1. 输入图片 (Sidebar) ---
st.sidebar.header("📂 上传与设置")
uploaded_file = st.sidebar.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # --- 核心修改：利用 session_state 缓存状态 ---
    # 判断是否是新上传的图片
    if "current_img_name" not in st.session_state or st.session_state.current_img_name != uploaded_file.name:
        # 如果是新图片，执行完整的初始化和 SVD 预计算
        st.session_state.current_img_name = uploaded_file.name
        
        # 1. 保存上传的文件到临时路径
        temp_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. 初始化对象
        img_obj = SVDImage(temp_path)
        
        # 3. 首次加载，执行耗时的全量分解
        with st.spinner('首次加载：正在计算底层 SVD 分解，后续滑动将瞬间完成...'):
            img_obj.precompute_svd()
            
        # 4. 把计算完毕的对象存入内存缓存
        st.session_state.img_obj = img_obj

        # 5. 打印奇异值（可选）
        # img_obj.plot_singular_values()
        
        # 6. 输出最大像素值（可选）
        # print(img_obj.max_val)

    # 从缓存中取出已经完成全量 SVD 分解的对象
    img_obj = st.session_state.img_obj
    N_max = min(img_obj.m, img_obj.n)

    # --- 2. k 的滑块与输入 ---
    st.sidebar.subheader("⚙️ 压缩参数")

    # 1. 初始化 session_state，将默认值设为 min(N_max, 50)
    if 'k_value' not in st.session_state:
        st.session_state.k_value = min (N_max, 50)

    # 2. 定义回调函数，用于同步状态
    def update_from_slider():
        st.session_state.k_value = st.session_state.slider_k

    def update_from_input():
        st.session_state.k_value = st.session_state.input_k

    # 3. 创建组件，绑定 key 和回调函数，并将 value 设为 session_state 中的值
    st.sidebar.slider(
        f"选择截断秩 $k$ (最大 {N_max})", 
        min_value=1, 
        max_value=N_max, 
        value=st.session_state.k_value,
        key='slider_k',              # 赋予唯一 key
        on_change=update_from_slider # 当滑块变动时触发
    )

    st.sidebar.number_input(
        "或者直接输入 $k$:", 
        min_value=1, 
        max_value=N_max, 
        value=st.session_state.k_value,
        key='input_k',               # 赋予唯一 key
        on_change=update_from_input  # 当输入框变动时触发
    )

    # 4. 最终使用的 k 值直接从 session_state 获取
    final_k = st.session_state.k_value

    # --- 3. 实时展示与评价 ---
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始图像")
        st.image(uploaded_file, use_column_width=True, caption=f"尺寸: {img_obj.m}x{img_obj.n}")

    # 执行压缩：由于矩阵已缓存，这里只进行切片和乘法，速度极快
    rec_matrix, metrics = img_obj.compress(k=final_k)

    with col2:
        st.subheader("压缩后图像")
        dtype = np.uint16 if img_obj.max_val > 255 else np.uint8
        # 注意防止浮点溢出，截断到合理范围
        display_img = Image.fromarray(np.clip(rec_matrix, 0, img_obj.max_val).astype(dtype))
        st.image(display_img, use_column_width=True, caption=f"截断秩 $k$ = {final_k}")

    # 展示指标
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("压缩率 $\\rho$", f"{metrics['rho']*100:.2f}%")
    m2.metric("峰值信噪比 PSNR", f"{metrics['psnr']:.2f} dB")
    m3.metric("结构相似性 SSIM", f"{metrics['ssim']:.4f}")

    # --- 4. 保存按钮 ---
    st.sidebar.divider()
    if st.sidebar.button("💾 保存当前结果到原路径"):
        # 注意：在网页环境下，st.sidebar.button 触发的是服务器端的保存
        # 因为你运行在本地，它会直接写到你 D:\Jacey Wei's... 那个路径下
        save_path = img_obj.save(rec_matrix, final_k)
        st.sidebar.success(f"已保存！")
        st.sidebar.info(f"路径: {save_path}")

else:
    st.info("请在左侧侧边栏上传一张图片以开始实验。")
    # 展示一个默认的占位图或说明