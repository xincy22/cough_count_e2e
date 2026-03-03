# 00 Data Preparation

EdgeAI数据预处理实验。这些脚本只需要运行一次来准备数据集。

## 步骤

1. **下载数据**
   ```bash
   python scripts/01_download.py
   ```

2. **构建manifest**
   ```bash
   python scripts/02_build_manifest.py
   ```

3. **分割受试者**
   ```bash
   python scripts/03_split_subjects.py
   ```

4. **预计算特征和标签**
   ```bash
   # 使用默认gaussian核, sigma=50ms
   python scripts/04_precompute_windows.py --mic both

   # 或尝试不同的density参数
   python scripts/04_precompute_windows.py --mic both --kernel skewed_gaussian --sigma-left-sec 0.04 --sigma-right-sec 0.08
   ```

5. **可视化检查**
   ```bash
   python scripts/05_visualize.py --mic out
   ```

6. **验证数据加载**
   ```bash
   python scripts/06_check_dataloader.py
   ```

## 输出

- `data/processed/edgeai/manifest.csv` - 数据清单
- `data/processed/edgeai/splits.json` - 受试者划分
- `data/processed/edgeai/npy/` - 预计算的特征和标签
