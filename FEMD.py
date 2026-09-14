import numpy as np
import pandas as pd
from PyEMD import EMD
from scipy.stats import kendalltau
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import os

# ------------------------ Function: IMF Plotting ------------------------
def plot_imfs(signal, imfs, time_samples=None, fignum=None, plotname=None, save_dir=None):
    if time_samples is None:
        time_samples = np.arange(signal.shape[0])
    if save_dir is None:
        save_dir = './'

    n_imfs = imfs.shape[0]
    plt.figure(num=fignum, figsize=(6, 8))
    axis_extent = np.max(np.abs(imfs[:-1, :]), axis=1).max()
    plt.subplots_adjust(hspace=0.5)

    ax = plt.subplot(n_imfs + 1, 1, 1)
    ax.plot(time_samples, signal)
    ax.set_xlim(time_samples[0], time_samples[-1])
    ax.set_ylim(signal.min(), signal.max())
    ax.grid(False)
    ax.set_ylabel('Signal')

    for i in range(n_imfs - 1):
        ax = plt.subplot(n_imfs + 1, 1, i + 2)
        ax.plot(time_samples, imfs[i, :])
        ax.set_xlim(time_samples[0], time_samples[-1])
        ax.set_ylim(-axis_extent, axis_extent)
        ax.grid(False)
        ax.set_ylabel(f'IMF {i + 1}')

    ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    ax.plot(time_samples, imfs[-1, :])
    ax.set_xlim(time_samples[0], time_samples[-1])
    ax.grid(False)
    ax.set_ylabel('Residual')
    ax.set_xlabel(plotname if plotname else "EMD Decomposition", fontsize=12)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if plotname:
        plt.savefig(os.path.join(save_dir, f"{plotname}.png"), bbox_inches='tight', dpi=300)
    plt.show()

# ------------------------ Data Loading and Preprocessing ------------------------
file_path = r"E:\zyh\岩相识别\补全\AKPO4条曲线补全.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file not found at: {file_path}")
try:
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    print(f"Successfully loaded data from {file_path}, shape: {data.shape}")
except Exception as e:
    raise ValueError(f"Error loading data from {file_path}: {str(e)}")
data = data[data[:, 0].argsort()]
if data.shape[1] < 6:
    raise ValueError("Data must have at least 6 columns (Depth, AC, GR, DEN, VP, Reservoir_Type)")

curves = {
    'AC': data[:, 1],
    'GR': data[:, 2],
    'DEN': data[:, 3],
    'VP': data[:, 4]
}
reservoir_type = data[:, 5].astype(int)

valid_mask = (reservoir_type >= 0) & (reservoir_type <= 13)
reservoir_type = reservoir_type[valid_mask]
for key in curves.keys():
    curves[key] = curves[key][valid_mask]

scaler = MinMaxScaler()
for key in curves.keys():
    curves[key] = scaler.fit_transform(curves[key].reshape(-1, 1)).flatten()

def compute_kendall_weight(imfs):
    n_imfs = len(imfs)
    kendall_weights = np.zeros(n_imfs)
    for t in range(n_imfs):
        kt = sum(abs(kendalltau(imfs[t], imfs[i])[0]) for i in range(n_imfs) if i != t)
        kendall_weights[t] = kt / (n_imfs - 1) if n_imfs > 1 else 1
    return kendall_weights / np.sum(kendall_weights) if np.sum(kendall_weights) > 0 else np.ones(n_imfs) / n_imfs

def compute_mi_weight(imfs):
    n_imfs = len(imfs)
    mi_weights = np.zeros(n_imfs)
    for t in range(n_imfs):
        q75, q25 = np.percentile(imfs[t], [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr * (len(imfs[t]) ** (-1/3))
        bins = int((np.max(imfs[t]) - np.min(imfs[t])) / bin_width) if bin_width > 0 else 10
        mt = sum(mutual_info_score(pd.cut(imfs[t], bins=bins, labels=False),
                                   pd.cut(imfs[i], bins=bins, labels=False))
                 for i in range(n_imfs) if i != t)
        mi_weights[t] = mt / (n_imfs - 1) if n_imfs > 1 else 1
    return mi_weights / np.sum(mi_weights) if np.sum(mi_weights) > 0 else np.ones(n_imfs) / n_imfs

def objective_function(alpha, imfs, y, kendall_w, mi_w, lambda_reg=0.1):
    beta = 1 - alpha
    fused_weights = alpha * kendall_w + beta * mi_w
    fused_weights /= np.sum(fused_weights) if np.sum(fused_weights) > 0 else 1
    weighted_imf = sum(fused_weights[i] * imfs[i] for i in range(len(imfs)))
    coef, _ = kendalltau(weighted_imf, y)
    reg_penalty = lambda_reg * ((alpha - 0.5) ** 2)
    return abs(coef) - reg_penalty

emd = EMD()
features = []
weights_dict = {}

time_range = np.arange(len(reservoir_type))

for curve_name, signal in curves.items():
    print(f"\nProcessing curve: {curve_name}")
    imfs = emd(signal)
    if len(imfs) == 0:
        raise ValueError(f"EMD decomposition failed for curve {curve_name}: No IMFs generated.")

    # 绘图：IMF 分量
    plot_imfs(signal=signal, imfs=imfs, time_samples=time_range,
              plotname=f"{curve_name}-EMD", save_dir="emd_figures")

    kendall_weights = compute_kendall_weight(imfs)
    mi_weights = compute_mi_weight(imfs)
    pbounds = {'alpha': (0.1, 0.9)}
    optimizer = BayesianOptimization(
        f=lambda alpha: objective_function(alpha, imfs, reservoir_type, kendall_weights, mi_weights, lambda_reg=0.1),
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=10)
    best_alpha = optimizer.max['params']['alpha']
    best_beta = 1 - best_alpha
    print(f"Best alpha for {curve_name}: {best_alpha:.4f}, Best beta: {best_beta:.4f}")

    fused_weights = best_alpha * kendall_weights + best_beta * mi_weights
    fused_weights /= np.sum(fused_weights) if np.sum(fused_weights) > 0 else 1

    weights_dict[curve_name] = {
        'Kendall_Weights': kendall_weights,
        'MI_Weights': mi_weights,
        'Fused_Weights': fused_weights
    }

    weighted_imf = sum(fused_weights[i] * imfs[i] for i in range(len(imfs)))
    features.append(weighted_imf)

weights_df = pd.DataFrame()
max_imfs = max(len(weights_dict[curve]['Fused_Weights']) for curve in weights_dict)
for curve_name in curves.keys():
    for weight_type in ['Kendall_Weights', 'MI_Weights', 'Fused_Weights']:
        weights = weights_dict[curve_name][weight_type]
        padded_weights = np.pad(weights, (0, max_imfs - len(weights)), 'constant', constant_values=np.nan)
        weights_df[f"{curve_name}_{weight_type}"] = padded_weights
weights_df.to_csv('imf_weights.csv', index=False)
print("\nWeights saved to 'imf_weights.csv'")

depth = data[:, 0][valid_mask]
feature_names = ['AC_Weighted', 'GR_Weighted', 'DEN_Weighted', 'VP_Weighted']
feature_df = pd.DataFrame(np.column_stack([depth] + features), columns=['Depth'] + feature_names)
feature_df['Reservoir_Type'] = reservoir_type
feature_df.to_csv('processed_well_log_data1.csv', index=False)
print("\nProcessed features and depth saved to 'processed_well_log_data.csv'")
