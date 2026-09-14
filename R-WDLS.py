import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import dirichlet
from collections import Counter
from sklearn.model_selection import train_test_split


class RWDLS:
    def __init__(self, k_neighbors=5, n_shadow=2, b_points=3,
                 target_min_count=100, small_threshold=10,
                 random_state=42):
        self.k = k_neighbors
        self.n_shadow = n_shadow
        self.b_points = b_points
        self.target_min_count = target_min_count
        self.small_threshold = small_threshold
        self.random_state = random_state
        np.random.seed(random_state)

    # =========================
    # 相对密度
    # =========================
    def _relative_density(self, X):
        n_neighbors = min(self.k + 1, len(X))
        if n_neighbors <= 1:
            return np.ones(len(X)) / len(X)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:, 1:]

        inv = 1.0 / (distances + 1e-8)
        local_density = np.sum(inv, axis=1)
        return local_density / np.sum(local_density)

    # =========================
    # shadow samples
    # =========================
    def _generate_shadow_samples(self, X):
        shadows = []
        stds = np.std(X, axis=0, ddof=1)

        for x in X:
            for _ in range(self.n_shadow):
                noise = np.random.normal(0, 0.1, size=X.shape[1]) * (stds + 1e-8)
                shadows.append(x + noise)

        return np.array(shadows)

    # =========================
    # 生成新样本
    # =========================
    def _generate_new_samples(self, candidate_pool, n_samples, X_all, y_all, cls):
        synthetic_samples = []
        max_trials = n_samples * 10
        trials = 0

        if len(candidate_pool) == 0:
            return np.array([])

        while len(synthetic_samples) < n_samples and trials < max_trials:
            trials += 1

            b = min(self.b_points, len(candidate_pool))
            idx = np.random.choice(len(candidate_pool), size=b, replace=(b < self.b_points))
            points = candidate_pool[idx]

            alpha = dirichlet([1] * b).rvs()[0]
            v = np.dot(alpha, points)

            k_neighbors = min(self.k, len(X_all) - 1)
            if k_neighbors < 1:
                synthetic_samples.append(v)
                continue

            nbrs_all = NearestNeighbors(n_neighbors=k_neighbors).fit(X_all)
            _, idx_all = nbrs_all.kneighbors([v])
            neighbor_labels = y_all[idx_all[0]]

            if np.mean(neighbor_labels == cls) >= 0.5:
                synthetic_samples.append(v)

        return np.array(synthetic_samples)

    # =========================
    # 核心过采样
    # =========================
    def fit_resample(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)

        # ⭐ 固定目标：100
        adaptive_target = self.target_min_count

        X_resampled = [X]
        y_resampled = [y]
        flags = [np.ones(len(y), dtype=int)]

        for cls, count in zip(classes, counts):

            max_multiplier = 10
            target_count = min(adaptive_target, count * max_multiplier)

            if count < target_count:
                n_to_sample = target_count - count
                X_class = X[y == cls]

                print(f"\n类别 {cls} 原始数量: {count} -> 目标: {target_count}")

                # =========================
                # 极小类（保持你原逻辑）
                # =========================
                if len(X_class) <= 5:
                    print(" -> 极小类，启用强制扩充")

                    new_samples = []

                    while len(new_samples) < n_to_sample:
                        idx = np.random.choice(len(X_class), 2, replace=True)
                        x1, x2 = X_class[idx]

                        alpha = np.random.rand()
                        v = alpha * x1 + (1 - alpha) * x2

                        noise = np.random.normal(0, 0.05, size=v.shape)
                        v = v + noise

                        new_samples.append(v)

                    new_samples = np.array(new_samples)

                else:
                    # =========================
                    # 小样本补齐
                    # =========================
                    if len(X_class) < self.small_threshold:
                        n_needed = self.small_threshold - len(X_class)
                        stds = np.std(X_class, axis=0, ddof=1)

                        noise = np.random.normal(
                            0, 0.05,
                            size=(n_needed, X_class.shape[1])
                        ) * (stds + 1e-8)

                        extra = X_class[np.random.choice(len(X_class), n_needed, replace=True)] + noise
                        X_class = np.vstack([X_class, extra])

                    rd = self._relative_density(X_class)
                    X_filtered = X_class[rd >= np.median(rd)]

                    shadow_samples = self._generate_shadow_samples(X_filtered)
                    candidate_pool = np.vstack([X_filtered, shadow_samples])

                    new_samples = self._generate_new_samples(
                        candidate_pool, n_to_sample, X, y, cls
                    )

                    # =========================
                    # 补齐机制（已修复）
                    # =========================
                    if len(new_samples) < n_to_sample:
                        print(" -> 触发补齐机制")

                        n_missing = n_to_sample - len(new_samples)
                        supplement = []

                        for _ in range(n_missing):
                            idx = np.random.choice(len(X_class), 2, replace=True)
                            x1, x2 = X_class[idx]

                            alpha = np.random.rand()
                            v = alpha * x1 + (1 - alpha) * x2

                            noise = np.random.normal(0, 0.05, size=v.shape)
                            v = v + noise

                            supplement.append(v)

                        supplement = np.array(supplement)

                        # ⭐ 修复 vstack bug
                        if len(new_samples) == 0:
                            new_samples = supplement
                        else:
                            new_samples = np.vstack([new_samples, supplement])

                X_resampled.append(new_samples)
                y_resampled.append(np.array([cls] * len(new_samples)))
                flags.append(np.zeros(len(new_samples), dtype=int))

        return np.vstack(X_resampled), np.hstack(y_resampled), np.hstack(flags)


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    file_path = r"E:\zyh\岩相识别\Revision\processed_well_log_data.csv"
    df = pd.read_csv(file_path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df.iloc[:, -1],
        random_state=42
    )

    print("\n原始数据分布:", Counter(df.iloc[:, -1]))
    print("训练集分布:", Counter(train_df.iloc[:, -1]))
    print("测试集分布:", Counter(test_df.iloc[:, -1]))

    X_train = train_df.iloc[:, 1:-1].values
    y_train = train_df.iloc[:, -1].values
    depth_train = train_df.iloc[:, 0].values

    rwdls = RWDLS(target_min_count=100, small_threshold=10)

    X_res, y_res, flags = rwdls.fit_resample(X_train, y_train)

    print("\n过采样后训练集分布:", Counter(y_res))

    depth_res = np.concatenate([
        depth_train,
        np.full(len(y_res) - len(y_train), np.nan)
    ])

    feature_names = df.columns[1:-1]
    label_name = df.columns[-1]

    df_train_res = pd.DataFrame(
        np.column_stack([depth_res, X_res, y_res, flags]),
        columns=["DEPTH"] + list(feature_names) + [label_name, "is_original"]
    )

    df_test = test_df.copy()
    df_test["is_original"] = 1

    save_train_path = r"E:\zyh\岩相识别\Revision\train_RWDLS原始.csv"
    save_test_path = r"E:\zyh\岩相识别\Revision\test_RWDLS原始.csv"

    df_train_res.to_csv(save_train_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(save_test_path, index=False, encoding="utf-8-sig")

    print("\n训练集保存:", save_train_path)
    print("测试集保存:", save_test_path)