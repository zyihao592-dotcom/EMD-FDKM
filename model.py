import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, classification_report
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.metrics import classification_report
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, m1, m2, kernel='rbf'):
        self.kpca_models = []
        self.residual_outputs = []
        self.m1 = m1
        self.m2 = m2
        self.kernel = kernel

    def fit_transform(self, X):
        outputs = []

        for _ in range(self.m1):
            kpca = KernelPCA(kernel=self.kernel)
            x_kpca = kpca.fit_transform(X)
            outputs.append(x_kpca)

        if not outputs:
            raise ValueError("Outputs list is empty. Check the first loop.")

        residual_output = outputs[-1]

        for _ in range(self.m2):
            kpca = KernelPCA(kernel=self.kernel)
            X_kpca = kpca.fit_transform(X)

            if residual_output.shape[1] == 1:
                residual_output = np.tile(residual_output, (1, X_kpca.shape[1]))

            X_kpca = X_kpca + residual_output
            residual_output = X_kpca
            self.residual_outputs.append(X_kpca)

        return X


def racos_optimization(X, y, parameter_space, num_iterations=100):
    best_solution = None
    best_accuracy = -np.inf

    for t in tqdm(range(num_iterations), desc="优化进度"):
        x_star = {
            'n_estimators': random.randint(30, 200),
            'max_depth': random.randint(1, 20),
            'min_samples_split': random.randint(2, 10),
            'n1': random.randint(*parameter_space['n1']),
            'n2': random.randint(*parameter_space['n2']),
            'm1': random.randint(1, 7),
            'm2': random.randint(0, 7),
            'KPCA-kernel': 'rbf'
        }

        if x_star['m1'] + x_star['m2'] > 7:
            continue

        feature_extractor = FeatureExtractor(m1=x_star['m1'], m2=x_star['m2'], kernel=x_star['KPCA-kernel'])
        X_features = feature_extractor.fit_transform(X)

        rf_model = RandomForestClassifier(
            n_estimators=157,
            max_depth=14,
            min_samples_split=3
        )
        selector = RFE(rf_model, n_features_to_select=5)
        X_selected = selector.fit_transform(X_features, y)

        rf_model.fit(X_selected, y)
        accuracy = rf_model.score(X_selected, y)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_solution = x_star

    return best_solution, best_accuracy


def plot_confusion_matrix(cm, classes, title='混淆矩阵', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6), dpi=160)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

def plot_metrics(y_true, y_pred, title_prefix):
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = sorted([int(k) for k in report.keys() if k.isdigit()])

    precision = [report[str(c)]['precision'] for c in classes]
    recall = [report[str(c)]['recall'] for c in classes]
    f1 = [report[str(c)]['f1-score'] for c in classes]

    plt.figure(figsize=(6, 12))

    plt.subplot(3, 1, 1)
    plt.plot(classes, precision, marker='o', label=title_prefix)
    plt.title(f"{title_prefix} Precision")
    plt.xlabel("Class")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(classes, recall, marker='o', label=title_prefix)
    plt.title(f"{title_prefix} Recall")
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.ylim(0, 1.05)
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(classes, f1, marker='o', label=title_prefix)
    plt.title(f"{title_prefix} F1-Score")
    plt.xlabel("Class")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.05)
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取训练集和测试集数据
    train_file_path = r"D:\Lithofacies\机器学习代码\EMD-revision\噪声影响\RWDLS_train30.csv"  # 替换为你的训练集路径
    test_file_path = r"D:\Lithofacies\机器学习代码\EMD-revision\噪声影响\RWDLS_test30.csv"   # 替换为你的测试集路径

    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)

    features = ['AC', 'GR', 'TNPH', 'VP']
    X_train = df_train[features]
    y_train = df_train['Reservoir_Type'].astype(int)  # 转为整数

    X_test = df_test[features]
    y_test = df_test['Reservoir_Type'].astype(int)

    # 归一化
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # 参数搜索空间
    parameter_space = {
        'n1': (100, 500),
        'n2': (500, 1000),
        'm1': (1, 7),
        'm2': (0, 7)
    }

    # 参数优化
    best_solution, best_accuracy = racos_optimization(X_train_normalized, y_train, parameter_space)
    print("最佳参数:", best_solution)
    print("最佳准确率:", best_accuracy)

    # 特征提取与训练
    feature_extractor = FeatureExtractor(m1=best_solution['m1'], m2=best_solution['m2'], kernel=best_solution['KPCA-kernel'])
    X_train_features = feature_extractor.fit_transform(X_train_normalized)
    X_test_features = feature_extractor.fit_transform(X_test_normalized)

    rf_final = RandomForestClassifier(
        n_estimators=157,
        max_depth=14,
        min_samples_split=3
    )
    selector_final = RFE(rf_final, n_features_to_select=5)
    X_train_selected = selector_final.fit_transform(X_train_features, y_train)
    X_test_selected = selector_final.transform(X_test_features)

    rf_final.fit(X_train_selected, y_train)

    # 预测
    y_train_pred = rf_final.predict(X_train_selected)
    y_test_pred = rf_final.predict(X_test_selected)


    # 混淆矩阵和报告
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    print("\n训练集分类报告:")
    print(classification_report(y_train, y_train_pred, digits=4))

    print("\n测试集分类报告:")
    print(classification_report(y_test, y_test_pred, digits=4))

    plot_confusion_matrix(cm_train, np.unique(y_train), title='Training Set Confusion Matrix')
    plot_confusion_matrix(cm_test, np.unique(y_test), title='Test Set Confusion Matrix')

    # 绘制训练集指标
    plot_metrics(y_train, y_train_pred, "Train Set")

    # 绘制测试集指标
    plot_metrics(y_test, y_test_pred, "Test Set")

    # 保存训练集与测试集分类结果
    train_results = pd.DataFrame({
        'DEPTH': df_train['DEPTH'],
        'is_original': df_train['is_original'],
        'True_Label': y_train.reset_index(drop=True),
        'Predicted_Label': pd.Series(y_train_pred)
    })

    test_results = pd.DataFrame({
        'DEPTH': df_test['DEPTH'],
        'is_original': df_test['is_original'],
        'True_Label': y_test.reset_index(drop=True),
        'Predicted_Label': pd.Series(y_test_pred)
    })

    # 合并训练集和测试集的结果
    final_results = pd.concat([train_results, test_results], ignore_index=True)

    # 保存至CSV文件
    final_results.to_csv(r"classification_results_all10.csv", index=False, encoding='utf-8-sig')

    print("包含训练集与测试集的分类结果已保存至 classification_results_all10.csv。")

    # import shap
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # 1. 创建 SHAP explainer
    # explainer = shap.TreeExplainer(rf_final)
    #
    # # 2. 计算 SHAP 值
    # shap_values = explainer.shap_values(X_test_selected)  # list 或 ndarray (n_samples, n_features, n_classes)
    #
    # selected_feature_names = [
    #     f.replace('_Weighted', '') for f, s in zip(features, selector_final.support_) if s
    # ]
    # # 3. 将多分类 shap 转为二维
    # # 方法：对每个样本和特征取每类的绝对值平均
    # # if isinstance(shap_values, list):
    # #     shap_values_2d = np.mean([np.abs(s) for s in shap_values], axis=0)
    # # else:
    # #     shap_values_2d = np.abs(shap_values)
    #
    # # 4. 对应 RFE 的特征名
    #
    # print(shap_values.shape, X_test_selected.shape)
    # shap_values_2d = shap_values[:, :, -1]
    # # 5. 绘制 SHAP 蜂群图
    # shap.summary_plot(
    #     shap_values_2d,
    #     X_test_selected,
    #     feature_names=selected_feature_names,
    #     plot_type="dot"
    # )
    #
    # # 6. 可选：绘制平均 |SHAP| 条形图
    # mean_abs_by_feature = np.mean(abs(shap_values_2d), axis=0)
    # plt.figure(figsize=(6, 4), dpi=120)
    # plt.bar(selected_feature_names, mean_abs_by_feature)
    # plt.title("Mean |SHAP| per feature")
    # plt.ylabel("Mean |SHAP|")
    # plt.xticks(rotation=30)
    # plt.tight_layout()
    # plt.show()