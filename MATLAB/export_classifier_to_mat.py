"""
export_classifier_to_mat.py
重训练 3 类 Random Forest 运动分类器并导出为 MATLAB .mat 格式
依赖: numpy, scipy, scikit-learn, joblib
"""

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import savemat

# ====== 1. 配置 ======
RESEARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'research')
ARTIFACTS_DIR = os.path.join(RESEARCH_DIR, 'artifacts')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

TARGET_CLASSES = ['arm_curl', 'jump_rope', 'push_up']

# ====== 2. 加载已有特征数据 ======
def load_features():
    import pickle
    pkl_path = os.path.join(ARTIFACTS_DIR, 'mimu_features_shortwin.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f'特征文件不存在: {pkl_path}\n请先运行 01b_mimu_feature_extraction_shortwin.ipynb')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['df']

# ====== 3. 训练 + 导出 ======
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('加载特征数据...')
    X, y, df = load_features()

    # 过滤目标类别
    mask = np.isin(y, TARGET_CLASSES)
    X_filtered = X[mask]
    y_filtered = y[mask]
    print(f'过滤后样本数: {len(y_filtered)} (类别: {np.unique(y_filtered)})')

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_filtered)
    print(f'标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}')

    # 训练 RF
    print('训练 Random Forest...')
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42,
    )
    rf.fit(X_scaled, y_encoded)
    print(f'训练集准确率: {rf.score(X_scaled, y_encoded):.4f}')

    # 导出 scaler
    savemat(os.path.join(OUTPUT_DIR, 'scaler_params.mat'), {
        'feature_mean': scaler.mean_.reshape(1, -1),
        'feature_std':  scaler.scale_.reshape(1, -1),
    })
    print('已导出 scaler_params.mat')

    # 导出 RF 模型
    n_trees = len(rf.estimators_)
    tree_children_left = np.zeros((n_trees,), dtype=object)
    tree_children_right = np.zeros((n_trees,), dtype=object)
    tree_feature = np.zeros((n_trees,), dtype=object)
    tree_threshold = np.zeros((n_trees,), dtype=object)
    tree_value = np.zeros((n_trees,), dtype=object)

    for i, tree_est in enumerate(rf.estimators_):
        t = tree_est.tree_
        tree_children_left[i] = t.children_left.astype(np.float64)
        tree_children_right[i] = t.children_right.astype(np.float64)
        tree_feature[i] = t.feature.astype(np.float64)
        tree_threshold[i] = t.threshold.astype(np.float64)
        tree_value[i] = t.value[:, 0, :].astype(np.float64)

    savemat(os.path.join(OUTPUT_DIR, 'rf_model_3class.mat'), {
        'n_trees': float(n_trees),
        'n_classes': float(len(le.classes_)),
        'tree_children_left': tree_children_left,
        'tree_children_right': tree_children_right,
        'tree_feature': tree_feature,
        'tree_threshold': tree_threshold,
        'tree_value': tree_value,
    })
    print(f'已导出 rf_model_3class.mat ({n_trees} 棵树)')

    # 导出标签映射
    savemat(os.path.join(OUTPUT_DIR, 'label_map.mat'), {
        'class_names': le.classes_,
        'class_indices': le.transform(le.classes_).astype(float),
    })
    print('已导出 label_map.mat')
    print('全部导出完成。')


if __name__ == '__main__':
    main()
