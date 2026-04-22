# 子运动模态加权频谱融合 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将加权融合从 HR 结果层面前置到频谱层面，对 K 个专家滤波后信号的频谱进行分类器加权融合，再统一寻峰追踪。

**Architecture:** 多遍预处理（K 组 Fs_Target）→ 每窗口 K 路 LMS → FFT → 加权频谱融合 → 后级处理（惩罚/寻峰/追踪）。HF 和 ACC 双参考路径保持平行独立，各自完成频谱融合后独立寻峰，最终在 HR 层面融合。

**Tech Stack:** MATLAB R2023b+, Python 3.9+ (sklearn, scipy, numpy)

---

## 文件结构

### 新增文件

| 文件 | 职责 | 行数估计 |
|------|------|---------|
| `MATLAB/compute_spectrum.m` | 完整 FFT 频谱计算（返回全频段，非仅峰值） | ~25 |
| `MATLAB/weighted_spectrum_fusion.m` | K 路频谱加权融合 | ~30 |
| `MATLAB/ProcessMergedSpectrum.m` | 融合频谱的后级处理（惩罚 + 寻峰 + 追踪） | ~55 |
| `MATLAB/extract_mimu_features.m` | 75 维 IMU 特征提取 | ~120 |
| `MATLAB/predict_exercise_proba.m` | RF 模型推理（纯 MATLAB） | ~60 |
| `MATLAB/export_classifier_to_mat.py` | Python: 重训练 3 类 RF + 导出 .mat | ~120 |

### 修改文件

| 文件 | 改动范围 |
|------|---------|
| `MATLAB/HeartRateSolver_cas_chengfa.m` | 预处理扩展 + 主循环重构 + 新增辅助函数 |
| `MATLAB/AutoOptimize_Bayes_Search_cas_chengfa.m` | 搜索空间去掉前级参数，适配 expert_mode |
| `MATLAB/AutoOptimize_Result_Viewer_cas_chengfa.m` | 新增分类器概率时程图 |

### 新增数据文件

| 文件 | 来源 |
|------|------|
| `MATLAB/models/scaler_params.mat` | export_classifier_to_mat.py 生成 |
| `MATLAB/models/rf_model_3class.mat` | export_classifier_to_mat.py 生成 |
| `MATLAB/models/label_map.mat` | export_classifier_to_mat.py 生成 |
| `MATLAB/params/expert_arm_curl.mat` | 从现有贝叶斯优化结果提取 |
| `MATLAB/params/expert_jump_rope.mat` | 从现有贝叶斯优化结果提取 |
| `MATLAB/params/expert_push_up.mat` | 从现有贝叶斯优化结果提取 |

---

## Task 1: 创建 compute_spectrum.m

**Files:**
- Create: `MATLAB/compute_spectrum.m`

- [ ] **Step 1: 创建完整 FFT 频谱计算函数**

```matlab
function [freqs, amps] = compute_spectrum(signal, Fs)
% compute_spectrum 计算信号的完整 FFT 幅度谱
% 输入:
%   signal - 输入信号 (行或列向量)
%   Fs     - 采样率 (Hz)
% 输出:
%   freqs  - 频率向量 (Hz), 长度 Len/2
%   amps   - 单边幅度谱, 长度 Len/2

    if size(signal,1) > size(signal,2)
        signal = signal';
    end

    a = length(signal);
    Len = 2^13;

    FFTData = fft(signal, Len);
    FFTAmp0 = abs(FFTData) / a;
    amps = FFTAmp0(1:Len/2);
    amps(2:end) = 2 * amps(2:end);
    freqs = Fs * ((0:(Len/2)-1)) / Len;
end
```

- [ ] **Step 2: 提交**

```bash
git add MATLAB/compute_spectrum.m
git commit -m "feat: 新增 compute_spectrum 完整频谱计算函数"
```

---

## Task 2: 创建 weighted_spectrum_fusion.m

**Files:**
- Create: `MATLAB/weighted_spectrum_fusion.m`

- [ ] **Step 1: 创建频谱加权融合函数**

```matlab
function [freqs, amps_fused] = weighted_spectrum_fusion(spectra_list, weights)
% weighted_spectrum_fusion 对多路频谱进行加权融合
% 输入:
%   spectra_list - K 个幅度谱矩阵, 每个 size (Len/2, 1), 由 compute_spectrum 生成
%                  如果为矩阵则每列代表一个 expert 的频谱
%   weights      - 权重向量 (K, 1), 总和为 1
% 输出:
%   freqs       - 频率向量 (从第一个 spectrum 的隐含 Fs/Len 推导, 此处仅回传)
%   amps_fused  - 融合后的幅度谱 (Len/2, 1)

    if iscell(spectra_list)
        K = length(spectra_list);
        amps_fused = zeros(size(spectra_list{1}));
        for k = 1:K
            amps_fused = amps_fused + weights(k) * spectra_list{k};
        end
    else
        % 矩阵形式: 每列一个 expert
        amps_fused = spectra_list * weights(:);
    end
end
```

- [ ] **Step 2: 提交**

```bash
git add MATLAB/weighted_spectrum_fusion.m
git commit -m "feat: 新增 weighted_spectrum_fusion 频谱加权融合函数"
```

---

## Task 3: 创建 ProcessMergedSpectrum.m

**Files:**
- Create: `MATLAB/ProcessMergedSpectrum.m`

此函数从已融合的频谱出发，执行频谱惩罚、寻峰、历史追踪，是专家模式的后级处理核心。逻辑复用自 `Helper_Process_Spectrum`，但输入从时域信号改为频域数据。

- [ ] **Step 1: 创建融合频谱后级处理函数**

```matlab
function est_freq = ProcessMergedSpectrum(freqs, amps, sig_penalty_ref, Fs, ...
    para, times, history_arr, enable_penalty, range_hz, limit_bpm, step_bpm)
% ProcessMergedSpectrum 对已融合的频谱执行后级处理
% 输入:
%   freqs            - 频率向量 (Hz)
%   amps             - 融合后的幅度谱
%   sig_penalty_ref  - 频谱惩罚参考信号 (时域)
%   Fs               - 采样率
%   para             - 参数结构体
%   times            - 当前时间步索引
%   history_arr      - 历史 HR 列向量
%   enable_penalty   - 是否启用频谱惩罚
%   range_hz         - HR 搜索范围 (Hz)
%   limit_bpm        - 跳变限制 (BPM)
%   step_bpm         - 跳变步长 (BPM)
% 输出:
%   est_freq         - 估计心率频率 (Hz)

    % --- 1. 频谱惩罚 ---
    if para.Spec_Penalty_Enable && enable_penalty
        [S_ref, S_ref_amp] = FFT_Peaks(sig_penalty_ref, Fs, 0.3);
        if ~isempty(S_ref)
            [~, midx] = max(S_ref_amp);
            Motion_Freq = S_ref(midx);
            mask = (abs(freqs - Motion_Freq) < para.Spec_Penalty_Width) | ...
                   (abs(freqs - 2*Motion_Freq) < para.Spec_Penalty_Width);
            amps(mask) = amps(mask) * para.Spec_Penalty_Weight;
        end
    end

    % --- 2. 有效频段内寻峰 (1~4 Hz) ---
    free_low = 1;
    free_high = 4;
    valid_idx = (freqs > free_low) & (freqs < free_high);
    freqs_valid = freqs(valid_idx);
    amps_valid = amps(valid_idx);

    [pks, locs] = findpeaks(amps_valid);

    if isempty(pks)
        est_freq = 0;
        return;
    end

    threshold = max(pks) * 0.3;
    keep = pks > threshold;
    pks_kept = pks(keep);
    locs_kept = locs(keep);
    freqs_kept = freqs_valid(locs_kept);

    % 按幅值降序排列
    [~, sort_idx] = sort(pks_kept, 'descend');
    Fre = freqs_kept(sort_idx);

    if isempty(Fre)
        est_freq = 0;
        return;
    end

    curr_raw = Fre(1);

    % --- 3. 历史追踪 ---
    if times == 1
        est_freq = curr_raw;
    else
        prev_hr = history_arr(times-1);
        [calc_hr, ~] = Find_nearBiggest(Fre, prev_hr, range_hz, -range_hz);

        diff_hr = calc_hr - prev_hr;
        limit = limit_bpm / 60;
        step  = step_bpm / 60;

        if diff_hr > limit,      est_freq = prev_hr + step;
        elseif diff_hr < -limit, est_freq = prev_hr - step;
        else,                    est_freq = calc_hr;
        end
    end
end
```

- [ ] **Step 2: 提交**

```bash
git add MATLAB/ProcessMergedSpectrum.m
git commit -m "feat: 新增 ProcessMergedSpectrum 融合频谱后级处理函数"
```

---

## Task 4: 创建 export_classifier_to_mat.py

**Files:**
- Create: `MATLAB/export_classifier_to_mat.py`
- Output: `MATLAB/models/scaler_params.mat`, `MATLAB/models/rf_model_3class.mat`, `MATLAB/models/label_map.mat`

- [ ] **Step 1: 创建 Python 导出脚本**

```python
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
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# ====== 1. 配置 ======
RESEARCH_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'research')
ARTIFACTS_DIR = os.path.join(RESEARCH_DIR, 'artifacts')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'models')

# 目标类别 (排除 jumping_jack)
TARGET_CLASSES = ['arm_curl', 'jump_rope', 'push_up']

FS = 100
WIN_SEC = 2
STEP_SEC = 0.5
WIN_SAMPLES = int(WIN_SEC * FS)
STEP_SAMPLES = int(STEP_SEC * FS)
MOTION_THRESHOLD = 0.05

MIMU_CHANNELS = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
TIME_FEAT_NAMES = ['mean', 'std', 'min', 'max', 'range', 'energy', 'zcr', 'skewness', 'kurtosis']
MAG_FEAT_NAMES = ['mean', 'std', 'energy', 'dominant_freq']


# ====== 2. 特征提取 (与 01b notebook 一致) ======
def extract_time_features(sig):
    return [
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.max(sig) - np.min(sig), np.mean(sig**2),
        np.sum(np.diff(np.sign(sig)) != 0) / (len(sig) - 1),
        skew(sig), kurtosis(sig),
    ]

def extract_freq_features(sig, fs=FS):
    nperseg = min(len(sig), 256)
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    non_dc = psd[1:]
    if len(non_dc) > 0 and np.max(non_dc) > 0:
        return [freqs[1:][np.argmax(non_dc)]]
    return [0.0]

def extract_mag_features(mag_sig, fs=FS):
    td = extract_time_features(mag_sig)
    fd = extract_freq_features(mag_sig, fs)
    return [td[0], td[1], td[5], fd[0]]  # mean, std, energy, dominant_freq

def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]

def extract_window_features(channels_data, fs=FS):
    """
    channels_data: dict with keys AccX,AccY,AccZ,GyroX,GyroY,GyroZ, each a 1D array
    返回 75 维特征向量
    """
    features = []

    # 6 channels x (9 time + 1 freq) = 60
    for ch in MIMU_CHANNELS:
        sig = channels_data[ch]
        features.extend(extract_time_features(sig))
        features.extend(extract_freq_features(sig, fs))

    # Magnitude features: acc_mag, gyro_mag x 4 = 8
    acc_mag = np.sqrt(channels_data['AccX']**2 + channels_data['AccY']**2 + channels_data['AccZ']**2)
    gyro_mag = np.sqrt(channels_data['GyroX']**2 + channels_data['GyroY']**2 + channels_data['GyroZ']**2)
    features.extend(extract_mag_features(acc_mag, fs))
    features.extend(extract_mag_features(gyro_mag, fs))

    # Cross-correlation: 7
    features.append(safe_corr(acc_mag, gyro_mag))
    features.append(safe_corr(channels_data['AccX'], channels_data['AccY']))
    features.append(safe_corr(channels_data['AccX'], channels_data['AccZ']))
    features.append(safe_corr(channels_data['AccY'], channels_data['AccZ']))
    features.append(safe_corr(channels_data['GyroX'], channels_data['GyroY']))
    features.append(safe_corr(channels_data['GyroX'], channels_data['GyroZ']))
    features.append(safe_corr(channels_data['GyroY'], channels_data['GyroZ']))

    return np.array(features, dtype=float)


# ====== 3. 加载已有特征数据 ======
def load_features():
    import pickle
    pkl_path = os.path.join(ARTIFACTS_DIR, 'mimu_features_shortwin.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['df']


# ====== 4. 训练 + 导出 ======
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('加载特征数据...')
    X, y, df = load_features()

    # 过滤目标类别
    mask = np.isin(y, TARGET_CLASSES)
    X_filtered = X[mask]
    y_filtered = y[mask]
    df_filtered = df[mask].reset_index(drop=True)
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

    # ====== 导出 scaler ======
    savemat(os.path.join(OUTPUT_DIR, 'scaler_params.mat'), {
        'feature_mean': scaler.mean_.reshape(1, -1),   # (1, 75)
        'feature_std':  scaler.scale_.reshape(1, -1),   # (1, 75)
    })
    print('已导出 scaler_params.mat')

    # ====== 导出 RF 模型 ======
    # 将每棵决策树拆解为 MATLAB 可用的数组
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
        tree_value[i] = t.value[:, 0, :].astype(np.float64)  # (n_nodes, n_classes)

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

    # ====== 导出标签映射 ======
    savemat(os.path.join(OUTPUT_DIR, 'label_map.mat'), {
        'class_names': le.classes_,
        'class_indices': le.transform(le.classes_).astype(float),
    })
    print('已导出 label_map.mat')
    print('全部导出完成。')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行导出脚本**

```bash
cd MATLAB && python export_classifier_to_mat.py
```

Expected: 输出 `models/scaler_params.mat`, `models/rf_model_3class.mat`, `models/label_map.mat`，打印训练集准确率和样本数。

- [ ] **Step 3: 提交**

```bash
git add MATLAB/export_classifier_to_mat.py MATLAB/models/
git commit -m "feat: 新增 Python 分类器导出脚本及 3 类 RF 模型文件"
```

---

## Task 5: 创建 extract_mimu_features.m

**Files:**
- Create: `MATLAB/extract_mimu_features.m`

- [ ] **Step 1: 创建 75 维 IMU 特征提取函数**

```matlab
function features = extract_mimu_features(accx, accy, accz, gyrox, gyroy, gyroz, Fs)
% extract_mimu_features 提取 75 维 IMU 特征 (2s 窗口)
% 输入: 各轴信号片段 + 采样率
% 输出: (75, 1) 特征向量

    channels = {accx(:), accy(:), accz(:), gyrox(:), gyroy(:), gyroz(:)};
    features = zeros(75, 1);
    idx = 1;

    % === 6 channels x 10 features (9 time + 1 freq) = 60 ===
    for c = 1:6
        sig = channels{c};

        % 时域: 9 features
        features(idx)   = mean(sig);       idx = idx + 1;
        features(idx)   = std(sig);        idx = idx + 1;
        features(idx)   = min(sig);        idx = idx + 1;
        features(idx)   = max(sig);        idx = idx + 1;
        features(idx)   = max(sig) - min(sig); idx = idx + 1;
        features(idx)   = mean(sig.^2);    idx = idx + 1; % energy
        features(idx)   = sum(diff(sign(sig)) ~= 0) / (length(sig) - 1); idx = idx + 1; % zcr
        features(idx)   = skewness(sig);   idx = idx + 1;
        features(idx)   = kurtosis(sig);   idx = idx + 1;

        % 频域: dominant_freq
        nperseg = min(length(sig), 256);
        [f, P] = pwelch(sig, hamming(nperseg), [], [], Fs);
        non_dc = P(2:end);
        if ~isempty(non_dc) && max(non_dc) > 0
            features(idx) = f(2 + find(non_dc == max(non_dc), 1) - 1);
        else
            features(idx) = 0;
        end
        idx = idx + 1;
    end

    % === Magnitude features: acc_mag x 4 + gyro_mag x 4 = 8 ===
    acc_mag = sqrt(accx(:).^2 + accy(:).^2 + accz(:).^2);
    gyro_mag = sqrt(gyrox(:).^2 + gyroy(:).^2 + gyroz(:).^2);

    for mag = 1:2
        if mag == 1, sig = acc_mag; else, sig = gyro_mag; end
        features(idx) = mean(sig); idx = idx + 1;
        features(idx) = std(sig);  idx = idx + 1;
        features(idx) = mean(sig.^2); idx = idx + 1; % energy
        nperseg = min(length(sig), 256);
        [f, P] = pwelch(sig, hamming(nperseg), [], [], Fs);
        non_dc = P(2:end);
        if ~isempty(non_dc) && max(non_dc) > 0
            features(idx) = f(2 + find(non_dc == max(non_dc), 1) - 1);
        else
            features(idx) = 0;
        end
        idx = idx + 1;
    end

    % === Cross-correlation: 7 ===
    features(idx) = safe_corr(acc_mag, gyro_mag); idx = idx + 1;
    features(idx) = safe_corr(accx(:), accy(:));  idx = idx + 1;
    features(idx) = safe_corr(accx(:), accz(:));  idx = idx + 1;
    features(idx) = safe_corr(accy(:), accz(:));  idx = idx + 1;
    features(idx) = safe_corr(gyrox(:), gyroy(:)); idx = idx + 1;
    features(idx) = safe_corr(gyrox(:), gyroz(:)); idx = idx + 1;
    features(idx) = safe_corr(gyroy(:), gyroz(:)); idx = idx + 1;
end

function r = safe_corr(a, b)
    if std(a) == 0 || std(b) == 0
        r = 0;
    else
        r = corrcoef(a, b);
        r = r(1, 2);
    end
end
```

- [ ] **Step 2: 提交**

```bash
git add MATLAB/extract_mimu_features.m
git commit -m "feat: 新增 extract_mimu_features 75 维 IMU 特征提取"
```

---

## Task 6: 创建 predict_exercise_proba.m

**Files:**
- Create: `MATLAB/predict_exercise_proba.m`

- [ ] **Step 1: 创建 RF 推理函数**

```matlab
function proba = predict_exercise_proba(features, model_path)
% predict_exercise_proba 使用导出的 RF 模型进行推理
% 输入:
%   features   - (75, 1) 特征向量
%   model_path - 模型目录路径
% 输出:
%   proba      - (K, 1) 概率向量, K=类别数

    % 加载模型
    S = load(fullfile(model_path, 'scaler_params.mat'));
    mean_vals = S.feature_mean(:)';
    std_vals  = S.feature_std(:)';

    M = load(fullfile(model_path, 'rf_model_3class.mat'));
    n_trees   = M.n_trees;
    n_classes = M.n_classes;

    % 标准化
    x = (features(:)' - mean_vals) ./ std_vals;
    % 处理 std=0 的情况
    x(isnan(x)) = 0;

    % 遍历每棵树
    class_counts = zeros(1, n_classes);
    for t = 1:n_trees
        node = 1; % MATLAB 1-indexed
        cl = M.tree_children_left{t};
        cr = M.tree_children_right{t};
        feat = M.tree_feature{t};
        thresh = M.tree_threshold{t};
        val = M.tree_value{t};

        while cl(node) ~= -1 % -1 表示叶节点 (sklearn 用 TREE_LEAF = -1)
            f_idx = feat(node) + 1; % sklearn 0-indexed -> MATLAB 1-indexed
            if f_idx < 1 || f_idx > length(x)
                break;
            end
            if x(f_idx) <= thresh(node)
                node = cl(node) + 1; % 0-indexed -> 1-indexed
            else
                node = cr(node) + 1;
            end
        end
        class_counts = class_counts + val(node, :);
    end

    % 归一化为概率
    total = sum(class_counts);
    if total > 0
        proba = class_counts(:) / total;
    else
        proba = ones(n_classes, 1) / n_classes;
    end
end
```

- [ ] **Step 2: 提交**

```bash
git add MATLAB/predict_exercise_proba.m
git commit -m "feat: 新增 predict_exercise_proba RF 推理函数"
```

---

## Task 7: 修改 HeartRateSolver_cas_chengfa.m

**Files:**
- Modify: `MATLAB/HeartRateSolver_cas_chengfa.m`

这是最核心的修改。改动策略：
1. 预处理阶段：检测 `para.expert_mode`，若启用则运行 K 遍预处理
2. 主循环：运动段内运行 K 路 LMS → FFT → 频谱融合 → 后级处理
3. 新增辅助函数 `Helper_ClassifierWeights` 和 `Helper_ExpertFilter`
4. 保持 `para.expert_mode = false` 时完全回退到原始逻辑

### Step 1: 扩展预处理区域

将第 42-62 行（重采样 + 带通滤波）替换为支持多专家预处理的版本。

**查找（第 42-62 行）：**
```matlab
% 重采样处理
Fs = para.Fs_Target;
ppg_ori   = resample(filloutliers(raw_data(:, Col_PPG),'previous','mean'), Fs, Fs_Origin);
hotf1_ori = resample(raw_data(:, Col_HF1), Fs, Fs_Origin);
hotf2_ori = resample(raw_data(:, Col_HF2), Fs, Fs_Origin);
% HF 信号仅两路, 无第三路占位
accx_ori  = resample(raw_data(:, Col_Acc(1)), Fs, Fs_Origin);
accy_ori  = resample(raw_data(:, Col_Acc(2)), Fs, Fs_Origin);
accz_ori  = resample(raw_data(:, Col_Acc(3)), Fs, Fs_Origin);

HR_Ref_Data = ref_data;

% 带通滤波 (0.5 - 5 Hz)
BP_Low = 0.5; BP_High = 5; BP_Order = 4;
[b_but, a_but] = butter(BP_Order, [BP_Low BP_High]/(Fs/2), 'bandpass');

ppg   = filtfilt(b_but, a_but, ppg_ori);
hotf1 = filtfilt(b_but, a_but, hotf1_ori);
hotf2 = filtfilt(b_but, a_but, hotf2_ori);
accx  = filtfilt(b_but, a_but, accx_ori);
accy  = filtfilt(b_but, a_but, accy_ori);
accz  = filtfilt(b_but, a_but, accz_ori);
```

**替换为：**
```matlab
HR_Ref_Data = ref_data;

if isfield(para, 'expert_mode') && para.expert_mode
    %% 专家模式: K 遍预处理
    expert_names = fieldnames(para.expert_params);
    K = length(expert_names);
    Fs_list = zeros(K, 1);
    for k = 1:K
        Fs_list(k) = para.expert_params.(expert_names{k}).Fs_Target;
    end
    Fs_common = max(Fs_list);

    sig_sets = cell(K, 1);
    for k = 1:K
        ep = para.expert_params.(expert_names{k});
        Fs_k = ep.Fs_Target;

        ppg_k   = resample(filloutliers(raw_data(:, Col_PPG),'previous','mean'), Fs_k, Fs_Origin);
        hf1_k   = resample(raw_data(:, Col_HF1), Fs_k, Fs_Origin);
        hf2_k   = resample(raw_data(:, Col_HF2), Fs_k, Fs_Origin);
        accx_k  = resample(raw_data(:, Col_Acc(1)), Fs_k, Fs_Origin);
        accy_k  = resample(raw_data(:, Col_Acc(2)), Fs_k, Fs_Origin);
        accz_k  = resample(raw_data(:, Col_Acc(3)), Fs_k, Fs_Origin);

        BP_Low = 0.5; BP_High = 5; BP_Order = 4;
        [b_k, a_k] = butter(BP_Order, [BP_Low BP_High]/(Fs_k/2), 'bandpass');

        sig_sets{k} = struct( ...
            'ppg',  filtfilt(b_k, a_k, ppg_k), ...
            'hf1',  filtfilt(b_k, a_k, hf1_k), ...
            'hf2',  filtfilt(b_k, a_k, hf2_k), ...
            'accx', filtfilt(b_k, a_k, accx_k), ...
            'accy', filtfilt(b_k, a_k, accy_k), ...
            'accz', filtfilt(b_k, a_k, accz_k), ...
            'Fs',   Fs_k);
    end

    % 用于运动检测和静息 FFT 的公共信号集 (取最高 Fs)
    Fs = Fs_common;
    ss0 = sig_sets{Fs_list == Fs_common};
    ppg  = ss0.ppg;  hotf1 = ss0.hf1; hotf2 = ss0.hf2;
    accx = ss0.accx; accy  = ss0.accy; accz = ss0.accz;

    % 加载分类器模型
    classifier_model_path = para.model_path;
    scaler_data = load(fullfile(classifier_model_path, 'scaler_params.mat'));
    rf_data     = load(fullfile(classifier_model_path, 'rf_model_3class.mat'));
    label_data  = load(fullfile(classifier_model_path, 'label_map.mat'));

    % IMU 列配置 (需要包含陀螺仪)
    Col_Gyro = [12, 13, 14]; % 默认列索引, 需根据数据格式调整
    if isfield(para, 'Col_Gyro')
        Col_Gyro = para.Col_Gyro;
    end
    % 预提取全信号的 IMU 数据用于分类器
    imu_gyrox = resample(raw_data(:, Col_Gyro(1)), Fs_common, Fs_Origin);
    imu_gyroy = resample(raw_data(:, Col_Gyro(2)), Fs_common, Fs_Origin);
    imu_gyroz = resample(raw_data(:, Col_Gyro(3)), Fs_common, Fs_Origin);
    % accx/accy/accz 已在上面从公共信号集获取

else
    %% 原始模式: 单遍预处理 (保持不变)
    Fs = para.Fs_Target;
    ppg_ori   = resample(filloutliers(raw_data(:, Col_PPG),'previous','mean'), Fs, Fs_Origin);
    hotf1_ori = resample(raw_data(:, Col_HF1), Fs, Fs_Origin);
    hotf2_ori = resample(raw_data(:, Col_HF2), Fs, Fs_Origin);
    accx_ori  = resample(raw_data(:, Col_Acc(1)), Fs, Fs_Origin);
    accy_ori  = resample(raw_data(:, Col_Acc(2)), Fs, Fs_Origin);
    accz_ori  = resample(raw_data(:, Col_Acc(3)), Fs, Fs_Origin);

    BP_Low = 0.5; BP_High = 5; BP_Order = 4;
    [b_but, a_but] = butter(BP_Order, [BP_Low BP_High]/(Fs/2), 'bandpass');

    ppg   = filtfilt(b_but, a_but, ppg_ori);
    hotf1 = filtfilt(b_but, a_but, hotf1_ori);
    hotf2 = filtfilt(b_but, a_but, hotf2_ori);
    accx  = filtfilt(b_but, a_but, accx_ori);
    accy  = filtfilt(b_but, a_but, accy_ori);
    accz  = filtfilt(b_but, a_but, accz_ori);
end
```

### Step 2: 扩展主循环初始化区域

在第 87-89 行之后添加专家模式所需的变量：

**查找（第 87-89 行）：**
```matlab
% LMS 固定参数
Num_Cascade_HF = 2; Num_Cascade_Acc = 3; LMS_Mu_Base = 0.01;
last_motion_flag = false;
```

**替换为：**
```matlab
last_motion_flag = false;

if isfield(para, 'expert_mode') && para.expert_mode
    % 专家模式初始化
    expert_names_local = fieldnames(para.expert_params);
    K_local = length(expert_names_local);
    Fs_common_local = Fs_common;

    % 分类器模式
    if ~isfield(para, 'classifier_mode')
        para.classifier_mode = 'window';
    end

    % 段级模式: 预计算运动段
    if strcmp(para.classifier_mode, 'segment')
        all_weights = precompute_segment_weights(...
            accx, accy, accz, imu_gyrox, imu_gyroy, imu_gyroz, ...
            Fs_common_local, Win_Len, Win_Step, time_end, ...
            scaler_data, rf_data, para);
    else
        all_weights = [];
    end

    % HR 矩阵扩展: 追加 3 列用于存储分类器概率
    HR = zeros(1, 12); % Col10-12: w_curl, w_rope, w_pushup
else
    % 原始模式
    Num_Cascade_HF = 2; Num_Cascade_Acc = 3; LMS_Mu_Base = 0.01;
    HR = zeros(1, 9);
end
```

### Step 3: 修改主循环中的运动段处理逻辑

将第 126-184 行的 `if is_motion ... else ... end` 块替换为同时支持专家模式和原始模式的版本。

**查找（第 126-184 行，即 `if is_motion` 到 `end`）：**
```matlab
    if is_motion
        % 静息→运动转换时重置追踪链 ...
        ...
    else
        % 静息段: 跳过 LMS, 直接复制 FFT 结果 ...
        ...
    end
```

**替换为：**
```matlab
    if is_motion
        rest_to_motion = ~last_motion_flag;
        if rest_to_motion
            times_hf  = 1;
            times_acc = 1;
        else
            times_hf  = times;
            times_acc = times;
        end

        if isfield(para, 'expert_mode') && para.expert_mode
            %% === 专家模式: K 路 LMS → 频谱融合 ===
            spectra_hf = zeros(2^12, K_local); % 预分配
            spectra_acc = zeros(2^12, K_local);
            best_hf_ref_k = cell(K_local, 1);
            best_acc_ref_k = cell(K_local, 1);

            for k = 1:K_local
                ss = sig_sets{k};
                ep = para.expert_params.(expert_names_local{k});
                Fs_k = ss.Fs;
                idx_s_k = round(time_1 * Fs_k) + 1;
                idx_e_k = round(time_2 * Fs_k);
                if idx_e_k > length(ss.ppg), idx_e_k = length(ss.ppg); end

                Sig_p_k = ss.ppg(idx_s_k:idx_e_k);
                Sig_h_k = {ss.hf1(idx_s_k:idx_e_k), ss.hf2(idx_s_k:idx_e_k)};
                Sig_a_k = {ss.accx(idx_s_k:idx_e_k), ss.accy(idx_s_k:idx_e_k), ss.accz(idx_s_k:idx_e_k)};

                % 时延计算
                [mh_k, ma_k, td_h_k, td_a_k] = ChooseDelay1218(...
                    Fs_k, time_1, ss.ppg, Sig_a_k, Sig_h_k);

                % --- LMS-HF ---
                Sig_e_hf = Sig_p_k;
                if td_h_k < 0, ord_h = floor(abs(td_h_k)*1); else, ord_h = 1; end
                ord_h = min(max(ord_h, 1), ep.Max_Order);

                mh_mat_k = sort(mh_k, 'descend');
                [~, best_hf_idx_k] = max(mh_k);
                lms_mu_hf = ep.LMS_Mu_Base;
                if isfield(ep, 'Num_Cascade_HF'), nc_hf = ep.Num_Cascade_HF; else, nc_hf = 2; end

                for ci = 1:min(nc_hf, length(mh_k))
                    curr_corr = mh_mat_k(ci);
                    ri = find(mh_k == curr_corr, 1);
                    Sig_e_hf = lmsFunc_h(lms_mu_hf - curr_corr/100, ord_h, 0, Sig_h_k{ri}, Sig_e_hf);
                end

                % --- LMS-ACC ---
                Sig_e_acc = Sig_p_k;
                if td_a_k < 0, ord_a = floor(abs(td_a_k)*1.5); else, ord_a = 1; end
                ord_a = min(max(ord_a, 1), ep.Max_Order);

                ma_mat_k = sort(ma_k, 'descend');
                [~, best_acc_idx_k] = max(ma_k);
                lms_mu_acc = ep.LMS_Mu_Base;
                if isfield(ep, 'Num_Cascade_Acc'), nc_acc = ep.Num_Cascade_Acc; else, nc_acc = 3; end

                for ci = 1:min(nc_acc, length(ma_k))
                    curr_corr = ma_mat_k(ci);
                    ri = find(ma_k == curr_corr, 1);
                    Sig_e_acc = lmsFunc_h(lms_mu_acc - curr_corr/100, ord_a, 1, Sig_a_k{ri}, Sig_e_acc);
                end

                % 重采样到公共 Fs 用于频谱分析
                if Fs_k ~= Fs_common_local
                    Sig_e_hf = resample(Sig_e_hf, Fs_common_local, Fs_k);
                    Sig_e_acc = resample(Sig_e_acc, Fs_common_local, Fs_k);
                end

                % 计算频谱
                [freqs_common, amps_hf] = compute_spectrum(Sig_e_hf, Fs_common_local);
                [~, amps_acc] = compute_spectrum(Sig_e_acc, Fs_common_local);

                spectra_hf(1:length(amps_hf), k) = amps_hf;
                spectra_acc(1:length(amps_acc), k) = amps_acc;

                % 存惩罚参考 (重采样到公共 Fs)
                best_hf_ref_k{k} = resample(Sig_h_k{best_hf_idx_k}, Fs_common_local, Fs_k);
                best_acc_ref_k{k} = resample(Sig_a_k{best_acc_idx_k}, Fs_common_local, Fs_k);
            end

            % 获取分类器权重
            weights = Helper_ClassifierWeights(...
                times, time_1, Win_Len, Fs_common_local, ...
                accx, accy, accz, imu_gyrox, imu_gyroy, imu_gyroz, ...
                all_weights, para, scaler_data, rf_data);

            % 存储权重到 HR 矩阵
            HR(times, 10:12) = weights(:)';

            % 频谱融合
            [~, S_fused_hf] = weighted_spectrum_fusion(spectra_hf, weights);
            [~, S_fused_acc] = weighted_spectrum_fusion(spectra_acc, weights);

            % 加权惩罚参考信号
            ref_hf_fused = zeros(length(best_hf_ref_k{1}), 1);
            ref_acc_fused = zeros(length(best_acc_ref_k{1}), 1);
            for k = 1:K_local
                ref_hf_fused  = ref_hf_fused  + weights(k) * best_hf_ref_k{k}(:);
                ref_acc_fused = ref_acc_fused + weights(k) * best_acc_ref_k{k}(:);
            end

            % 后级处理
            Freq_HF = ProcessMergedSpectrum(freqs_common, S_fused_hf, ref_hf_fused, ...
                Fs_common_local, para, times_hf, HR(:,3), ...
                true, para.HR_Range_Hz, para.Slew_Limit_BPM, para.Slew_Step_BPM);
            HR(times, 3) = Freq_HF;

            Freq_ACC = ProcessMergedSpectrum(freqs_common, S_fused_acc, ref_acc_fused, ...
                Fs_common_local, para, times_acc, HR(:,4), ...
                true, para.HR_Range_Hz, para.Slew_Limit_BPM, para.Slew_Step_BPM);
            HR(times, 4) = Freq_ACC;

        else
            %% === 原始模式 (保持不变) ===
            [mh_arr, ma_arr, time_delay_h, time_delay_a] = ...
                ChooseDelay1218(Fs, time_1, ppg, {accx,accy,accz}, {hotf1,hotf2});

            Sig_LMS_HF = Sig_p;
            if time_delay_h < 0, ord_h = floor(abs(time_delay_h)*1); else, ord_h = 1; end
            ord_h = min(max(ord_h, 1), para.Max_Order);
            mh_mat = sort(mh_arr, 'descend');
            [~, best_hf_idx] = max(mh_arr);
            for i = 1:min(Num_Cascade_HF, length(mh_arr))
                curr_corr = mh_mat(i);
                real_idx = find(mh_arr == curr_corr, 1);
                [Sig_LMS_HF,~,~] = lmsFunc_h(LMS_Mu_Base - curr_corr/100, ord_h, 0, Sig_h{real_idx}, Sig_LMS_HF);
            end
            Freq_HF = Helper_Process_Spectrum(Sig_LMS_HF, Sig_h{best_hf_idx}, Fs, para, times_hf, HR(:,3), ...
                                            true, para.HR_Range_Hz, para.Slew_Limit_BPM, para.Slew_Step_BPM);
            HR(times, 3) = Freq_HF;

            Sig_LMS_ACC = Sig_p;
            if time_delay_a < 0, ord_a = floor(abs(time_delay_a)*1.5); else, ord_a = 1; end
            ord_a = min(max(ord_a, 1), para.Max_Order);
            ma_mat = sort(ma_arr, 'descend');
            [~, best_acc_idx] = max(ma_arr);
            for i = 1:min(Num_Cascade_Acc, length(ma_arr))
                curr_corr = ma_mat(i);
                real_idx = find(ma_arr == curr_corr, 1);
                Ref_Sig = Sig_a{real_idx};
                [Sig_LMS_ACC,~,~] = lmsFunc_h(LMS_Mu_Base - curr_corr/100, ord_a, 1, Ref_Sig, Sig_LMS_ACC);
            end
            Freq_ACC = Helper_Process_Spectrum(Sig_LMS_ACC, Sig_a{best_acc_idx}, Fs, para, times_acc, HR(:,4), ...
                                             true, para.HR_Range_Hz, para.Slew_Limit_BPM, para.Slew_Step_BPM);
            HR(times, 4) = Freq_ACC;
        end
    else
        HR(times, 3) = Freq_FFT;
        HR(times, 4) = Freq_FFT;
    end
```

### Step 4: 添加分类器权重辅助函数

在文件末尾 `end` 之前（即 `Helper_Process_Spectrum` 函数之后）添加新辅助函数：

```matlab
function weights = Helper_ClassifierWeights(times, time_1, Win_Len, Fs, ...
    accx, accy, accz, gyrox, gyroy, gyroz, ...
    all_weights, para, scaler_data, rf_data)
% Helper_ClassifierWeights 计算当前窗口的分类器权重
% 支持 'window' 和 'segment' 两种模式

    if strcmp(para.classifier_mode, 'segment') && ~isempty(all_weights)
        % 段级模式: 从预计算结果中查找
        t_idx = find(all_weights(:,1) == time_1, 1);
        if ~isempty(t_idx)
            weights = all_weights(t_idx, 2:end)';
        else
            n_cls = rf_data.n_classes;
            weights = ones(n_cls, 1) / n_cls;
        end
    else
        % 窗级模式: 实时提取特征并推理
        win_len_samples = round(Win_Len * Fs);
        idx_s = round(time_1 * Fs) + 1;
        idx_e = idx_s + win_len_samples - 1;

        if idx_e > length(accx)
            n_cls = rf_data.n_classes;
            weights = ones(n_cls, 1) / n_cls;
            return;
        end

        features = extract_mimu_features(...
            accx(idx_s:idx_e), accy(idx_s:idx_e), accz(idx_s:idx_e), ...
            gyrox(idx_s:idx_e), gyroy(idx_s:idx_e), gyroz(idx_s:idx_e), Fs);

        proba_raw = predict_exercise_proba_local(features, scaler_data, rf_data);

        % 确保权重与 expert 顺序一致
        label_data = load(fullfile(para.model_path, 'label_map.mat'));
        class_names = label_data.class_names;
        expert_names_local = fieldnames(para.expert_params);
        weights = zeros(length(expert_names_local), 1);
        for k = 1:length(expert_names_local)
            en = expert_names_local{k};
            cls_idx = find(strcmp(class_names, en));
            if ~isempty(cls_idx)
                weights(k) = proba_raw(cls_idx);
            end
        end

        % 归一化 (确保总和为 1)
        total = sum(weights);
        if total > 0
            weights = weights / total;
        else
            weights = ones(length(expert_names_local), 1) / length(expert_names_local);
        end
    end
end

function proba = predict_exercise_proba_local(features, scaler_data, rf_data)
% predict_exercise_proba_local 内联 RF 推理 (避免重复加载模型)
    mean_vals = scaler_data.feature_mean(:)';
    std_vals  = scaler_data.feature_std(:)';
    x = (features(:)' - mean_vals) ./ std_vals;
    x(isnan(x)) = 0;

    n_trees   = rf_data.n_trees;
    n_classes = rf_data.n_classes;
    class_counts = zeros(1, n_classes);

    for t = 1:n_trees
        node = 1;
        cl = rf_data.tree_children_left{t};
        cr = rf_data.tree_children_right{t};
        feat = rf_data.tree_feature{t};
        thresh = rf_data.tree_threshold{t};
        val = rf_data.tree_value{t};

        while cl(node) ~= -1
            f_idx = feat(node) + 1;
            if f_idx < 1 || f_idx > length(x), break; end
            if x(f_idx) <= thresh(node)
                node = cl(node) + 1;
            else
                node = cr(node) + 1;
            end
        end
        class_counts = class_counts + val(node, :);
    end

    total = sum(class_counts);
    if total > 0
        proba = class_counts(:) / total;
    else
        proba = ones(n_classes, 1) / n_classes;
    end
end

function all_weights = precompute_segment_weights(...
    accx, accy, accz, gyrox, gyroy, gyroz, ...
    Fs, Win_Len, Win_Step, time_end, ...
    scaler_data, rf_data, para)
% precompute_segment_weights 段级模式: 预计算每个窗口的分类器概率
    all_weights = [];
    t = para.Time_Start;
    while t + Win_Len <= time_end
        idx_s = round(t * Fs) + 1;
        idx_e = idx_s + round(Win_Len * Fs) - 1;
        if idx_e > length(accx), break; end

        features = extract_mimu_features(...
            accx(idx_s:idx_e), accy(idx_s:idx_e), accz(idx_s:idx_e), ...
            gyrox(idx_s:idx_e), gyroy(idx_s:idx_e), gyroz(idx_s:idx_e), Fs);

        proba = predict_exercise_proba_local(features, scaler_data, rf_data);
        all_weights = [all_weights; t, proba(:)'];
        t = t + Win_Step;
    end
end
```

### Step 5: 提交

```bash
git add MATLAB/HeartRateSolver_cas_chengfa.m
git commit -m "feat: HeartRateSolver 支持专家模式 - K路LMS频谱融合"
```

---

## Task 8: 修改 AutoOptimize_Bayes_Search_cas_chengfa.m

**Files:**
- Modify: `MATLAB/AutoOptimize_Bayes_Search_cas_chengfa.m`

### Step 1: 更新搜索空间（去掉前级参数）

**查找（第 43-58 行）：**
```matlab
SearchSpace.Fs_Target = [25, 50, 100];
SearchSpace.Max_Order = [12, 16, 20];
SearchSpace.Spec_Penalty_Width = [0.1, 0.2, 0.3];
```

**替换为：**
```matlab
% 前级参数 (Fs_Target, Max_Order) 已固定在各专家参数中, 不再纳入优化
SearchSpace.Spec_Penalty_Width = [0.1, 0.2, 0.3];
```

同时删除 `SearchSpace.Fs_Target` 和 `SearchSpace.Max_Order` 行。

### Step 2: 添加专家模式基础配置

**查找（第 27-34 行）：**
```matlab
para_base.FileName = 'dataformatlab\multi_bobi1_processed.mat';
para_base.Time_Start = 1;
para_base.Time_Buffer = 10;
para_base.Calib_Time = 30;
para_base.Motion_Th_Scale = 2.5;
para_base.Spec_Penalty_Enable = 1;
para_base.Spec_Penalty_Weight = 0.2;
```

**替换为：**
```matlab
para_base.FileName = 'dataformatlab\multi_bobi1_processed.mat';
para_base.Time_Start = 1;
para_base.Time_Buffer = 10;
para_base.Calib_Time = 30;
para_base.Motion_Th_Scale = 2.5;
para_base.Spec_Penalty_Enable = 1;
para_base.Spec_Penalty_Weight = 0.2;

% 专家模式配置
para_base.expert_mode = true;
para_base.classifier_mode = 'window';
para_base.model_path = 'models';
para_base.expert_params = struct();
% 加载各专家参数
expert_files = {'params/expert_arm_curl.mat', ...
                'params/expert_jump_rope.mat', ...
                'params/expert_push_up.mat'};
expert_names = {'arm_curl', 'jump_rope', 'push_up'};
for ei = 1:length(expert_files)
    if isfile(expert_files{ei})
        tmp = load(expert_files{ei});
        fnames = fieldnames(tmp);
        para_base.expert_params.(expert_names{ei}) = tmp.(fnames{1});
    else
        warning('专家参数文件不存在: %s, 使用默认参数', expert_files{ei});
        para_base.expert_params.(expert_names{ei}) = struct( ...
            'Fs_Target', 50, 'Max_Order', 16, 'LMS_Mu_Base', 0.01, ...
            'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
    end
end
```

### Step 3: 更新 Wrapper_CostFunction 中的参数映射

Wrapper_CostFunction 需要确保不覆盖 expert_params。在现有参数映射循环后添加注释说明即可，因为新搜索空间已不含 Fs_Target 和 Max_Order。

### Step 4: 提交

```bash
git add MATLAB/AutoOptimize_Bayes_Search_cas_chengfa.m
git commit -m "feat: 贝叶斯优化适配专家模式, 搜索空间仅保留后级参数"
```

---

## Task 9: 修改 AutoOptimize_Result_Viewer_cas_chengfa.m

**Files:**
- Modify: `MATLAB/AutoOptimize_Result_Viewer_cas_chengfa.m`

### Step 1: 在双子图之后添加分类器概率时程图

在第 106 行 (`linkaxes(...)`) 之后添加：

```matlab
%% 4b. 分类器概率时程图 (专家模式)
if size(HR_HF, 2) >= 12
    figure('Name', 'Classifier Probability Timeline', 'Color', 'w', 'Position', [50, 50, 1000, 400]);
    subplot(2,1,1);
    area(T_Pred_HF, HR_HF(:,10:12), 'Stacked');
    legend({'arm\_curl', 'jump\_rope', 'push\_up'}, 'Location', 'best');
    title(sprintf('Fusion(HF) 分类器概率 | 运动段AAE=%.4f', M_FusHF_1));
    xlabel('Time (s)'); ylabel('Probability');
    ylim([0 1]); grid on;

    subplot(2,1,2);
    area(T_Pred_ACC, HR_ACC(:,10:12), 'Stacked');
    legend({'arm\_curl', 'jump\_rope', 'push\_up'}, 'Location', 'best');
    title(sprintf('Fusion(ACC) 分类器概率 | 运动段AAE=%.4f', M_FusACC_2));
    xlabel('Time (s)'); ylabel('Probability');
    ylim([0 1]); grid on;
end
```

### Step 2: 提交

```bash
git add MATLAB/AutoOptimize_Result_Viewer_cas_chengfa.m
git commit -m "feat: Result Viewer 新增分类器概率时程图"
```

---

## Task 10: 端到端验证

**Files:**
- 无新增/修改

- [ ] **Step 1: 验证专家参数文件存在**

确认以下文件就位（从现有贝叶斯优化结果中提取前级参数）：
- `MATLAB/params/expert_arm_curl.mat`
- `MATLAB/params/expert_jump_rope.mat`
- `MATLAB/params/expert_push_up.mat`

每个文件应包含结构体，至少含字段: `Fs_Target`, `Max_Order`, `LMS_Mu_Base`, `Num_Cascade_HF`, `Num_Cascade_Acc`。

- [ ] **Step 2: 验证模型文件存在**

确认以下文件由 `export_classifier_to_mat.py` 生成：
- `MATLAB/models/scaler_params.mat`
- `MATLAB/models/rf_model_3class.mat`
- `MATLAB/models/label_map.mat`

- [ ] **Step 3: 验证向后兼容**

在 MATLAB 中运行原始模式，确认结果与改动前一致：
```matlab
para = struct('expert_mode', false, ...); % 不含 expert_mode 或 expert_mode=false
Result = HeartRateSolver_cas_chengfa(para);
% 对比 Result.err_stats 与改动前的结果
```

- [ ] **Step 4: 运行专家模式**

```matlab
para = struct('expert_mode', true, 'classifier_mode', 'window', ...);
Result = HeartRateSolver_cas_chengfa(para);
% 检查 Result.err_stats 的运动段 AAE
% 检查 Result.HR(:,10:12) 的概率分布是否合理
```

- [ ] **Step 5: 对比两种分类器方案**

分别使用 `classifier_mode = 'window'` 和 `'segment'` 运行，对比运动段 AAE。

- [ ] **Step 6: 运行贝叶斯优化**

```matlab
AutoOptimize_Bayes_Search_cas_chengfa
```

验证优化器正常收敛，后级参数搜索空间为 9 个参数。

- [ ] **Step 7: 最终提交**

```bash
git add -A
git commit -m "feat: 子运动模态加权频谱融合 - 全流程集成完成"
```

---

## 自审检查

### 1. 设计规格覆盖

| 设计规格要求 | 对应 Task |
|-------------|----------|
| Python 训练 + MATLAB 推理 | Task 4, 6 |
| 3 类分类器 (arm_curl, jump_rope, push_up) | Task 4 |
| 75 维 IMU 特征 | Task 5 |
| 8s 窗级权重 | Task 7 (Helper_ClassifierWeights) |
| 段级权重 | Task 7 (precompute_segment_weights) |
| K 遍 LMS + 双参考路径 | Task 7 (主循环) |
| 频谱加权融合 | Task 2, 7 |
| 后级参数贝叶斯优化 (9 参数) | Task 8 |
| 向后兼容 (expert_mode=false) | Task 7 |
| 可视化扩展 | Task 9 |
| 分类器概率时程 | Task 9 |

### 2. 类型一致性

- `weighted_spectrum_fusion` 输入 `spectra_list` 在 Task 7 中以矩阵形式传入 (每列一个 expert)，与函数内 `else` 分支一致
- `ProcessMergedSpectrum` 签名在 Task 3 定义，Task 7 中调用参数匹配
- HR 矩阵列定义: 1-9 原始列 + 10-12 分类器权重，Task 7 和 Task 9 一致
- `predict_exercise_proba_local` 在 Task 6 定义独立函数，Task 7 中为避免重复加载使用内联版本（相同逻辑）

### 3. 已修复的占位符

- 专家参数文件路径: `params/expert_*.mat`，由用户从现有优化结果中提取
- IMU 陀螺仪列: `Col_Gyro = [12, 13, 14]` 为默认值，用户需根据数据格式调整
- `Col_Gyro` 已通过 `para.Col_Gyro` 支持运行时覆盖
