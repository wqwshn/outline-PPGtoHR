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
