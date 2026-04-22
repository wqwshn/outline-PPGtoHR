function QuickTest(dataset, mode)
%% QuickTest 快速调试脚本
% 用法:
%   QuickTest                    % bobi1, 标准模式 vs 专家模式(基线参数)
%   QuickTest('bobi2')           % 换数据集
%   QuickTest('bobi1', 'std')    % 仅标准模式
%   QuickTest('bobi1', 'expert') % 仅专家模式 (优先加载优化后参数)
%   QuickTest('bobi1', 'compare') % 优化后评估: 专家(基线) vs 专家(优化后) 对比
%
% 'compare' 模式需要先运行 AutoOptimize_Bayes_Search_cas_chengfa 生成
% Best_Params_Expert_Result_multi_*.mat 文件。

arguments
    dataset (1,:) char = 'bobi1'
    mode    (1,:) char = 'both'   % 'both' | 'std' | 'expert' | 'compare'
end

clc; close all;

%% 1. 数据文件定位
data_dir = 'dataformatlab';
data_file = fullfile(data_dir, sprintf('multi_%s_processed.mat', dataset));
if ~isfile(data_file)
    error('数据文件不存在: %s', data_file);
end
fprintf('数据文件: %s\n', data_file);

%% 2. 专家参数加载 (前级参数, 固定不变)
expert_sources = struct( ...
    'arm_curl',  fullfile(data_dir, 'Best_Params_Result_multi_wanju1_processed.mat'), ...
    'jump_rope', fullfile(data_dir, 'Best_Params_Result_multi_tiaosheng2_processed.mat'), ...
    'push_up',   fullfile(data_dir, 'Best_Params_Result_multi_fuwo2_processed.mat'));

expert_names = fieldnames(expert_sources);
expert_params = struct();
fprintf('\n--- 加载专家前级参数 ---\n');
for i = 1:length(expert_names)
    en = expert_names{i};
    ef = expert_sources.(en);
    if isfile(ef)
        tmp = load(ef, 'Best_Para_HF');
        bp = tmp.Best_Para_HF;
        expert_params.(en) = struct( ...
            'Fs_Target', bp.Fs_Target, 'Max_Order', bp.Max_Order, ...
            'LMS_Mu_Base', 0.01, 'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
        fprintf('  %s: Fs=%dHz, MaxOrder=%d\n', en, bp.Fs_Target, bp.Max_Order);
    else
        expert_params.(en) = struct( ...
            'Fs_Target', 25, 'Max_Order', 16, ...
            'LMS_Mu_Base', 0.01, 'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
        fprintf('  %s: 文件不存在, 使用默认值\n', en);
    end
end

%% 3. 后级参数: 基线 vs 优化后
% 基线参数 (标准模式贝叶斯优化的结果, 或默认值)
[para_base, base_loaded] = LoadBackEndParams(data_dir, dataset, data_file);
fprintf('\n');

% 优化后参数 (专家模式专属贝叶斯优化)
expert_param_file = fullfile(data_dir, sprintf('Best_Params_Expert_Result_multi_%s_processed.mat', dataset));
optimized_loaded = false;
para_optimized = [];
if isfile(expert_param_file)
    tmp = load(expert_param_file, 'Best_Para_Expert_ACC');
    bp_exp = tmp.Best_Para_Expert_ACC;
    para_optimized = para_base;
    override_fields = {'Spec_Penalty_Width', 'Spec_Penalty_Weight', ...
        'HR_Range_Hz', 'Slew_Limit_BPM', 'Slew_Step_BPM', ...
        'HR_Range_Rest', 'Slew_Limit_Rest', 'Slew_Step_Rest', ...
        'Smooth_Win_Len', 'Time_Bias'};
    for fi = 1:length(override_fields)
        fn = override_fields{fi};
        if isfield(bp_exp, fn)
            para_optimized.(fn) = bp_exp.(fn);
        end
    end
    optimized_loaded = true;
    fprintf('优化后参数: 已从 %s 加载\n', expert_param_file);
else
    fprintf('优化后参数: 未找到 (需先运行 AutoOptimize_Bayes_Search_cas_chengfa)\n');
end

%% 4. 根据模式执行
switch mode
    case 'compare'
        RunCompare(dataset, data_file, expert_params, para_base, para_optimized, optimized_loaded);
    case 'std'
        RunStd(dataset, para_base);
    case 'expert'
        RunExpert(dataset, expert_params, para_base, para_optimized, optimized_loaded);
    otherwise  % 'both'
        RunBoth(dataset, para_base, expert_params, para_base);
end

fprintf('\n=== QuickTest 完成 ===\n');
end

%% ========================================================================
%  执行模式
%  ========================================================================

function RunStd(dataset, para)
    fprintf('\n=== 运行标准模式 ===\n');
    tic;
    res = HeartRateSolver_cas_chengfa(para);
    fprintf('耗时: %.1f s\n', toc);
    PrintStats('标准模式', res);
    PlotSingle('标准模式', res, dataset);
end

function RunExpert(dataset, expert_params, para_baseline, para_optimized, optimized_loaded)
    % 优先使用优化后参数, 否则回退基线
    if optimized_loaded
        para = BuildExpertPara(para_optimized, expert_params);
        label = '专家模式 (优化后)';
    else
        para = BuildExpertPara(para_baseline, expert_params);
        label = '专家模式 (基线参数)';
    end
    fprintf('\n=== 运行%s ===\n', label);
    tic;
    res = HeartRateSolver_cas_chengfa(para);
    fprintf('耗时: %.1f s\n', toc);
    PrintStats(label, res);
    PlotSingle(label, res, dataset);
    PlotClassifierProba(res, dataset, para.classifier_mode);
end

function RunBoth(dataset, para_std, expert_params, para_exp_backend)
    % 标准模式
    fprintf('\n=== 运行标准模式 ===\n');
    tic;
    res_std = HeartRateSolver_cas_chengfa(para_std);
    t_std = toc;
    fprintf('耗时: %.1f s\n', t_std);
    PrintStats('标准模式', res_std);

    % 专家模式 (基线参数)
    para_exp = BuildExpertPara(para_exp_backend, expert_params);
    fprintf('\n=== 运行专家模式 (基线参数) ===\n');
    tic;
    res_exp = HeartRateSolver_cas_chengfa(para_exp);
    t_exp = toc;
    fprintf('耗时: %.1f s\n', t_exp);
    PrintStats('专家模式', res_exp);

    % 对比图
    PlotCompare2('标准模式', res_std, '专家模式 (基线)', res_exp, dataset);
    PlotClassifierProba(res_exp, dataset, para_exp.classifier_mode);
end

function RunCompare(dataset, data_file, expert_params, para_baseline, para_optimized, optimized_loaded)
    if ~optimized_loaded
        error('未找到优化后参数文件, 无法对比。请先运行 AutoOptimize_Bayes_Search_cas_chengfa。');
    end

    % 专家模式 - 基线参数
    para_bl = BuildExpertPara(para_baseline, expert_params);
    fprintf('\n=== 运行专家模式 (优化前 - 基线参数) ===\n');
    tic;
    res_bl = HeartRateSolver_cas_chengfa(para_bl);
    fprintf('耗时: %.1f s\n', toc);
    PrintStats('专家(优化前)', res_bl);

    % 专家模式 - 优化后参数
    para_op = BuildExpertPara(para_optimized, expert_params);
    fprintf('\n=== 运行专家模式 (优化后) ===\n');
    tic;
    res_op = HeartRateSolver_cas_chengfa(para_op);
    fprintf('耗时: %.1f s\n', toc);
    PrintStats('专家(优化后)', res_op);

    % 四子图对比: 优化前HF/ACC + 优化后HF/ACC
    PlotCompare4(res_bl, res_op, dataset);
    PlotClassifierProba(res_op, dataset, para_op.classifier_mode);

    % 优化效果摘要
    fprintf('\n=== 优化效果摘要 ===\n');
    es_bl = res_bl.err_stats;
    es_op = res_op.err_stats;
    fprintf('%-14s | %8s | %8s | %8s\n', '指标', '优化前', '优化后', '改善');
    fprintf('---------------|----------|----------|----------\n');
    metrics = {'Fus-HF Total', 'Fus-HF Motion', 'Fus-ACC Total', 'Fus-ACC Motion'};
    bl_vals = [es_bl(4,1), es_bl(4,3), es_bl(5,1), es_bl(5,3)];
    op_vals = [es_op(4,1), es_op(4,3), es_op(5,1), es_op(5,3)];
    for i = 1:length(metrics)
        delta = bl_vals(i) - op_vals(i);
        sign = '+'; if delta < 0, sign = '-'; end
        fprintf('%-14s | %7.2f  | %7.2f  | %s%.2f\n', metrics{i}, bl_vals(i), op_vals(i), sign, abs(delta));
    end
end

%% ========================================================================
%  参数构建
%  ========================================================================

function [para, loaded] = LoadBackEndParams(data_dir, dataset, data_file)
    bobi_param_file = fullfile(data_dir, sprintf('Best_Params_Result_multi_%s_processed.mat', dataset));
    loaded = false;

    para = struct();
    para.FileName = data_file;
    para.Time_Start = 1;
    para.Time_Buffer = 10;
    para.Calib_Time = 30;
    para.Fs_Target = 25;
    para.Motion_Th_Scale = 2.5;
    para.Spec_Penalty_Enable = 1;
    para.Spec_Penalty_Weight = 0.2;

    if isfile(bobi_param_file)
        tmp = load(bobi_param_file, 'Best_Para_HF');
        bp = tmp.Best_Para_HF;
        loaded = true;
        fprintf('基线参数: 从 %s 加载', bobi_param_file);
        fields = {'Max_Order', 'Spec_Penalty_Width', 'HR_Range_Hz', 'Slew_Limit_BPM', ...
            'Slew_Step_BPM', 'HR_Range_Rest', 'Slew_Limit_Rest', 'Slew_Step_Rest', ...
            'Smooth_Win_Len', 'Time_Bias'};
        defaults = {16, 0.2, 25/60, 10, 7, 30/60, 6, 4, 7, 5};
        for i = 1:length(fields)
            if isfield(bp, fields{i}), para.(fields{i}) = bp.(fields{i});
            else, para.(fields{i}) = defaults{i}; end
        end
    else
        fprintf('基线参数: 使用默认值');
        para.Max_Order = 16;
        para.Spec_Penalty_Width = 0.2;
        para.HR_Range_Hz = 25/60;
        para.Slew_Limit_BPM = 10;
        para.Slew_Step_BPM = 7;
        para.HR_Range_Rest = 30/60;
        para.Slew_Limit_Rest = 6;
        para.Slew_Step_Rest = 4;
        para.Smooth_Win_Len = 7;
        para.Time_Bias = 5;
    end
end

function para = BuildExpertPara(para_backend, expert_params)
    para = para_backend;
    para.expert_mode = true;
    para.classifier_mode = 'window';
    para.model_path = 'models';
    para.expert_params = expert_params;
end

%% ========================================================================
%  可视化
%  ========================================================================

function PrintStats(label, res)
    es = res.err_stats;
    fprintf('\n%s 误差统计 (AAE, BPM):\n', label);
    fprintf('%-14s | %8s | %8s | %8s\n', 'Method', 'Total', 'Rest', 'Motion');
    fprintf('---------------|----------|----------|----------\n');
    names = {'LMS-HF', 'LMS-ACC', 'FFT', 'Fus-HF', 'Fus-ACC'};
    for i = 1:5
        fprintf('%-14s | %7.2f  | %7.2f  | %7.2f\n', names{i}, es(i,1), es(i,2), es(i,3));
    end
end

function PlotSingle(label, res, dataset)
    figure('Name', sprintf('QuickTest: %s', label), 'Color', 'w', 'Position', [50 50 1000 500]);
    HR = res.HR; T_Pred = res.T_Pred;
    motion_bg = HR(:, 8) * 220;
    a = area(T_Pred, motion_bg, 'FaceColor', [0.94 0.94 0.96], 'EdgeColor', 'none', 'BaseValue', 0);
    hold on;
    plot(HR(:,1), HR(:,2)*60, 'k-', 'LineWidth', 2.5, 'DisplayName', '真值');
    plot(T_Pred, HR(:,5)*60, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, 'DisplayName', 'FFT');
    plot(T_Pred, HR(:,6)*60, 'm.-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Fusion-HF');
    plot(T_Pred, HR(:,7)*60, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Fusion-ACC');
    uistack(a, 'bottom');
    xlabel('Time (s)'); ylabel('HR (BPM)'); ylim([50 200]); grid on;
    legend('Location', 'bestoutside');
    es = res.err_stats;
    title(sprintf('%s | %s | Fus-HF T=%.1f M=%.1f | Fus-ACC T=%.1f M=%.1f', ...
        label, dataset, es(4,1), es(4,3), es(5,1), es(5,3)), 'FontSize', 11);
end

function PlotCompare4(res_bl, res_op, dataset)
% 四子图: 优化前HF / 优化前ACC / 优化后HF / 优化后ACC
% 各子图仅显示对应方案的曲线, 标题含 Motion/Total AAE (与摘要表一致)
    figure('Name', 'QuickTest: 优化前后四方案对比', 'Color', 'w', 'Position', [50 50 1600 1000]);

    % 方案定义: {结果, 标签, 主曲线列, Fusion列(用于标题), 路径名}
    schemes = {
        res_bl,  '优化前', 6, 4, 'HF';    % Fusion-HF: col6=融合结果, err_stats row4
        res_bl,  '优化前', 7, 5, 'ACC';   % Fusion-ACC: col7=融合结果, err_stats row5
        res_op,  '优化后', 6, 4, 'HF';
        res_op,  '优化后', 7, 5, 'ACC'
    };

    for p = 1:4
        ax = subplot(2, 2, p);
        R = schemes{p,1};
        label = schemes{p,2};
        fus_col = schemes{p,3};
        err_row = schemes{p,4};
        path_name = schemes{p,5};

        HR = R.HR;
        T_Pred = R.T_Pred;
        es = R.err_stats;

        motion_bg = HR(:, 8) * 220;
        a = area(T_Pred, motion_bg, 'FaceColor', [0.94 0.94 0.96], 'EdgeColor', 'none', 'BaseValue', 0);
        hold on;
        plot(HR(:,1), HR(:,2)*60, 'k-', 'LineWidth', 2.5, 'DisplayName', '真值');
        plot(T_Pred, HR(:,5)*60, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1, 'DisplayName', 'FFT');

        if strcmp(path_name, 'HF')
            plot(T_Pred, HR(:,fus_col)*60, 'm-', 'LineWidth', 2, 'DisplayName', 'Fusion-HF');
        else
            plot(T_Pred, HR(:,fus_col)*60, 'b-', 'LineWidth', 2, 'DisplayName', 'Fusion-ACC');
        end

        uistack(a, 'bottom');
        ylabel('HR (BPM)'); ylim([50 200]); grid on; set(gca, 'GridAlpha', 0.3);
        legend('Location', 'bestoutside');

        title(sprintf('%s Fusion-%s | Total=%.2f  Motion=%.2f', ...
            label, path_name, es(err_row,1), es(err_row,3)), ...
            'FontSize', 11, 'FontWeight', 'bold');

        if p >= 3, xlabel('Time (s)'); end
    end
    linkaxes(findobj(gcf, 'type', 'axes'), 'x');
end

function PlotCompare2(label1, res1, label2, res2, dataset)
    figure('Name', 'QuickTest 对比', 'Color', 'w', 'Position', [50 50 1400 900]);
    data_sets = {res1, res2};
    titles = {label1, label2};
    for p = 1:2
        ax = subplot(2, 1, p);
        R = data_sets{p};
        HR = R.HR;
        T_Pred = R.T_Pred;
        motion_bg = HR(:, 8) * 220;
        a = area(T_Pred, motion_bg, 'FaceColor', [0.94 0.94 0.96], 'EdgeColor', 'none', 'BaseValue', 0);
        hold on;
        plot(HR(:,1), HR(:,2)*60, 'k-', 'LineWidth', 2.5, 'DisplayName', '真值');
        plot(T_Pred, HR(:,5)*60, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, 'DisplayName', 'FFT');
        plot(T_Pred, HR(:,6)*60, 'm.-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Fusion-HF');
        plot(T_Pred, HR(:,7)*60, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Fusion-ACC');
        uistack(a, 'bottom');
        ylabel('HR (BPM)'); ylim([50 200]); grid on; set(gca, 'GridAlpha', 0.3);
        es = R.err_stats;
        title(sprintf('%s | %s | Fus-HF T=%.1f M=%.1f | Fus-ACC T=%.1f M=%.1f', ...
            titles{p}, dataset, es(4,1), es(4,3), es(5,1), es(5,3)), 'FontSize', 11, 'FontWeight', 'bold');
        legend('Location', 'bestoutside');
        if p == 2, xlabel('Time (s)'); end
    end
    linkaxes(findobj(gcf, 'type', 'axes'), 'x');
end

function PlotClassifierProba(res, dataset, classifier_mode)
    if size(res.HR, 2) >= 12
        figure('Name', '分类器概率时程', 'Color', 'w', 'Position', [50 50 1000 300]);
        area(res.T_Pred, res.HR(:,10:12));
        legend('arm\_curl', 'jump\_rope', 'push\_up', 'Location', 'best');
        title(sprintf('分类器概率 [%s, %s]', dataset, classifier_mode));
        xlabel('Time (s)'); ylabel('Probability'); ylim([0 1]); grid on;
    end
end
