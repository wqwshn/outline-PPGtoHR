function QuickTest(dataset, mode)
%% QuickTest 快速调试脚本 - 标准模式 vs 专家模式对比
% 用法:
%   QuickTest                  % bobi1, 标准模式 vs 专家模式对比
%   QuickTest('bobi2')         % 换数据集
%   QuickTest('bobi1', 'std')    % 仅标准模式
%   QuickTest('bobi1', 'expert') % 仅专家模式
%
% 前置: 需先运行 python export_classifier_to_mat.py 生成 models/ 目录

arguments
    dataset (1,:) char = 'bobi1'
    mode    (1,:) char = 'both'   % 'both' | 'std' | 'expert'
end

clc; close all;

%% 1. 数据文件定位
data_dir = 'dataformatlab';
data_file = fullfile(data_dir, sprintf('multi_%s_processed.mat', dataset));
if ~isfile(data_file)
    error('数据文件不存在: %s', data_file);
end
fprintf('数据文件: %s\n', data_file);

%% 2. 专家参数加载 (从各运动场景的贝叶斯优化结果中提取)
expert_sources = struct( ...
    'arm_curl',  fullfile(data_dir, 'Best_Params_Result_multi_wanju1_processed.mat'), ...
    'jump_rope', fullfile(data_dir, 'Best_Params_Result_multi_tiaosheng2_processed.mat'), ...
    'push_up',   fullfile(data_dir, 'Best_Params_Result_multi_fuwo2_processed.mat'));

expert_names = fieldnames(expert_sources);
expert_params = struct();
fprintf('\n--- 加载专家参数 ---\n');
for i = 1:length(expert_names)
    en = expert_names{i};
    ef = expert_sources.(en);
    if isfile(ef)
        tmp = load(ef, 'Best_Para_HF');
        bp = tmp.Best_Para_HF;
        expert_params.(en) = struct( ...
            'Fs_Target', bp.Fs_Target, 'Max_Order', bp.Max_Order, ...
            'LMS_Mu_Base', 0.01, 'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
        fprintf('  %s: Fs=%dHz, MaxOrder=%d [from %s]\n', en, bp.Fs_Target, bp.Max_Order, ef);
    else
        expert_params.(en) = struct( ...
            'Fs_Target', 25, 'Max_Order', 16, ...
            'LMS_Mu_Base', 0.01, 'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
        fprintf('  %s: 文件不存在, 使用默认值 (Fs=25, MaxOrder=16)\n', en);
    end
end

%% 3. 基线参数 (从波比跳的贝叶斯优化结果加载, 无则用默认值)
bobi_param_file = fullfile(data_dir, sprintf('Best_Params_Result_multi_%s_processed.mat', dataset));
base_para_loaded = false;
if isfile(bobi_param_file)
    tmp = load(bobi_param_file, 'Best_Para_HF');
    bp_base = tmp.Best_Para_HF;
    base_para_loaded = true;
    fprintf('\n基线参数: 从 %s 加载\n', bobi_param_file);
else
    fprintf('\n基线参数: 使用默认值 (未找到 %s)\n', bobi_param_file);
end

%% 4. 构建标准模式参数
para_std = struct();
para_std.FileName = data_file;
para_std.Time_Start = 1;
para_std.Time_Buffer = 10;
para_std.Calib_Time = 30;
para_std.Fs_Target = 25;
para_std.Motion_Th_Scale = 2.5;
para_std.Spec_Penalty_Enable = 1;
para_std.Spec_Penalty_Weight = 0.2;

if base_para_loaded
    % 从基线文件中提取后级参数
    if isfield(bp_base, 'Max_Order'),         para_std.Max_Order = bp_base.Max_Order;         else, para_std.Max_Order = 16; end
    if isfield(bp_base, 'Spec_Penalty_Width'), para_std.Spec_Penalty_Width = bp_base.Spec_Penalty_Width; else, para_std.Spec_Penalty_Width = 0.2; end
    if isfield(bp_base, 'HR_Range_Hz'),        para_std.HR_Range_Hz = bp_base.HR_Range_Hz;     else, para_std.HR_Range_Hz = 25/60; end
    if isfield(bp_base, 'Slew_Limit_BPM'),     para_std.Slew_Limit_BPM = bp_base.Slew_Limit_BPM; else, para_std.Slew_Limit_BPM = 10; end
    if isfield(bp_base, 'Slew_Step_BPM'),      para_std.Slew_Step_BPM = bp_base.Slew_Step_BPM;   else, para_std.Slew_Step_BPM = 7; end
    if isfield(bp_base, 'HR_Range_Rest'),       para_std.HR_Range_Rest = bp_base.HR_Range_Rest;   else, para_std.HR_Range_Rest = 30/60; end
    if isfield(bp_base, 'Slew_Limit_Rest'),     para_std.Slew_Limit_Rest = bp_base.Slew_Limit_Rest; else, para_std.Slew_Limit_Rest = 6; end
    if isfield(bp_base, 'Slew_Step_Rest'),      para_std.Slew_Step_Rest = bp_base.Slew_Step_Rest;   else, para_std.Slew_Step_Rest = 4; end
    if isfield(bp_base, 'Smooth_Win_Len'),      para_std.Smooth_Win_Len = bp_base.Smooth_Win_Len;   else, para_std.Smooth_Win_Len = 7; end
    if isfield(bp_base, 'Time_Bias'),           para_std.Time_Bias = bp_base.Time_Bias;             else, para_std.Time_Bias = 5; end
else
    % 默认参数
    para_std.Max_Order = 16;
    para_std.Spec_Penalty_Width = 0.2;
    para_std.HR_Range_Hz = 25/60;
    para_std.Slew_Limit_BPM = 10;
    para_std.Slew_Step_BPM = 7;
    para_std.HR_Range_Rest = 30/60;
    para_std.Slew_Limit_Rest = 6;
    para_std.Slew_Step_Rest = 4;
    para_std.Smooth_Win_Len = 7;
    para_std.Time_Bias = 5;
end

%% 5. 构建专家模式参数 (在标准参数基础上覆盖)
para_exp = para_std;
para_exp.expert_mode = true;
para_exp.classifier_mode = 'window';
para_exp.model_path = 'models';
para_exp.expert_params = expert_params;

% 尝试从专家模式优化结果加载后级参数
expert_param_file = fullfile(data_dir, sprintf('Best_Params_Expert_Result_multi_%s_processed.mat', dataset));
if isfile(expert_param_file)
    tmp = load(expert_param_file, 'Best_Para_Expert_ACC');
    bp_exp = tmp.Best_Para_Expert_ACC;
    % 用专家模式专属参数覆盖后级参数
    override_fields = {'Spec_Penalty_Width', 'Spec_Penalty_Weight', ...
        'HR_Range_Hz', 'Slew_Limit_BPM', 'Slew_Step_BPM', ...
        'HR_Range_Rest', 'Slew_Limit_Rest', 'Slew_Step_Rest', ...
        'Smooth_Win_Len', 'Time_Bias'};
    for fi = 1:length(override_fields)
        fn = override_fields{fi};
        if isfield(bp_exp, fn)
            para_exp.(fn) = bp_exp.(fn);
        end
    end
    fprintf('专家后级参数: 从 %s 加载\n', expert_param_file);
else
    fprintf('专家后级参数: 使用标准模式基线值 (未找到 %s)\n', expert_param_file);
end

%% 6. 运行解算
res_std = []; res_exp = [];
if strcmp(mode, 'both') || strcmp(mode, 'std')
    fprintf('\n=== 运行标准模式 ===\n');
    tic;
    res_std = HeartRateSolver_cas_chengfa(para_std);
    t_std = toc;
    fprintf('标准模式耗时: %.1f s\n', t_std);
    PrintStats('标准模式', res_std);
end

if strcmp(mode, 'both') || strcmp(mode, 'expert')
    fprintf('\n=== 运行专家模式 ===\n');
    tic;
    res_exp = HeartRateSolver_cas_chengfa(para_exp);
    t_exp = toc;
    fprintf('专家模式耗时: %.1f s\n', t_exp);
    PrintStats('专家模式', res_exp);
end

%% 7. 绘图对比
if ~isempty(res_std) && ~isempty(res_exp)
    % 双子图对比
    figure('Name', 'QuickTest: 标准模式 vs 专家模式', 'Color', 'w', 'Position', [50 50 1400 900]);

    data_sets = {res_std, res_exp};
    titles = {'标准模式', '专家模式 (频谱融合)'};
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

        % 标题含关键指标
        es = R.err_stats;
        title(sprintf('%s | 数据: %s | Fusion-HF Total=%.1f Motion=%.1f | Fusion-ACC Total=%.1f Motion=%.1f', ...
            titles{p}, dataset, es(4,1), es(4,3), es(5,1), es(5,3)), 'FontSize', 11, 'FontWeight', 'bold');
        legend('Location', 'bestoutside');
        if p == 2, xlabel('Time (s)'); end
    end
    linkaxes(findobj(gcf, 'type', 'axes'), 'x');

    % 专家模式分类器概率图
    if size(res_exp.HR, 2) >= 12
        figure('Name', '分类器概率时程', 'Color', 'w', 'Position', [50 50 1000 300]);
        area(res_exp.T_Pred, res_exp.HR(:,10:12));
        legend('arm\_curl', 'jump\_rope', 'push\_up', 'Location', 'best');
        title(sprintf('分类器概率 [%s, %s]', dataset, para_exp.classifier_mode));
        xlabel('Time (s)'); ylabel('Probability'); ylim([0 1]); grid on;
    end

elseif ~isempty(res_std)
    PlotSingle('标准模式', res_std, dataset);
elseif ~isempty(res_exp)
    PlotSingle('专家模式', res_exp, dataset);
    if size(res_exp.HR, 2) >= 12
        figure('Name', '分类器概率时程', 'Color', 'w', 'Position', [50 50 1000 300]);
        area(res_exp.T_Pred, res_exp.HR(:,10:12));
        legend('arm\_curl', 'jump\_rope', 'push\_up', 'Location', 'best');
        title(sprintf('分类器概率 [%s, %s]', dataset, para_exp.classifier_mode));
        xlabel('Time (s)'); ylabel('Probability'); ylim([0 1]); grid on;
    end
end

fprintf('\n=== QuickTest 完成 ===\n');
end

%% 辅助函数
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
