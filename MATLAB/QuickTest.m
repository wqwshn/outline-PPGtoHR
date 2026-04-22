function QuickTest(data_key, mode)
% QuickTest 快速调试脚本: 一条命令运行标准/专家模式并对比结果
%
% 用法:
%   QuickTest                    % 默认: bobi1 专家模式
%   QuickTest('bobi2')           % 指定数据集, 专家模式
%   QuickTest('bobi1', 'std')    % 标准模式 (基线)
%   QuickTest('bobi1', 'expert') % 专家模式
%   QuickTest('bobi1', 'both')   % 两种模式对比 (默认)

if nargin < 1, data_key = 'bobi1'; end
if nargin < 2, mode = 'both'; end

%% 数据集映射
data_map = struct( ...
    'bobi1',     'dataformatlab\multi_bobi1_processed.mat', ...
    'bobi2',     'dataformatlab\multi_bobi2_processed.mat', ...
    'fuwo1',     'dataformatlab\multi_fuwo1_processed.mat', ...
    'fuwo2',     'dataformatlab\multi_fuwo2_processed.mat', ...
    'tiaosheng1','dataformatlab\multi_tiaosheng1_processed.mat', ...
    'tiaosheng2','dataformatlab\multi_tiaosheng2_processed.mat', ...
    'tiaosheng3','dataformatlab\multi_tiaosheng3_processed.mat', ...
    'wanju1',    'dataformatlab\multi_wanju1_processed.mat', ...
    'wanju2',    'dataformatlab\multi_wanju2_processed.mat', ...
    'kaihe1',    'dataformatlab\multi_kaihe1_processed.mat', ...
    'kaihe2',    'dataformatlab\multi_kaihe2_processed.mat');

if ~isfield(data_map, data_key)
    error('未知数据集: %s\n可用: bobi1, bobi2, fuwo1/2, tiaosheng1/2/3, wanju1/2, kaihe1/2', data_key);
end

data_file = data_map.(data_key);
if ~isfile(data_file)
    error('数据文件不存在: %s', data_file);
end

%% 专家参数: 从各运动场景的贝叶斯优化结果中提取
expert_files = struct( ...
    'arm_curl',   'dataformatlab\Best_Params_Result_multi_wanju1_processed.mat', ...
    'jump_rope',  'dataformatlab\Best_Params_Result_multi_tiaosheng2_processed.mat', ...
    'push_up',    'dataformatlab\Best_Params_Result_multi_fuwo2_processed.mat');

%% 构建通用基础参数
base = build_base_para(data_file);

%% === 标准模式 ===
if strcmp(mode, 'std') || strcmp(mode, 'both')
    fprintf('\n======== 标准模式 ========\n');
    % 尝试加载该数据集已有的最优参数
    bp_file = sprintf('dataformatlab\\Best_Params_Result_%s', ...
        strrep(data_file, 'dataformatlab\', ''));
    if isfile(bp_file)
        bp = load(bp_file);
        % 使用 HF 最优参数
        para_std = bp.Best_Para_HF;
        fprintf('已加载最优参数: %s\n', bp_file);
    else
        para_std = base;
        fprintf('使用默认参数 (未找到最优参数文件)\n');
    end
    para_std.expert_mode = false;
    para_std.FileName = data_file;
    tic;
    Res_std = HeartRateSolver_cas_chengfa(para_std);
    t_std = toc;
    print_result('标准模式', Res_std, t_std);
end

%% === 专家模式 ===
if strcmp(mode, 'expert') || strcmp(mode, 'both')
    fprintf('\n======== 专家模式 ========\n');

    % 加载专家参数
    expert_names = fieldnames(expert_files);
    ep = struct();
    for i = 1:length(expert_names)
        en = expert_names{i};
        ef = expert_files.(en);
        if isfile(ef)
            tmp = load(ef);
            % 从 HF 最优参数中提取前级参数
            bp_hf = tmp.Best_Para_HF;
            ep.(en) = struct( ...
                'Fs_Target',       bp_hf.Fs_Target, ...
                'Max_Order',       bp_hf.Max_Order, ...
                'LMS_Mu_Base',     0.01, ...
                'Num_Cascade_HF',  2, ...
                'Num_Cascade_Acc', 3);
            fprintf('  %s: Fs=%d, Order=%d\n', en, bp_hf.Fs_Target, bp_hf.Max_Order);
        else
            warning('  %s: 参数文件不存在 (%s), 使用默认值', en, ef);
            ep.(en) = struct( ...
                'Fs_Target', 25, 'Max_Order', 16, 'LMS_Mu_Base', 0.01, ...
                'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
        end
    end

    % 尝试加载波比跳基线的最优后级参数
    bp_file = sprintf('dataformatlab\\Best_Params_Result_%s', ...
        strrep(data_file, 'dataformatlab\', ''));
    if isfile(bp_file)
        bp = load(bp_file);
        para_exp = bp.Best_Para_HF;
        fprintf('后级参数: 已加载 %s\n', bp_file);
    else
        para_exp = base;
        fprintf('后级参数: 使用默认值\n');
    end

    para_exp.expert_mode = true;
    para_exp.classifier_mode = 'window';
    para_exp.model_path = 'models';
    para_exp.expert_params = ep;
    para_exp.FileName = data_file;

    tic;
    Res_exp = HeartRateSolver_cas_chengfa(para_exp);
    t_exp = toc;
    print_result('专家模式', Res_exp, t_exp);

    % 绘制分类器概率时程图
    if size(Res_exp.HR, 2) >= 12
        figure('Name', 'Expert Mode Results', 'Color', 'w', 'Position', [50, 50, 1200, 700]);

        % 子图1: 心率轨迹
        subplot(3,1,1);
        T_Pred = Res_exp.HR(:,1) + para_exp.Time_Bias;
        mask_m = Res_exp.HR(:, 8) == 1;
        area(T_Pred, Res_exp.HR(:,8)*220, 'FaceColor', [0.94 0.94 0.96], 'EdgeColor', 'none', 'BaseValue', 0);
        hold on;
        plot(Res_exp.HR(:,1), Res_exp.HR(:,2)*60, 'k-', 'LineWidth', 2.5, 'DisplayName', '真值');
        plot(T_Pred, Res_exp.HR(:,6)*60, 'm.-', 'LineWidth', 1.5, 'DisplayName', '融合-HF');
        plot(T_Pred, Res_exp.HR(:,7)*60, 'b.-', 'LineWidth', 1.5, 'DisplayName', '融合-ACC');
        uistack(findobj(gca, 'Type', 'area'), 'bottom');
        title(sprintf('专家模式: %s | 运动AAE=%.2f BPM', data_key, Res_exp.err_stats(4,3)*60));
        ylabel('HR (BPM)'); ylim([50 200]); grid on; legend('Location', 'bestoutside');

        % 子图2: 分类器概率
        subplot(3,1,2);
        area(T_Pred, Res_exp.HR(:,10:12));
        legend({'arm\_curl', 'jump\_rope', 'push\_up'}, 'Location', 'best');
        ylabel('概率'); ylim([0 1.05]); grid on;
        title('分类器概率分布');
    end
end

%% === 对比输出 ===
if strcmp(mode, 'both') && exist('Res_std', 'var') && exist('Res_exp', 'var')
    fprintf('\n======== 对比 ========\n');
    fprintf('%-12s | %8s | %8s | %8s\n', '指标', '标准', '专家', '改善');
    fprintf('-------------------------------------------------\n');
    labels = {'全局AAE', '静息AAE', '运动AAE'};
    for c = 1:3
        v_std = Res_std.err_stats(4, c) * 60;
        v_exp = Res_exp.err_stats(4, c) * 60;
        delta = v_std - v_exp;
        sign = '+'; if delta < 0, sign = ''; end
        fprintf('%-12s | %6.2f   | %6.2f   | %s%.2f\n', labels{c}, v_std, v_exp, sign, delta);
    end
end

end

%% ========== 辅助函数 ==========

function para = build_base_para(data_file)
    para = struct();
    para.FileName = data_file;
    para.Fs_Target = 25;
    para.Time_Start = 1;
    para.Time_Buffer = 10;
    para.Calib_Time = 30;
    para.Time_Bias = 5;
    para.Motion_Th_Scale = 2.5;
    para.Spec_Penalty_Enable = 1;
    para.Spec_Penalty_Weight = 0.2;
    para.Spec_Penalty_Width = 0.2;
    para.Max_Order = 16;
    para.HR_Range_Hz = 30/60;
    para.Slew_Limit_BPM = 10;
    para.Slew_Step_BPM = 7;
    para.HR_Range_Rest = 30/60;
    para.Slew_Limit_Rest = 6;
    para.Slew_Step_Rest = 4;
    para.Smooth_Win_Len = 7;
end

function print_result(label, Res, elapsed)
    fprintf('  运行耗时: %.1fs\n', elapsed);
    fprintf('  %-10s 全局=%.2f  静息=%.2f  运动=%.2f (BPM)\n', '融合HF:', ...
        Res.err_stats(4,:)*60);
    fprintf('  %-10s 全局=%.2f  静息=%.2f  运动=%.2f (BPM)\n', '融合ACC:', ...
        Res.err_stats(5,:)*60);
end
