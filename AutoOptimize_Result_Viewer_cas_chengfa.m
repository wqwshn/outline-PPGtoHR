%% 自动参数寻优结果展示 (AutoOptimize_Result_Viewer)
% 功能：
% 1. 读取 AutoOptimize_Bayes_Search 生成的 'Best_Params_Result.mat'。
% 2. 复现最优结果：分别为 HF最优配置 和 ACC最优配置 运行 HeartRateSolver。
% 3. 绘制双子图对比 (Subplots) 并打印详细统计表格。
% 
% 适配版本：HeartRateSolver (ACC/HF 独立路径版 - 统一ACC判据)

clc; clear; close all;

%% 1. 加载寻优结果
ResultFile = 'Best_Params_20260119dualtiaosheng1_processed.mat';
if ~isfile(ResultFile)
    error('未找到结果文件 %s，请先运行 AutoOptimize_Bayes_Search.m', ResultFile);
end
load(ResultFile); % 加载 Best_Para_HF, Best_Para_ACC, SearchSpace 等

fprintf('成功加载寻优结果。\n');
fprintf('Fusion(HF)  记录最低误差: %.4f\n', Min_Err_HF);
fprintf('Fusion(ACC) 记录最低误差: %.4f\n', Min_Err_ACC);

%% 2. 重新计算最优结果
fprintf('正在复现最优结果...\n');

% --- 计算 HF 最优数据 ---
Final_Res_HF = HeartRateSolver_cas_chengfa(Best_Para_HF);
HR_HF = Final_Res_HF.HR;
T_Pred_HF = Final_Res_HF.T_Pred;
Th_HF_Case = Final_Res_HF.Motion_Threshold; 
Stats_HF = Final_Res_HF.err_stats; 

% 提取指标用于图例
E_FFT_1    = Stats_HF(3, 1);
E_FusHF_1  = Stats_HF(4, 1);
E_FusACC_1 = Stats_HF(5, 1);

% --- 计算 ACC 最优数据 ---
Final_Res_ACC = HeartRateSolver_cas_chengfa(Best_Para_ACC);
HR_ACC = Final_Res_ACC.HR;
T_Pred_ACC = Final_Res_ACC.T_Pred;
Th_ACC_Case = Final_Res_ACC.Motion_Threshold;
Stats_ACC = Final_Res_ACC.err_stats;

% 提取指标用于图例
E_FFT_2    = Stats_ACC(3, 1);
E_FusHF_2  = Stats_ACC(4, 1);
E_FusACC_2 = Stats_ACC(5, 1);

% [新增] 保存滤波后的PPG数据为CSV格式
fprintf('\n正在保存滤波后的PPG数据...\n');

% 获取数据文件名（不含路径和扩展名），用于生成输出文件名
% 从已加载的变量中获取 Data_FileName（如果存在）
if exist('Data_FileName', 'var')
    [~, data_basename, ~] = fileparts(Data_FileName);
else
    % 向后兼容：如果参数文件中没有Data_FileName字段，使用默认名称
    data_basename = 'result';
end

% 保存HF路径滤波后的PPG数据
csv_hf_filename = sprintf('PPG_LMS_HF_%s.csv', data_basename);
Save_PPG_To_CSV(Final_Res_HF.PPG_LMS_HF, Final_Res_HF.Time_Windows, HR_HF, csv_hf_filename, Best_Para_HF.Fs_Target);
fprintf('  HF滤波数据已保存至: %s\n', csv_hf_filename);

% 保存ACC路径滤波后的PPG数据
csv_acc_filename = sprintf('PPG_LMS_ACC_%s.csv', data_basename);
Save_PPG_To_CSV(Final_Res_ACC.PPG_LMS_ACC, Final_Res_ACC.Time_Windows, HR_ACC, csv_acc_filename, Best_Para_ACC.Fs_Target);
fprintf('  ACC滤波数据已保存至: %s\n', csv_acc_filename);

%% 3. 组合绘图 (Combined Subplots)
figure('Name', 'Bayes Optimization Comparison: HF vs ACC Best Cases', 'Color', 'w', 'Position', [50, 50, 1200, 900]);

% === 子图 1: HF 最优参数结果 ===
ax1 = subplot(2, 1, 1);
% 1. 背景 (使用第8列 ACC Flag)
motion_area_1 = HR_HF(:, 8) * 220; 
a1 = area(T_Pred_HF, motion_area_1, 'FaceColor', [0.94 0.94 0.96], 'EdgeColor', 'none', 'BaseValue', 0);
hold on;
% 2. 曲线
h_fft1 = plot(T_Pred_HF, HR_HF(:,5)*60, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, 'DisplayName', sprintf('纯FFT (AAE=%.1f)', E_FFT_1));
h_lms_h1 = plot(T_Pred_HF, HR_HF(:,3)*60, 'b:', 'LineWidth', 1.5, 'DisplayName', '纯LMS-热膜');
h_lms_a1 = plot(T_Pred_HF, HR_HF(:,4)*60, 'm:', 'LineWidth', 1.5, 'DisplayName', '纯LMS-Acc');
% 重点: 这里的 Target 是 Fusion-HF
h_fus_h1 = plot(T_Pred_HF, HR_HF(:,6)*60, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', sprintf('融合-热膜 (AAE=%.1f) [TARGET]', E_FusHF_1));
h_fus_a1 = plot(T_Pred_HF, HR_HF(:,7)*60, 'm.-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', sprintf('融合-Acc (AAE=%.1f)', E_FusACC_1));
h_ref1 = plot(HR_HF(:,1), HR_HF(:,2)*60, 'k-', 'LineWidth', 2.5, 'DisplayName', '真值 (Ref)');
% 3. 美化
uistack(a1, 'bottom'); 
title(sprintf('(1) Fusion(HF) 最优参数结果 | Min Error: %.4f | Motion Th (ACC Unified): %.3f', ...
    Min_Err_HF, Th_HF_Case(1)), 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Heart Rate (BPM)'); 
ylim([50 90]); xlim([min(T_Pred_HF) max(T_Pred_HF)]);
grid on; set(gca, 'GridAlpha', 0.3);
legend([h_ref1, h_fft1, h_fus_h1, h_fus_a1, h_lms_h1, h_lms_a1, a1], 'Location', 'bestoutside', 'NumColumns', 1);

% === 子图 2: ACC 最优参数结果 ===
ax2 = subplot(2, 1, 2);
% 1. 背景 (使用第8列 ACC Flag)
motion_area_2 = HR_ACC(:, 8) * 220; 
a2 = area(T_Pred_ACC, motion_area_2, 'FaceColor', [0.94 0.94 0.96], 'EdgeColor', 'none', 'BaseValue', 0);
hold on;
% 2. 曲线
h_fft2 = plot(T_Pred_ACC, HR_ACC(:,5)*60, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, 'DisplayName', sprintf('纯FFT (AAE=%.1f)', E_FFT_2));
h_lms_h2 = plot(T_Pred_ACC, HR_ACC(:,3)*60, 'b:', 'LineWidth', 1.5, 'DisplayName', '纯LMS-热膜');
h_lms_a2 = plot(T_Pred_ACC, HR_ACC(:,4)*60, 'm:', 'LineWidth', 1.5, 'DisplayName', '纯LMS-Acc');
% 重点: 这里的 Target 是 Fusion-Acc
h_fus_h2 = plot(T_Pred_ACC, HR_ACC(:,6)*60, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', sprintf('融合-热膜 (AAE=%.1f)', E_FusHF_2));
h_fus_a2 = plot(T_Pred_ACC, HR_ACC(:,7)*60, 'm.-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', sprintf('融合-Acc (AAE=%.1f) [TARGET]', E_FusACC_2));
h_ref2 = plot(HR_ACC(:,1), HR_ACC(:,2)*60, 'k-', 'LineWidth', 2.5, 'DisplayName', '真值 (Ref)');
% 3. 美化
uistack(a2, 'bottom'); 
title(sprintf('(2) Fusion(ACC) 最优参数结果 | Min Error: %.4f | Motion Th (ACC Unified): %.3f', ...
    Min_Err_ACC, Th_ACC_Case(1)), 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Heart Rate (BPM)'); xlabel('Time (s)');
ylim([50 90]); xlim([min(T_Pred_ACC) max(T_Pred_ACC)]);
grid on; set(gca, 'GridAlpha', 0.3);
legend([h_ref2, h_fft2, h_fus_h2, h_fus_a2, h_lms_h2, h_lms_a2, a2], 'Location', 'bestoutside', 'NumColumns', 1);

linkaxes([ax1, ax2], 'x');

%% 4. 详细误差表与参数对比
% --- 误差表 (HF Case) ---
fprintf('\n--- 图1: Fusion(HF) 最优参数误差详情 ---\n');
PrintDetailedStats(HR_HF, Final_Res_HF.HR_Ref_Interp);

% --- 误差表 (ACC Case) ---
fprintf('\n--- 图2: Fusion(ACC) 最优参数误差详情 ---\n');
PrintDetailedStats(HR_ACC, Final_Res_ACC.HR_Ref_Interp);

% --- 全参数对比表 ---
fprintf('\n=========================================================================\n');
fprintf('                方案极限能力全参数对比 (Bayes Optimization Result)         \n');
fprintf('=========================================================================\n');
fprintf('比较项             | HF 最优方案 (Best Fusion-HF) | ACC 最优方案 (Best Fusion-Acc)\n');
fprintf('-------------------|------------------------------|------------------------------\n');
% 误差指标对比
fprintf('Target Error (AAE) | %10.4f                   | %10.4f\n', Min_Err_HF, Min_Err_ACC);
fprintf('Motion Error       | %10.4f                   | %10.4f\n', Stats_HF(4,3), Stats_ACC(5,3));
fprintf('Rest Error         | %10.4f                   | %10.4f\n', Stats_HF(4,2), Stats_ACC(5,2));
fprintf('-------------------|------------------------------|------------------------------\n');
fprintf('配置参数详情       | HF Configuration             | ACC Configuration\n');
fprintf('-------------------|------------------------------|------------------------------\n');
% 打印 SearchSpace 中定义的所有参数
fprintf('Fs_Target          | %-28d | %-28d\n', Best_Para_HF.Fs_Target, Best_Para_ACC.Fs_Target);
fprintf('Max_Order          | %-28d | %-28d\n', Best_Para_HF.Max_Order, Best_Para_ACC.Max_Order);
fprintf('Spec_Penalty_Width | %-28.2f | %-28.2f\n', Best_Para_HF.Spec_Penalty_Width, Best_Para_ACC.Spec_Penalty_Width);
fprintf('-------------------|------------------------------|------------------------------\n');
fprintf('Motion HR Range    | %-28.3f | %-28.3f\n', Best_Para_HF.HR_Range_Hz, Best_Para_ACC.HR_Range_Hz);
fprintf('Motion Slew Limit  | %-28d | %-28d\n', Best_Para_HF.Slew_Limit_BPM, Best_Para_ACC.Slew_Limit_BPM);
fprintf('Motion Slew Step   | %-28d | %-28d\n', Best_Para_HF.Slew_Step_BPM, Best_Para_ACC.Slew_Step_BPM);
fprintf('-------------------|------------------------------|------------------------------\n');
fprintf('Rest HR Range      | %-28.3f | %-28.3f\n', Best_Para_HF.HR_Range_Rest, Best_Para_ACC.HR_Range_Rest);
fprintf('Rest Slew Limit    | %-28d | %-28d\n', Best_Para_HF.Slew_Limit_Rest, Best_Para_ACC.Slew_Limit_Rest);
fprintf('Rest Slew Step     | %-28d | %-28d\n', Best_Para_HF.Slew_Step_Rest, Best_Para_ACC.Slew_Step_Rest);
fprintf('-------------------|------------------------------|------------------------------\n');
fprintf('Smooth_Win_Len     | %-28d | %-28d\n', Best_Para_HF.Smooth_Win_Len, Best_Para_ACC.Smooth_Win_Len);
fprintf('Time_Bias          | %-28d | %-28d\n', Best_Para_HF.Time_Bias, Best_Para_ACC.Time_Bias);
fprintf('=========================================================================\n');

%% 辅助函数: 打印详细统计
function PrintDetailedStats(HR, HR_Ref_Interp)
    mask_motion = (HR(:, 8) == 1);
    mask_rest   = (HR(:, 8) == 0);
    col_indices = [3, 4, 5, 6, 7];
    col_names   = {'LMS(HF)', 'LMS(Acc)', 'Pure FFT', 'Fusion(HF)', 'Fusion(Acc)'};

    fprintf('%-12s | %-10s | %-10s | %-10s\n', 'Method', 'Total AAE', 'Rest AAE', 'Motion AAE');
    fprintf('------------------------------------------------------\n');
    for k = 1:length(col_indices)
        col = col_indices(k);
        abs_err = abs(HR(:, col) - HR_Ref_Interp) * 60;
        val_total  = mean(abs_err);
        val_rest   = mean(abs_err(mask_rest));
        val_motion = mean(abs_err(mask_motion));
        fprintf('%-12s | %6.2f      | %6.2f      | %6.2f\n', col_names{k}, val_total, val_rest, val_motion);
    end
end

%% 辅助函数: 保存PPG数据到CSV文件
function Save_PPG_To_CSV(PPG_Cell, Time_Windows, HR, csv_filename, Fs)
    % PPG_Cell: cell数组，每个元素是一个时间窗的PPG信号
    % Time_Windows: cell数组，每个元素是[起始时间, 结束时间]
    % HR: 心率矩阵，HR(:,1)是时间，HR(:,2)是参考真值（Hz）
    % csv_filename: 输出CSV文件名
    % Fs: 采样率（Hz），用于计算时间戳

    num_windows = length(PPG_Cell);
    dt = 1 / Fs;  % 采样间隔

    % 打开文件进行写入
    fid = fopen(csv_filename, 'w');
    if fid == -1
        error('无法创建文件: %s', csv_filename);
    end

    % 写入CSV表头
    fprintf(fid, 'Time(s),PPG_Value,HR_Time(s),HR_True(BPM)\n');

    % 遍历每个时间窗
    for win_idx = 1:num_windows
        ppg_data = PPG_Cell{win_idx};
        time_win = Time_Windows{win_idx};
        start_time = time_win(1);

        % 获取该时间窗对应的心率时间和真值
        if win_idx <= size(HR, 1)
            hr_time = HR(win_idx, 1);
            hr_true_bpm = HR(win_idx, 2) * 60;  % 转换为BPM
            hr_time_str = sprintf('%.6f', hr_time);
            hr_true_str = sprintf('%.4f', hr_true_bpm);
        else
            hr_time_str = '';
            hr_true_str = '';
        end

        % 计算该时间窗内每个样本的全局时间并写入
        for sample_idx = 1:length(ppg_data)
            ppg_time = start_time + (sample_idx - 1) * dt;
            ppg_value = ppg_data(sample_idx);

            fprintf(fid, '%.6f,%.6f', ppg_time, ppg_value);
            fprintf(fid, ',%s', hr_time_str);
            fprintf(fid, ',%s\n', hr_true_str);
        end
    end

    fclose(fid);
end