function [Result] = HeartRateSolver_cas_chengfa(para)
%% HeartRateSolver_cas - 核心心率解算函数 (级联版 + FFT频谱惩罚对照)
% 功能：根据传入的 para 参数结构体，读取数据并执行 LMS+FFT 融合算法
%
% 修改说明 (按用户要求):
% 1. [Path C 修改] 纯 FFT 路径现在开启了频谱惩罚功能。
%    - 参考信号：使用 ACC Z轴 (Sig_a{3})，与 ACC LMS 路径s保持一致。
%    - 目的：在运动段计算中，通过给纯 FFT 也加上频谱惩罚，来更公平地对比 LMS 算法本身带来的去噪增益。
% 2. 运动检测：统一采用 ACC 信号标准差。
%
% 输入：para (包含路径、LMS参数、FFT追踪参数、带通滤波参数等)
% 输出：Result 结构体

%% 1. 数据加载与预处理
% 检查文件是否存在
if ~isfile(para.FileName)
    error(['文件不存在: ' para.FileName]); 
end
load(para.FileName, 'data', 'ref_data'); 

raw_data = table2array(data);

% % 全局固定参数(老数据包格式)
% Fs_Origin = 125;      % 原始采样率
% Col_PPG  = 6;           
% Col_HF1  = 4; Col_HF2 = 5;        
% Col_Acc  = [8, 9, 10];

% 全局固定参数(适配新数据包格式（多光谱）)
Fs_Origin = 100;      % 原始采样率
Col_PPG  = 6;           
Col_HF1  = 4; Col_HF2 = 5;        
Col_Acc  = [9, 10, 11];        

% % Spo2的数据格式
% Fs_Origin = 125;      % 原始采样率
% Col_PPG  = 6;           
% Col_HF1  = 2; Col_HF2 = 3;        
% Col_Acc  = [8, 9, 10]; 

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

%% 2. 运动阈值校准 (统一仅使用 ACC 阈值)
calib_len = min(round(para.Calib_Time * Fs), length(ppg));

% 2.1 ACC 阈值计算 (保留并作为唯一标准)
acc_mag = sqrt(accx.^2 + accy.^2 + accz.^2); 
acc_baseline_std = std(acc_mag(1:calib_len));
Motion_Threshold_ACC = para.Motion_Th_Scale * acc_baseline_std;

%% 3. 主循环初始化
Win_Len = 8; Win_Step = 1;
time_end = length(ppg_ori)/Fs - para.Time_Buffer;

% HR 矩阵列定义: 
% 1:Time, 2:Ref, 
% 3:Pure_LMS_HF, 4:Pure_LMS_ACC, 5:Pure_FFT, 
% 6:Fusion_HF,   7:Fusion_ACC,    
% 8:Motion_Flag_ACC(0/1), 9:Motion_Flag_HF(0/1) 
HR = zeros(1, 9); 

stop_flag = 1;
times = 1;
time_1 = para.Time_Start;

% LMS 固定参数
Num_Cascade_HF = 2; Num_Cascade_Acc = 3; LMS_Mu_Base = 0.01;

%% 4. 核心处理循环
while stop_flag
    time_2 = time_1 + Win_Len;
    idx_s = round(time_1*Fs) + 1;
    idx_e = round(time_2*Fs);
    if idx_e > length(ppg), break; end
    
    % 截取信号
    Sig_p = ppg(idx_s:idx_e);
    Sig_h = {hotf1(idx_s:idx_e), hotf2(idx_s:idx_e)};
    Sig_a = {accx(idx_s:idx_e), accy(idx_s:idx_e), accz(idx_s:idx_e)};
    
    Sig_acc_mag = acc_mag(idx_s:idx_e);
    
    HR(times, 1) = time_1; 
    HR(times, 2) = Find_realHR('dummy', time_1, HR_Ref_Data); 
    
    % =====================================================================
    % 4.1 运动状态判断 (统一修改为 ACC 标准)
    % =====================================================================
    is_motion = std(Sig_acc_mag) > Motion_Threshold_ACC;
    
    % 将同一运动状态写入两列，供后续融合逻辑使用
    HR(times, 8) = is_motion; % 原 ACC 运动标记
    HR(times, 9) = is_motion; % 原 HF 运动标记 (现已强制同步为 ACC 结果)

    % 计算相关性与时延
    [mh_arr, ma_arr, time_delay_h, time_delay_a] = ...
        ChooseDelay1218(Fs, time_1, ppg, {accx,accy,accz}, {hotf1,hotf2});

    % =====================================================================
    % 路径 A: Pure LMS (HF Reference) - 纯 HF 生态
    % =====================================================================
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

    % HF 路径频谱惩罚
    Freq_HF = Helper_Process_Spectrum(Sig_LMS_HF, Sig_h{best_hf_idx}, Fs, para, times, HR(:,3), ...
                                    true, para.HR_Range_Hz, para.Slew_Limit_BPM, para.Slew_Step_BPM);
    HR(times, 3) = Freq_HF;

    % =====================================================================
    % 路径 B: Pure LMS (ACC Reference) - 保持原样
    % =====================================================================
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

    % ACC 路径频谱惩罚
    Freq_ACC = Helper_Process_Spectrum(Sig_LMS_ACC, Sig_a{best_acc_idx}, Fs, para, times, HR(:,4), ...
                                     true, para.HR_Range_Hz, para.Slew_Limit_BPM, para.Slew_Step_BPM);
    HR(times, 4) = Freq_ACC;

    % =====================================================================
    % 路径 C: Pure FFT (原始信号) [修改核心]
    % =====================================================================
    Sig_FFT = Sig_p - mean(Sig_p);
    Sig_FFT = Sig_FFT .* hamming(length(Sig_FFT));
    
    % [修改] 启用频谱惩罚
    % 为了对比 LMS 的去噪效果，此处将 Pure FFT 也加入频谱惩罚。
    % 1. 参考信号：Sig_a{3} (ACC Z轴)，与 ACC LMS 路径的参考一致。
    % 2. 启用标志：改为 true (原为 false)。
    % 3. 追踪参数：保持 Rest 参数不变 (以免影响 Fusion 逻辑中的静息段回退功能)。
    Freq_FFT = Helper_Process_Spectrum(Sig_FFT, Sig_a{3}, Fs, para, times, HR(:,5), ...
                                     true, para.HR_Range_Rest, para.Slew_Limit_Rest, para.Slew_Step_Rest);
    HR(times, 5) = Freq_FFT;

    % 循环推进
    time_1 = time_1 + Win_Step;
    times = times + 1;
    if time_1 > time_end, stop_flag = 0; end
end

%% 5. 后处理与融合决策
% 全局平滑
for c = 3:5
    HR(:, c) = smoothdata(HR(:, c), 'movmedian', para.Smooth_Win_Len);
end

% 5.1 融合决策 (逻辑更新)
for i = 1:size(HR, 1)
    % --- Fusion HF 决策 ---
    if HR(i, 9) == 1 % 判定为运动 (基于 ACC)
        HR(i, 6) = HR(i, 3); % 使用 LMS(HF) 结果
    else             % 判定为静息
        HR(i, 6) = HR(i, 5); % 回退到 FFT
    end
    
    % --- Fusion ACC 决策 ---
    if HR(i, 8) == 1 % 判定为运动 (基于 ACC)
        HR(i, 7) = HR(i, 4); % 使用 LMS(Acc) 结果
    else             % 判定为静息
        HR(i, 7) = HR(i, 5); % 回退到 FFT
    end
end

% 连接点微调
HR(:, 6) = smoothdata(HR(:, 6), 'movmedian', 3);
HR(:, 7) = smoothdata(HR(:, 7), 'movmedian', 3);

%% 6. 指标计算
T_Pred = HR(:,1) + para.Time_Bias;
HR_Ref_Interp = interp1(HR(:,1), HR(:,2), T_Pred, 'linear', 'extrap');

mask_motion = (HR(:, 8) == 1);
mask_rest   = (HR(:, 8) == 0);

col_indices = [3, 4, 5, 6, 7];
err_stats   = zeros(5, 3); % [All, Rest, Motion] AAE

for k = 1:length(col_indices)
    col = col_indices(k);
    abs_err = abs(HR(:, col) - HR_Ref_Interp) * 60;
    
    val_total  = mean(abs_err);
    val_rest   = mean(abs_err(mask_rest));
    val_motion = mean(abs_err(mask_motion));
    
    err_stats(k, :) = [val_total, val_rest, val_motion];
end

% 提取优化目标：Fusion(HF) 的 Total AAE 是1；Motion AAE 是3
Result.Err_Fus_HF = err_stats(4, 1); 

% 打包输出结果
Result.HR = HR;
Result.err_stats = err_stats;
Result.T_Pred = T_Pred;
Result.Motion_Threshold = [Motion_Threshold_ACC, Motion_Threshold_ACC]; 
Result.HR_Ref_Interp = HR_Ref_Interp;

end

%% 辅助函数 (保持不变)
function est_freq = Helper_Process_Spectrum(sig_in, sig_penalty_ref, Fs, para, times, history_arr, enable_penalty, range_hz, limit_bpm, step_bpm)
    % 1. 频谱惩罚
    [S_rls, S_rls_amp] = FFT_Peaks(sig_in, Fs, 0.3);
    
    if para.Spec_Penalty_Enable && enable_penalty
        % 计算参考信号(ACC或HF)的频谱
        [S_ref, S_ref_amp] = FFT_Peaks(sig_penalty_ref, Fs, 0.3);
        if ~isempty(S_ref)
            [~, midx] = max(S_ref_amp); 
            Motion_Freq = S_ref(midx);
            
            % 在 PPG 频谱中抑制该频率及其倍频
            mask = (abs(S_rls - Motion_Freq) < para.Spec_Penalty_Width) | ...
                   (abs(S_rls - 2*Motion_Freq) < para.Spec_Penalty_Width);
            S_rls_amp(mask) = S_rls_amp(mask) * para.Spec_Penalty_Weight;
        end
    end
    
    % 2. 寻峰
    Fre = Find_maxpeak(S_rls, S_rls, S_rls_amp);
    if isempty(Fre), Fre = 0; end
    curr_raw = Fre(1);
    
    % 3. 历史追踪
    if times == 1
        est_freq = curr_raw;
    else
        prev_hr = history_arr(times-1);
        [calc_hr, ~] = Find_nearBiggest(Fre, prev_hr, range_hz, -range_hz);
        
        diff_hr = calc_hr - prev_hr;
        limit   = limit_bpm / 60;
        step    = step_bpm / 60; 
        
        if diff_hr > limit,      est_freq = prev_hr + step;
        elseif diff_hr < -limit, est_freq = prev_hr - step;
        else,                    est_freq = calc_hr;
        end
    end
end