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
Col_Gyro = [12, 13, 14]; if isfield(para, 'Col_Gyro'), Col_Gyro = para.Col_Gyro; end

% % Spo2的数据格式
% Fs_Origin = 125;      % 原始采样率
% Col_PPG  = 6;           
% Col_HF1  = 2; Col_HF2 = 3;        
% Col_Acc  = [8, 9, 10]; 

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

    % 公共信号集用于运动检测和静息 FFT (取最高 Fs)
    Fs = Fs_common;
    ss0 = sig_sets{find(Fs_list == Fs_common, 1)};
    ppg  = ss0.ppg;  hotf1 = ss0.hf1; hotf2 = ss0.hf2;
    accx = ss0.accx; accy  = ss0.accy; accz = ss0.accz;

    % 加载分类器
    classifier_model_path = para.model_path;
    scaler_data = load(fullfile(classifier_model_path, 'scaler_params.mat'));
    rf_data     = load(fullfile(classifier_model_path, 'rf_model_3class.mat'));

    % IMU 陀螺仪数据 (分类器 + 运动检测共用)
    imu_gyrox = resample(raw_data(:, Col_Gyro(1)), Fs_common, Fs_Origin);
    imu_gyroy = resample(raw_data(:, Col_Gyro(2)), Fs_common, Fs_Origin);
    imu_gyroz = resample(raw_data(:, Col_Gyro(3)), Fs_common, Fs_Origin);

    % 运动检测用的带通滤波陀螺仪
    [b_g, a_g] = butter(4, [0.5 5]/(Fs_common/2), 'bandpass');
    gyrox = filtfilt(b_g, a_g, imu_gyrox);
    gyroy = filtfilt(b_g, a_g, imu_gyroy);
    gyroz = filtfilt(b_g, a_g, imu_gyroz);

else
    %% 原始模式: 单遍预处理
    Fs = para.Fs_Target;
    ppg_ori   = resample(filloutliers(raw_data(:, Col_PPG),'previous','mean'), Fs, Fs_Origin);
    hotf1_ori = resample(raw_data(:, Col_HF1), Fs, Fs_Origin);
    hotf2_ori = resample(raw_data(:, Col_HF2), Fs, Fs_Origin);
    accx_ori  = resample(raw_data(:, Col_Acc(1)), Fs, Fs_Origin);
    accy_ori  = resample(raw_data(:, Col_Acc(2)), Fs, Fs_Origin);
    accz_ori  = resample(raw_data(:, Col_Acc(3)), Fs, Fs_Origin);
    gyrox_ori = resample(raw_data(:, Col_Gyro(1)), Fs, Fs_Origin);
    gyroy_ori = resample(raw_data(:, Col_Gyro(2)), Fs, Fs_Origin);
    gyroz_ori = resample(raw_data(:, Col_Gyro(3)), Fs, Fs_Origin);

    BP_Low = 0.5; BP_High = 5; BP_Order = 4;
    [b_but, a_but] = butter(BP_Order, [BP_Low BP_High]/(Fs/2), 'bandpass');

    ppg   = filtfilt(b_but, a_but, ppg_ori);
    hotf1 = filtfilt(b_but, a_but, hotf1_ori);
    hotf2 = filtfilt(b_but, a_but, hotf2_ori);
    accx  = filtfilt(b_but, a_but, accx_ori);
    accy  = filtfilt(b_but, a_but, accy_ori);
    accz  = filtfilt(b_but, a_but, accz_ori);
    gyrox = filtfilt(b_but, a_but, gyrox_ori);
    gyroy = filtfilt(b_but, a_but, gyroy_ori);
    gyroz = filtfilt(b_but, a_but, gyroz_ori);
end

%% 2. 运动阈值校准 (MIMU 六轴: ACC + Gyro 联合判定)
calib_len = min(round(para.Calib_Time * Fs), length(ppg));

% 2.1 幅值计算
acc_mag = sqrt(accx.^2 + accy.^2 + accz.^2);
gyro_mag = sqrt(gyrox.^2 + gyroy.^2 + gyroz.^2);

% 2.2 基线校准: 各自归一化后联合判定
acc_baseline_std = std(acc_mag(1:calib_len));
gyro_baseline_std = std(gyro_mag(1:calib_len));
Motion_Threshold_ACC = para.Motion_Th_Scale * acc_baseline_std;
Motion_Threshold_Gyro = para.Motion_Th_Scale * gyro_baseline_std;

%% 3. 主循环初始化
Win_Len = 8; Win_Step = 1;
time_end = length(ppg)/Fs - para.Time_Buffer;

stop_flag = 1;
times = 1;
time_1 = para.Time_Start;
last_motion_flag = false;

if isfield(para, 'expert_mode') && para.expert_mode
    expert_names_local = fieldnames(para.expert_params);
    K_local = length(expert_names_local);
    Fs_common_local = Fs_common;
    if ~isfield(para, 'classifier_mode'), para.classifier_mode = 'window'; end

    % 段级模式: 预计算全部分类器权重
    if strcmp(para.classifier_mode, 'segment')
        all_weights = precompute_segment_weights(...
            accx, accy, accz, imu_gyrox, imu_gyroy, imu_gyroz, ...
            Fs_common_local, Win_Len, Win_Step, time_end, ...
            scaler_data, rf_data, para);
    else
        all_weights = [];
    end

    HR = zeros(1, 12); % Col10-12: 分类器概率
else
    Num_Cascade_HF = 2; Num_Cascade_Acc = 3; LMS_Mu_Base = 0.01;
    HR = zeros(1, 9);
end

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
    Sig_gyro_mag = gyro_mag(idx_s:idx_e);
    
    HR(times, 1) = time_1; 
    HR(times, 2) = Find_realHR('dummy', time_1, HR_Ref_Data); 
    
    % =====================================================================
    % 4.1 运动状态判断 (MIMU 六轴: ACC 或 Gyro 任一超阈值即判定运动)
    % =====================================================================
    is_motion = (std(Sig_acc_mag) > Motion_Threshold_ACC) || ...
                (std(Sig_gyro_mag) > Motion_Threshold_Gyro);
    
    % 将同一运动状态写入两列，供后续融合逻辑使用
    HR(times, 8) = is_motion; % 原 ACC 运动标记
    HR(times, 9) = is_motion; % 原 HF 运动标记 (现已强制同步为 ACC 结果)

    % =====================================================================
    % 路径 C: Pure FFT (始终运行)
    % =====================================================================
    Sig_FFT = Sig_p - mean(Sig_p);
    Sig_FFT = Sig_FFT .* hamming(length(Sig_FFT));
    Freq_FFT = Helper_Process_Spectrum(Sig_FFT, Sig_a{3}, Fs, para, times, HR(:,5), ...
                                     true, para.HR_Range_Rest, para.Slew_Limit_Rest, para.Slew_Step_Rest);
    HR(times, 5) = Freq_FFT;

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
            %% === 专家模式: K 路 LMS -> 频谱融合 ===
            spectra_hf = zeros(2^12, K_local);
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

                [mh_k, ma_k, td_h_k, td_a_k] = ChooseDelay1218(Fs_k, time_1, ss.ppg, Sig_a_k, Sig_h_k);

                % LMS-HF (阶数由延时的绝对值决定)
                Sig_e_hf = Sig_p_k;
                ord_h = max(floor(abs(td_h_k)), 1);
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

                % LMS-ACC (阶数由延时的绝对值决定, 系数与 HF 一致为 1.0)
                Sig_e_acc = Sig_p_k;
                ord_a = max(floor(abs(td_a_k)), 1);
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

                % 重采样到公共 Fs
                if Fs_k ~= Fs_common_local
                    Sig_e_hf = resample(Sig_e_hf, Fs_common_local, Fs_k);
                    Sig_e_acc = resample(Sig_e_acc, Fs_common_local, Fs_k);
                end

                % 计算频谱
                [freqs_common, amps_hf] = compute_spectrum(Sig_e_hf, Fs_common_local);
                [~, amps_acc] = compute_spectrum(Sig_e_acc, Fs_common_local);
                spectra_hf(1:length(amps_hf), k) = amps_hf;
                spectra_acc(1:length(amps_acc), k) = amps_acc;

                best_hf_ref_k{k} = resample(Sig_h_k{best_hf_idx_k}, Fs_common_local, Fs_k);
                best_acc_ref_k{k} = resample(Sig_a_k{best_acc_idx_k}, Fs_common_local, Fs_k);
            end

            % 分类器权重
            weights = Helper_ClassifierWeights(times, time_1, Win_Len, Fs_common_local, ...
                accx, accy, accz, imu_gyrox, imu_gyroy, imu_gyroz, ...
                all_weights, para, scaler_data, rf_data);
            HR(times, 10:12) = weights(:)';

            % 频谱融合
            [~, S_fused_hf] = weighted_spectrum_fusion(spectra_hf, weights);
            [~, S_fused_acc] = weighted_spectrum_fusion(spectra_acc, weights);

            % 加权惩罚参考
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

            % 路径 A: LMS-HF (阶数由延时的绝对值决定)
            Sig_LMS_HF = Sig_p;
            ord_h = max(floor(abs(time_delay_h)), 1);
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

            % 路径 B: LMS-ACC (阶数由延时的绝对值决定, 系数与 HF 一致为 1.0)
            Sig_LMS_ACC = Sig_p;
            ord_a = max(floor(abs(time_delay_a)), 1);
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
        % 静息段: 跳过 LMS, 直接复制 FFT 结果
        HR(times, 3) = Freq_FFT;
        HR(times, 4) = Freq_FFT;
    end

    % 状态记录与循环推进
    last_motion_flag = is_motion;
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
Result.Motion_Threshold = [Motion_Threshold_ACC, Motion_Threshold_Gyro]; 
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

function weights = Helper_ClassifierWeights(times, time_1, Win_Len, Fs, ...
    accx, accy, accz, gyrox, gyroy, gyroz, ...
    all_weights, para, scaler_data, rf_data)
% Helper_ClassifierWeights 计算当前窗口的分类器权重

    if strcmp(para.classifier_mode, 'segment') && ~isempty(all_weights)
        % 段级模式
        t_idx = find(abs(all_weights(:,1) - time_1) < 0.01, 1);
        if ~isempty(t_idx)
            weights = all_weights(t_idx, 2:end)';
        else
            n_cls = rf_data.n_classes;
            weights = ones(n_cls, 1) / n_cls;
        end
    else
        % 窗级模式: 实时推理
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

        % 按专家顺序排列权重
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
        total = sum(weights);
        if total > 0, weights = weights / total;
        else, weights = ones(length(expert_names_local), 1) / length(expert_names_local);
        end
    end
end

function proba = predict_exercise_proba_local(features, scaler_data, rf_data)
% predict_exercise_proba_local 内联 RF 推理
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
% precompute_segment_weights 段级模式: 预计算每个窗口概率
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