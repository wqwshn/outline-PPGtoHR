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
