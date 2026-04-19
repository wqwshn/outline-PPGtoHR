function [mh_arr,ma_arr,time_delay_h,time_delay_a] = ChooseDelay1218(Fs,time_1,ppg,acc_signals,hf_signals)
% 计算 PPG 与参考信号(ACC/HF)之间的时延和各通道最大相关系数
% 输入:
%   acc_signals - ACC信号cell数组, 如 {accx, accy, accz}
%   hf_signals  - HF信号cell数组,  如 {hotf1, hotf2}
% 输出:
%   mh_arr - HF各通道最大相关系数(向量)
%   ma_arr - ACC各通道最大相关系数(向量)
%   time_delay_h - HF最优时延(采样点偏移)
%   time_delay_a - ACC最优时延(采样点偏移)

    num_acc = length(acc_signals);
    num_hf  = length(hf_signals);

    DelayHow_a = zeros(11, num_acc + 1);  % Col1=lag, Col2..N=各通道相关系数
    DelayHow_h = zeros(11, num_hf + 1);

    p1 = floor(time_1*Fs);
    p2 = p1 + 8*Fs - 1;
    if p2 > length(ppg), p2 = length(ppg); end
    ppg_seg = ppg(p1:p2);

    for ii = -5:5
        row = ii + 6;  % -5->1, 0->6, 5->11
        DelayHow_a(row, 1) = ii;
        DelayHow_h(row, 1) = ii;

        p1 = floor((time_1 + ii/Fs)*Fs);
        p2 = p1 + 8*Fs - 1;

        if p1 < 1 || p2 > length(ppg)
            DelayHow_a(row, 2:end) = 0;
            DelayHow_h(row, 2:end) = 0;
        else
            for ch = 1:num_acc
                DelayHow_a(row, ch+1) = corr(ppg_seg, acc_signals{ch}(p1:p2));
            end
            for ch = 1:num_hf
                DelayHow_h(row, ch+1) = corr(ppg_seg, hf_signals{ch}(p1:p2));
            end
        end
    end

    % 处理 NaN (如通道全零导致 corr 为 NaN)
    DelayHow_h(isnan(DelayHow_h)) = 0;
    DelayHow_a(isnan(DelayHow_a)) = 0;

    % 各通道最大相关系数
    mh_arr = max(abs(DelayHow_h(:, 2:end)), [], 1);  % 1×num_hf 向量
    ma_arr = max(abs(DelayHow_a(:, 2:end)), [], 1);  % 1×num_acc 向量

    % --- HF 最优时延: 取相关系数最大的通道, 再取该通道中最大相关对应的时延 ---
    [~, best_hf_ch] = max(mh_arr);
    target_col = abs(DelayHow_h(:, best_hf_ch + 1));
    [~, max_row] = max(target_col);
    time_delay_h = DelayHow_h(max_row, 1);

    % --- ACC 最优时延 ---
    [~, best_acc_ch] = max(ma_arr);
    target_col_a = abs(DelayHow_a(:, best_acc_ch + 1));
    [~, max_row_a] = max(target_col_a);
    time_delay_a = DelayHow_a(max_row_a, 1);

end
