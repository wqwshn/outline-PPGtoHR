function Fre_Sorted = Find_maxpeak(Freqs, ~, Amps)
% Find_maxpeak 根据幅值大小对频率峰值进行降序排序
% 
% 输入:
%   Freqs : 候选峰值的频率位置 (对应代码中的 S_rls)
%   ~     : 占位符 (对应代码中重复传入的 S_rls，此处忽略)
%   Amps  : 候选峰值的幅值 (对应代码中的 S_rls_amp)
%
% 输出:
%   Fre_Sorted : 按幅值降序排列后的频率数组
%                Fre_Sorted(1) 即为最大峰对应的频率

    % 1. 鲁棒性检查：如果输入为空，直接返回空
    if isempty(Freqs) || isempty(Amps)
        Fre_Sorted = [];
        return;
    end

    % 2. 确保输入为列向量，防止维度不匹配
    Freqs = Freqs(:);
    Amps  = Amps(:);

    % 3. 核心逻辑：对幅值进行降序排序 (Descend)
    % val 没用到，我们只需要排序后的索引 idx
    [~, idx] = sort(Amps, 'descend');

    % 4. 根据排序索引重新排列频率数组
    Fre_Sorted = Freqs(idx);

end