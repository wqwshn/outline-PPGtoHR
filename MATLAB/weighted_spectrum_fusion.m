function [freqs, amps_fused] = weighted_spectrum_fusion(spectra_list, weights)
% weighted_spectrum_fusion 对多路频谱进行加权融合
% 输入:
%   spectra_list - K 个幅度谱矩阵, 每个 size (Len/2, 1), 由 compute_spectrum 生成
%                  如果为矩阵则每列代表一个 expert 的频谱
%   weights      - 权重向量 (K, 1), 总和为 1
% 输出:
%   freqs       - 频率向量 (从第一个 spectrum 的隐含 Fs/Len 推导, 此处仅回传)
%   amps_fused  - 融合后的幅度谱 (Len/2, 1)

    if iscell(spectra_list)
        K = length(spectra_list);
        amps_fused = zeros(size(spectra_list{1}));
        for k = 1:K
            amps_fused = amps_fused + weights(k) * spectra_list{k};
        end
    else
        % 矩阵形式: 每列一个 expert
        amps_fused = spectra_list * weights(:);
    end
end