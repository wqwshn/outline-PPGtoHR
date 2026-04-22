function [freqs, amps] = compute_spectrum(signal, Fs)
% compute_spectrum 计算信号的完整 FFT 幅度谱
% 输入:
%   signal - 输入信号 (行或列向量)
%   Fs     - 采样率 (Hz)
% 输出:
%   freqs  - 频率向量 (Hz), 长度 Len/2
%   amps   - 单边幅度谱, 长度 Len/2

    if size(signal,1) > size(signal,2)
        signal = signal';
    end

    a = length(signal);
    Len = 2^13;

    FFTData = fft(signal, Len);
    FFTAmp0 = abs(FFTData) / a;
    amps = FFTAmp0(1:Len/2);
    amps(2:end) = 2 * amps(2:end);
    freqs = Fs * ((0:(Len/2)-1)) / Len;
end