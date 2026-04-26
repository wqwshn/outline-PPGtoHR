% function [Fre,Famp] = FFT_Peaks(Signal,Fs,percent)  
% %输出大于百分之多少的maxPeak幅值的谱峰位置和幅值,峰必须在1~3Hz
% %Signal 要做fft的信号
% %Fs 采样率
% %percent 需要提取的大于最大peak幅值百分之多少的peaks
% 
% a = max(size(Signal)); %信号长度
% Len = 2^13;            %补零后的长度
% if size(Signal,1) > size(Signal,2)
%     Signal = Signal';
% end
% 
% %加窗  看起来泄露现象好一点 不会有明显的旁瓣
% % w = hann(length(Signal))'; %汉宁窗主瓣够窄，比较好。
% % Signal = Signal.*w;
% 
% FFTData = fft(Signal, Len);
% FFTAmp0 = abs(FFTData)/a;                 %频谱关于Len/2对称
% FFTAmp1 = FFTAmp0(1:Len/2); 
% FFTAmp1(2:end) = 2*FFTAmp1(2:end);
% Frequence1 = Fs*((0:(Len/2)-1))/Len;
% 
% free_low = 1*Len/Fs+1;
% free_high = 4*Len/Fs;  %峰必须在1~3Hz之间
% [pks,locs] = findpeaks(FFTAmp1);  %找所有的峰
% pks_2 = pks((locs < free_high) & (locs>free_low));
% locss = locs((pks>max(pks_2)*percent));
% locss = locss((locss < free_high) & (locss>free_low));  
% Fre = Frequence1(locss);
% Famp = FFTAmp1(locss);
% 
% rmss = rms(FFTAmp1);
% ppeak = max(FFTAmp1);
% cff = ppeak/rmss;
% end

function [Fre, Famp] = FFT_Peaks(Signal, Fs, percent)
    % 确保输入是行向量
    if size(Signal,1) > size(Signal,2)
        Signal = Signal';
    end

    a = length(Signal); 
    Len = 2^13;            
    
    FFTData = fft(Signal, Len);
    FFTAmp0 = abs(FFTData)/a;                 
    FFTAmp1 = FFTAmp0(1:Len/2); 
    FFTAmp1(2:end) = 2*FFTAmp1(2:end);
    Frequence1 = Fs*((0:(Len/2)-1))/Len;

    % 定义有效频率范围 (通常为 1~4 Hz)
    free_low = 0.7 * Len/Fs + 1;
    free_high = 4 * Len/Fs; 
    
    [pks, locs] = findpeaks(FFTAmp1); 
    
    % --- 修复开始 ---
    % 筛选出在有效频率范围内的峰
    valid_indices = (locs < free_high) & (locs > free_low);
    pks_2 = pks(valid_indices);
    
    % 如果有效范围内没有峰 (例如静止状态或信号太弱)
    if isempty(pks_2)
        Fre = []; 
        Famp = [];
        return; % 直接返回空值，防止下方 max() 报错
    end
    % --- 修复结束 ---

    % 只有在 pks_2 非空时，才计算阈值
    threshold = max(pks_2) * percent;
    
    % 再次筛选：必须大于阈值，且在有效范围内
    locss = locs((pks > threshold) & valid_indices);
    
    Fre = Frequence1(locss);
    Famp = FFTAmp1(locss);
end
