function [ratio1] = PpgPeace(Signal,Fs)
%计算FFT 找0~1Hz面积的大小与1~3Hz面积比例

Signal = zscore(Signal);
a = max(size(Signal));
Len = 2^10;  %补零后的长度
if size(Signal,1) > size(Signal,2)
    Signal = Signal';
end
% Signal_long = zeros(Len,1);
% Signal_long(1:a) = Signal;
% %加窗  看起来泄露现象好一点 不会有明显的旁瓣
% w = hann(length(Signal_long)); %汉宁窗主瓣够窄，比较好。
% Signal_long = Signal_long.*w;

FFTData = fft(Signal,Len);
FFTAmp0 = abs(FFTData)/a;                 %频谱关于Len/2对称
FFTAmp1 = FFTAmp0(1:Len/2); 
FFTAmp1(2:end) = 2*FFTAmp1(2:end);
Frequence1 = Fs*((0:(Len/2)-1))/Len;
%figure
% plot(Frequence1,FFTAmp1,'LineWidth',2)
% xlim([0,3.5]);
% set(gca,'FontSize',18)
% title('频谱','FontSize',18)
% xlabel('f(Hz)','FontSize',18)
% ylabel('FFTAmplitude','FontSize',18)
%   
% p=find(FFTAmp1==max(FFTAmp1));
% text(Frequence1(p),FFTAmp1(p),num2str(Frequence1(p)),'color','r','FontSize',18);

int1 = 1; 
int2 = 1;
int3 = 3;
% int3 = 1.5;
% int4 = 2;
% int5 = 2.5;
% int6 = 3;

Sq01 = sum(FFTAmp1(1:floor(int1*Len/Fs)).^2);   %1Hz以内的点数量
Sq12 = sum(FFTAmp1(floor(int2*Len/Fs)+1:floor(int3*Len/Fs)).^2);
% Sq23 = sum(FFTAmp1(floor(int2*Len/Fs)+1:floor(int3*Len/Fs)));
% Sq34 = sum(FFTAmp1(floor(int3*Len/Fs)+1:floor(int4*Len/Fs)));
% Sq45 = sum(FFTAmp1(floor(int4*Len/Fs)+1:floor(int5*Len/Fs)));
% Sq56 = sum(FFTAmp1(floor(int5*Len/Fs)+1:floor(int6*Len/Fs)));


ratio1 = Sq01/Sq12;
% ratio2 = Sq12;
% ratio3 = Sq23;
% ratio4 = Sq34;
% ratio5 = Sq45;
% ratio6 = Sq56;


% ratio = Sq01/Sq13;

end

