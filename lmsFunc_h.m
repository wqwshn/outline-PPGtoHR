function [e,w,ee]=lmsFunc_h(mu,M,K,u,d)
% Normalized LMS
% Call:
% [e,w]=nlms(mu,M,u,d,a);
%
% Input arguments:
% mu = step size, dim 1x1  步长
% M = filter length, dim 1x1 FIR阶数
% u = input signal, dim Nx1  加速度信号
% d = desired signal, dim Nx1   ppg信号
% a = constant, dim 1x1    一个常数
%
% Output arguments:
% e = estimation error, dim Nx1    d(n)-y(n)
% w = final filter coefficients, dim Mx1    最终的FIR系数
%intial value 0
 
u = zscore(u);
d = zscore(d);

% K = 0;
w=zeros(M+K,1); %This is a vertical column

%input signal length
N=length(u);
%make sure that u and d are colon vectors
u=u(:);
d=d(:);
%NLMS
ee=zeros(1,N);
for n=M:N-K %Start at M (Filter Length) and Loop to N (Length of Sample)
    uvec=u(n+K:-1:n-M+1); %Array, start at n, decrement to n-m+1
    e(n)=d(n)-w'*uvec;
    w=w+2*mu*uvec*e(n);
    % y(n) = w'*uvec; %In ALE, this will be the narrowband noise.
end