function [mh1,mh2,mh3,ma1,ma2,ma3,time_delay_h,time_delay_a] = ChooseDelay1218(Fs,time_1,ppg,accx,accy,accz,hotf1,hotf2,hotf3)
% -5:5相当于算了这个时间窗前后移动5个采样点，
% 然后取这十次最大相关度作为这个时间窗的相关度结果，
% 并且同时得到时延这个参数    
    DelayHow_a = zeros(11, 4);  % 预分配内存 (11行: -5到5, 4列: lag, x, y, z)
    DelayHow_h = zeros(11, 4);

    p1 = floor(time_1*Fs);
    p2 = p1 + 8*Fs -1;
    
    % 边界检查：防止索引超出数据长度
    if p2 > length(ppg)
        p2 = length(ppg);
    end
    
    ssss = ppg(p1:p2);

    times = 1;

    for ii = -5:1:5
        times_1 = time_1 + ii/Fs;
        DelayHow_a(times,1) = ii;
        DelayHow_h(times,1) = ii;
        
        p1 = floor(times_1*Fs);
        p2 = p1 + 8*Fs -1;
        
        % 边界保护
        if p1 < 1 || p2 > length(accx)
            % 如果移位后越界，相关系数设为0
            DelayHow_a(times,2:4) = 0;
            DelayHow_h(times,2:4) = 0;
        else
            Signal_a_x = accx(p1:p2);
            Signal_a_y = accy(p1:p2);
            Signal_a_z = accz(p1:p2);
            Signal_h_1 = hotf1(p1:p2);
            Signal_h_2 = hotf2(p1:p2);
            Signal_h_3 = hotf3(p1:p2); 

            % 计算相关系数 (注意：corr 输入列向量)
            DelayHow_a(times,2) = corr(ssss,Signal_a_x);
            DelayHow_a(times,3) = corr(ssss,Signal_a_y);
            DelayHow_a(times,4) = corr(ssss,Signal_a_z);

            DelayHow_h(times,2) = corr(ssss,Signal_h_1);
            DelayHow_h(times,3) = corr(ssss,Signal_h_2);
            DelayHow_h(times,4) = corr(ssss,Signal_h_3);
        end
        
        times = times + 1;
    end
    
    % --- 【修正点1】处理 NaN (因为hotf3是0，corr结果会是NaN) ---
    DelayHow_h(isnan(DelayHow_h)) = 0;
    DelayHow_a(isnan(DelayHow_a)) = 0;
   
    % --- 计算加速度最大相关 ---
    ma1 = max(abs(DelayHow_a(:,2)));
    ma2 = max(abs(DelayHow_a(:,3)));
    ma3 = max(abs(DelayHow_a(:,4)));
    
    % --- 计算热膜最大相关 ---
    mh1 = max(abs(DelayHow_h(:,2)));
    mh2 = max(abs(DelayHow_h(:,3)));
    mh3 = max(abs(DelayHow_h(:,4)));
    
    % --- 【修正点2】热膜延时计算 (强制标量) ---
    mh_mat = sort([mh1,mh2,mh3],'descend');
    % find 可能会返回多个索引，比如 [1, 2]，这里强制取第1个
    tmp_idx = find([mh1,mh2,mh3] == mh_mat(1));
    indexxh = tmp_idx(1); 
    
    % 找到该通道中相关系数绝对值最大的行索引
    target_col = abs(DelayHow_h(:, indexxh+1)); % +1 是因为第1列是lag
    max_val = max(target_col);
    time_delay_h_r = find(target_col == max_val);
    
    if ~isempty(time_delay_h_r)
        time_delay_h = DelayHow_h(time_delay_h_r(1), 1); % 强制取第1个
    else
        time_delay_h = 0; % 默认值，防止空数组报错
    end

    % --- 【修正点3】加速度延时计算 (强制标量) ---
    ma_mat = sort([ma1,ma2,ma3],'descend');
    tmp_idx_a = find([ma1,ma2,ma3] == ma_mat(1));
    indexxa = tmp_idx_a(1);
    
    target_col_a = abs(DelayHow_a(:, indexxa+1));
    max_val_a = max(target_col_a);
    time_delay_a_r = find(target_col_a == max_val_a);
    
    if ~isempty(time_delay_a_r)
        time_delay_a = DelayHow_a(time_delay_a_r(1), 1);
    else
        time_delay_a = 0;
    end
    
end