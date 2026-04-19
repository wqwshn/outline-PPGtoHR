function HR_real = Find_realHR(experi_name, time_current, HR_Ref_Data)
% Find_realHR - 适配版
% 输入:
%   experi_name: (未使用，仅保留接口兼容性)
%   time_current: 当前时间窗的起始时间 (秒)
%   HR_Ref_Data: 心率真值矩阵，两列 [Time(s), BPM]
% 输出:
%   HR_real: 对应时刻的心率值 (单位: Hz, 即 BPM/60)

    % 1. 提取时间和心率列
    Ref_Time = HR_Ref_Data(:, 1);
    Ref_BPM  = HR_Ref_Data(:, 2);
    
    % 2. 确定查询时间点
    % 注意：主程序取的是8秒窗 (time_current 到 time_current+8)
    % 通常取窗口中心作为心率代表点，或者是窗口结束点
    % 原程序逻辑似乎有些许偏移 (Delay)，这里我们取窗口中心时间
    Window_Len = 8;
    Query_Time = time_current + Window_Len / 2;
    
    % 3. 线性插值查找心率
    % 使用 'extrap' 防止时间点超出范围报错，但要注意数据是否覆盖
    try
        BPM_Found = interp1(Ref_Time, Ref_BPM, Query_Time, 'linear', 'extrap');
    catch
        % 如果插值失败，返回 NaN 或 0
        BPM_Found = 0;
    end
    
    % 4. 转换单位
    % 原程序画图时乘以了60，说明 HR 变量里存的是 Hz
    HR_real = BPM_Found / 60;

end