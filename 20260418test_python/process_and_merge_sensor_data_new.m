function process_and_merge_sensor_data_new(sensor_csv_filepath, gt_csv_filepath)
% PROCESS_AND_MERGE_SENSOR_DATA_NEW
% 功能：整合 CSV 读取、重建 100Hz 高精度时间轴、信号清洗与全频道带通滤波。
% 特色：消除变量名修改警告，统计原始数据修正量，支持三路 PPG 及六轴 IMU 处理。
%
% =========================================================================
% 【代码使用示例】
% process_and_merge_sensor_data_new('multi-tiaosheng1.csv', 'multi-tiaosheng1_ref.csv')
% =========================================================================

    %% --- 1. 读取传感器 CSV 文件 (解决变量名警告) ---
    fprintf('================================================\n');
    fprintf('1. 正在读取传感器 CSV 文件: %s ...\n', sensor_csv_filepath);
    
    if ~isfile(sensor_csv_filepath)
        error('错误: 传感器CSV文件不存在，请检查路径: %s', sensor_csv_filepath);
    end

    % 获取导入选项
    opts = detectImportOptions(sensor_csv_filepath);
    
    % 【关键修复】：设置为 'preserve' 以保留原始表头（如包含括号的名称），消除修改警告
    opts.VariableNamingRule = 'preserve'; 
    
    raw_data = readtable(sensor_csv_filepath, opts);
    totalRows = height(raw_data);
    fprintf('  - CSV读取成功，共获取 %d 行数据。\n', totalRows);

    %% --- 2. 提取数据并重建高精度时间轴 ---
    fprintf('\n2. 正在提取数据并重建 100Hz 时间轴...\n');
    newData = table();
    
    % 基于 100Hz 采样率重建时间轴 (t = 0, 0.01, 0.02 ...)
    fs = 100;
    newData.Time_s = ((0:totalRows-1) / fs)';
    
    % 提取电信号 (使用括号内的原始列名进行精确索引)
    newData.Uc1 = raw_data.("Uc1(mV)");
    newData.Uc2 = raw_data.("Uc2(mV)");
    newData.Ut1 = raw_data.("Ut1(mV)");
    newData.Ut2 = raw_data.("Ut2(mV)");

    % 提取 PPG 信号
    newData.PPG_Green = double(raw_data.("PPG_Green"));
    newData.PPG_Red   = double(raw_data.("PPG_Red"));
    newData.PPG_IR    = double(raw_data.("PPG_IR"));

    % 提取运动信号 (加速度与陀螺仪)
    newData.AccX = raw_data.("AccX(g)");
    newData.AccY = raw_data.("AccY(g)");
    newData.AccZ = raw_data.("AccZ(g)");
    newData.GyroX = raw_data.("GyroX(dps)");
    newData.GyroY = raw_data.("GyroY(dps)");
    newData.GyroZ = raw_data.("GyroZ(dps)");

    %% --- 3. 信号清洗、修正统计与带通滤波 ---
    fprintf('\n3. 正在执行异常检测、清洗与带通滤波 (0.5-5Hz)...\n');
    
    % 滤波参数：0.5-5Hz 带通
    filt_cutoff = [0.5 5];
    filt_order = 4;
    nyquist = fs / 2;
    [b, a] = butter(filt_order, filt_cutoff / nyquist, 'bandpass');
    
    % 定义待处理队列
    signals_to_process = {'Uc1', 'Uc2', 'Ut1', 'Ut2', ...
                          'PPG_Green', 'PPG_Red', 'PPG_IR', ...
                          'AccX', 'AccY', 'AccZ', ...
                          'GyroX', 'GyroY', 'GyroZ'};
    
    for k = 1:length(signals_to_process)
        sig_name = signals_to_process{k};
        filt_name = [sig_name, '_Filt'];
        
        raw_sig = newData.(sig_name);
        original_sig = raw_sig; % 备份原始数据用于对比
        
        % A. 填补缺失值 (NaN)
        raw_sig = fillmissing(raw_sig, 'nearest'); 
        
        % B. PPG 负值物理逻辑修正 (仅针对 PPG)
        if contains(sig_name, 'PPG')
            neg_mask = raw_sig < 0;
            if any(neg_mask)
                raw_sig(neg_mask) = NaN;
                raw_sig = fillmissing(raw_sig, 'linear');
                raw_sig = fillmissing(raw_sig, 'nearest');
            end
        end
        
        % C. 异常毛刺清洗 (3倍 MAD 准则)
        window_size = fs; % 1秒滑动窗口
        raw_sig_clean = filloutliers(raw_sig, 'linear', 'movmedian', window_size);
        
        % --- 统计并汇报数据变动量 ---
        changed_mask = (raw_sig_clean ~= original_sig) | (isnan(original_sig) & ~isnan(raw_sig_clean));
        num_changed = sum(changed_mask);
        
        if num_changed > 0
            fprintf('    - [数据统计] 通道 %-10s : 修正了 %4d 个原始数据点。\n', sig_name, num_changed);
        end
        
        % 覆盖清洗后的数据，并生成对应的滤波数据列
        newData.(sig_name) = raw_sig_clean; 
        newData.(filt_name) = filtfilt(b, a, raw_sig_clean);
    end

    %% --- 4. 读取并清洗心率真值文件 ---
    fprintf('\n4. 正在读取并清洗真值文件: %s ...\n', gt_csv_filepath);
    if ~isfile(gt_csv_filepath)
        error('错误: 真值文件不存在: %s', gt_csv_filepath);
    end

    try
        % 跳过表头信息，从第4行开始读取
        opts_gt = detectImportOptions(gt_csv_filepath);
        opts_gt.DataLines = [4 Inf]; 
        opts_gt.VariableNamesLine = 0;     
        T_ext = readtable(gt_csv_filepath, opts_gt);
        
        raw_gt_time = T_ext{:, 2}; 
        raw_gt_val  = T_ext{:, 3};
    catch ME
        error('错误: 真值解析失败，请检查文件格式: %s', ME.message);
    end

    % 转换时间格式为秒
    if isnumeric(raw_gt_time)
        sig_gt_seconds = raw_gt_time; 
    else
        try
            sig_gt_seconds = seconds(duration(string(raw_gt_time))); 
        catch
            sig_gt_seconds = seconds(duration(string(raw_gt_time), 'InputFormat', 'h:mm:ss'));
        end
    end

    % 剔除无效的 NaN 数据
    sig_gt_bpm = double(raw_gt_val);
    valid_mask = ~isnan(sig_gt_seconds) & ~isnan(sig_gt_bpm);
    clean_time = sig_gt_seconds(valid_mask);
    clean_bpm  = sig_gt_bpm(valid_mask);
    
    fprintf('  - 真值清洗完成，提取到 %d 组有效数据。\n', length(clean_time));

    %% --- 5. 导出结果为 MAT 文件 ---
    fprintf('\n5. 正在导出并保存数据...\n');
    
    ref_data = [clean_time, clean_bpm];
    data = newData;

    [folder, name, ~] = fileparts(sensor_csv_filepath);
    final_mat_filename = fullfile(folder, [name, '_processed.mat']);
    
    save(final_mat_filename, 'data', 'ref_data');

    fprintf('================================================\n');
    fprintf('✅ 处理成功！\n- 时间轴：100Hz 重新生成\n- 警告：已消除\n- 结果文件：%s\n', final_mat_filename);
end