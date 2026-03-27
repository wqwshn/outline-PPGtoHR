    %% 自动参数寻优脚本 - 贝叶斯优化版 (AutoOptimize_Bayes_Search)
% 功能：
% 1. 使用贝叶斯优化 (Bayesian Optimization) 寻找最优参数。
% 2. [优化目标变更] 目标函数现在聚焦于 **Motion AAE** (运动段误差)。
%    - 旨在提升运动状态下的解算精度，忽略静息段误差。
% 3. 增强功能：并行计算 (Parallel Pool) + 多轮重启机制 (Restart Strategy)。
% 4. 流程：
%    - Round 1: 寻找 Fusion(HF) 的最小 Motion AAE。
%    - Round 2: 寻找 Fusion(ACC) 的最小 Motion AAE。
%
% 依赖：HeartRateSolver_cas_chengfa (返回结果已针对 Motion AAE 调整)

clc; clear; close all;

%% 1. 启动并行计算池 (Parallel Pool)
% 检查当前环境是否已有并行池，若无则自动启动
poolobj = gcp('nocreate'); 
if isempty(poolobj)
    fprintf('正在启动并行计算池 (可能需要几秒钟)...\n');
    parpool; % 使用默认核心数启动
else
    fprintf('检测到已存在并行池: %d 个 Workers\n', poolobj.NumWorkers);
end

%% 2. 基础配置
% 数据文件路径 (请确认路径正确)
para_base.FileName = 'data\20260119dualtiaosheng1_processed.mat';
para_base.Time_Start = 1;
para_base.Time_Buffer = 10;
para_base.Calib_Time = 60;
para_base.Motion_Th_Scale = 3;
para_base.Spec_Penalty_Enable = 1;
para_base.Spec_Penalty_Weight = 0.2;

% --- 贝叶斯优化核心设置 ---
% 单次优化的最大迭代次数
Max_Iterations = 50; 
% 初始随机种子点数 (先随机跑10次打基础，防止模型跑偏)
Num_Seed_Points = 10; 
% 重启次数 (每个方案独立跑几次，取最优)
Num_Repeats = 3;

%% 3. 定义搜索空间 (离散列表)
% 注意：此处通过索引映射方式处理非连续数值
SearchSpace.Fs_Target = [32, 125]; 
SearchSpace.Max_Order = [12, 16, 20];
SearchSpace.Spec_Penalty_Width = [0.1, 0.2, 0.3, 0.4];

SearchSpace.HR_Range_Hz = [15, 20, 25, 30, 35, 40] / 60;
SearchSpace.Slew_Limit_BPM = 8:15;        
SearchSpace.Slew_Step_BPM = [5, 7, 9];    

SearchSpace.HR_Range_Rest = [20, 25, 30, 35, 40, 50] / 60;
SearchSpace.Slew_Limit_Rest = 5:8;        
SearchSpace.Slew_Step_Rest = 3:5;         

SearchSpace.Smooth_Win_Len = [5, 7, 9];
SearchSpace.Time_Bias = [4, 5, 6]; 



% % AMPD方法
% SearchSpace.Fs_Target = [125, 125]; 
% % --- 2. 核心前处理与滤波 ---
% SearchSpace.Max_Order = [12, 16, 20, 24]; % 可以适当拉高上限，LMS 压力更大了
% 
% % --- 3. 新增：时域陷波器参数 (替代了原来的 Spec_Penalty_Width) ---
% SearchSpace.Notch_Q_Factor = [5, 10, 20]; 
% SearchSpace.Harmonic_Penalty_Enable = [0, 1]; % 0关闭，1开启二次谐波陷波
% 
% % --- 4. 推荐新增：AMPD 时间窗 ---
% SearchSpace.AMPD_Win_Len = [6, 8, 10]; % 需在主循环里接入该参数
% 
% % --- 5. 保持不变：追踪与平滑参数 ---
% SearchSpace.HR_Range_Hz = [15, 20, 25, 30, 35, 40] / 60;
% SearchSpace.Slew_Limit_BPM = 8:15;        
% SearchSpace.Slew_Step_BPM = [5, 7, 9];    
% 
% SearchSpace.HR_Range_Rest = [20, 25, 30, 35, 40, 50] / 60;
% SearchSpace.Slew_Limit_Rest = 5:8;        
% SearchSpace.Slew_Step_Rest = 3:5;         
% 
% SearchSpace.Smooth_Win_Len = [5, 7, 9];
% SearchSpace.Time_Bias = [4, 5, 6];

%% 4. 构建优化变量 (Index Mapping)
% 将每个参数列表映射为 [1, N] 的整数变量，供贝叶斯优化器使用
param_names = fieldnames(SearchSpace);
opt_vars = [];

fprintf('正在初始化贝叶斯优化变量空间...\n');
for i = 1:length(param_names)
    p_name = param_names{i};
    p_list = SearchSpace.(p_name);
    var_name = ['Idx_', p_name];
    % 创建 Integer 类型的优化变量，范围 [1, length]
    opt_vars = [opt_vars, optimizableVariable(var_name, [1, length(p_list)], 'Type', 'integer')];
end

%% 5. 第一轮寻优: 针对 Fusion(HF) (3次独立运行取最优)
fprintf('\n======================================================\n');
fprintf('ROUND 1: 寻找 Fusion(HF) 的最优参数 (目标: Motion AAE)\n');
fprintf('======================================================\n');

% 定义目标函数
ObjFcn_HF = @(T) Wrapper_CostFunction(T, SearchSpace, para_base, 'HF');

% 初始化全局最优记录
Global_Min_HF = Inf;
Best_Para_HF = [];

for run_idx = 1:Num_Repeats
    fprintf('\n--- Fusion(HF) Run %d / %d ---\n', run_idx, Num_Repeats);
    
    % 执行贝叶斯优化
    results_hf = bayesopt(ObjFcn_HF, opt_vars, ...
        'MaxObjectiveEvaluations', Max_Iterations, ...
        'NumSeedPoints', Num_Seed_Points, ...            % [策略] 增加初始点
        'AcquisitionFunctionName', 'expected-improvement-plus', ... % [策略] 自动跳出局部最优
        'IsObjectiveDeterministic', true, ...
        'UseParallel', true, ...                         % [加速] 开启并行
        'PlotFcn', {@plotMinObjective}, ...              % 绘图
        'Verbose', 0);                                   % 减少命令行刷屏
    
    % 检查本次运行结果是否打破了历史记录
    current_min = results_hf.MinObjective;
    if current_min < Global_Min_HF
        fprintf('  >>> 发现更好的解! Motion Error从 %.4f 降低到 %.4f\n', Global_Min_HF, current_min);
        Global_Min_HF = current_min;
        Best_Para_HF = Extract_Best_Params(results_hf, SearchSpace, para_base);
    else
        fprintf('  >>> 本次结果 (%.4f) 未超过历史最佳 (%.4f)\n', current_min, Global_Min_HF);
    end
end

fprintf('\n>> Round 1 (HF) 最终最低 Motion 误差: %.4f\n', Global_Min_HF);

%% 6. 第二轮寻优: 针对 Fusion(ACC) (3次独立运行取最优)
fprintf('\n======================================================\n');
fprintf('ROUND 2: 寻找 Fusion(ACC) 的最优参数 (目标: Motion AAE)\n');
fprintf('======================================================\n');

% 定义目标函数
ObjFcn_ACC = @(T) Wrapper_CostFunction(T, SearchSpace, para_base, 'ACC');

% 初始化全局最优记录
Global_Min_ACC = Inf;
Best_Para_ACC = [];

for run_idx = 1:Num_Repeats
    fprintf('\n--- Fusion(ACC) Run %d / %d ---\n', run_idx, Num_Repeats);
    
    % 执行贝叶斯优化
    results_acc = bayesopt(ObjFcn_ACC, opt_vars, ...
        'MaxObjectiveEvaluations', Max_Iterations, ...
        'NumSeedPoints', Num_Seed_Points, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'IsObjectiveDeterministic', true, ...
        'UseParallel', true, ...
        'PlotFcn', {@plotMinObjective}, ...
        'Verbose', 0);
    
    % 检查本次运行结果
    current_min = results_acc.MinObjective;
    if current_min < Global_Min_ACC
        fprintf('  >>> 发现更好的解! Motion Error从 %.4f 降低到 %.4f\n', Global_Min_ACC, current_min);
        Global_Min_ACC = current_min;
        Best_Para_ACC = Extract_Best_Params(results_acc, SearchSpace, para_base);
    else
        fprintf('  >>> 本次结果 (%.4f) 未超过历史最佳 (%.4f)\n', current_min, Global_Min_ACC);
    end
end

fprintf('\n>> Round 2 (ACC) 最终最低 Motion 误差: %.4f\n', Global_Min_ACC);

%% 7. 保存结果
% 为了兼容 Viewer，将变量命名为 Min_Err_HF/ACC
Min_Err_HF = Global_Min_HF;
Min_Err_ACC = Global_Min_ACC;

% [修改] 使用数据文件名作为后缀
% 从文件路径中提取文件名（不含路径和扩展名）
[~, data_filename, ~] = fileparts(para_base.FileName);
SaveFileName = sprintf('Best_Params_%s.mat', data_filename);
% 保存数据文件完整路径，供Viewer使用
Data_FileName = para_base.FileName;

save(SaveFileName, 'Best_Para_HF', 'Best_Para_ACC', 'Min_Err_HF', 'Min_Err_ACC', 'SearchSpace', 'Data_FileName');

fprintf('\n=== 全部寻优结束 ===\n');
fprintf('结果已保存至: %s\n', SaveFileName);
fprintf('数据文件: %s\n', Data_FileName);
fprintf('请运行 AutoOptimize_Result_Viewer.m 查看详细图表。\n');
% 不自动关闭并行池，以免下次运行又要等待启动
% delete(gcp('nocreate')); 


%% ========================================================================
%  辅助函数
%  ========================================================================

function Error_Val = Wrapper_CostFunction(Idx_Table, SearchSpace, para_base, Target_Mode)
    % 1. 解码参数 (Index -> Value)
    current_para = para_base;
    param_names = fieldnames(SearchSpace);
    
    for i = 1:length(param_names)
        p_name = param_names{i};
        var_name = ['Idx_', p_name];
        idx = Idx_Table.(var_name); 
        current_para.(p_name) = SearchSpace.(p_name)(idx);
    end
    
    % 2. 运行核心算法
    try
        Res = HeartRateSolver_cas_chengfa(current_para);
        
        % 3. 根据目标模式返回对应的 Motion 误差
        if strcmp(Target_Mode, 'HF')
            % [修改确认] Solver 内部已将 Err_Fus_HF 赋值为 Motion AAE
            Error_Val = Res.Err_Fus_HF;      
        elseif strcmp(Target_Mode, 'ACC')
            % [修改核心] 读取第5行(FusACC) 第3列(Motion AAE)
            Error_Val = Res.err_stats(5, 1); 
        else
            error('未知的优化目标模式');
        end
        
    catch
        % 惩罚项: 运行失败给予极大误差，防止算法选择此区域
        Error_Val = 999; 
    end
end

function Best_Para = Extract_Best_Params(bayes_results, SearchSpace, para_base)
    % 从 bayesopt 结果对象中提取最优参数并转换为真实值
    
    % [修正] 使用 XAtMinObjective 获取最优参数的 Table
    best_idx_table = bayes_results.XAtMinObjective;
    
    Best_Para = para_base;
    param_names = fieldnames(SearchSpace);
    
    for i = 1:length(param_names)
        p_name = param_names{i};
        var_name = ['Idx_', p_name];
        
        % [鲁棒性处理] 
        % 如果优化器找到了多个参数组合得到完全相同的最小误差，XAtMinObjective 会有多行。
        % 我们默认取第一行即可。
        if height(best_idx_table) > 1
            % 如果是多行 Table，使用 {} 或 () 索引提取第一行对应列的值
            best_idx = best_idx_table{1, var_name};
        else
            % 如果是单行 Table，直接使用点索引
            best_idx = best_idx_table.(var_name);
        end
        
        % 映射回真实参数值
        Best_Para.(p_name) = SearchSpace.(p_name)(best_idx);
    end
end