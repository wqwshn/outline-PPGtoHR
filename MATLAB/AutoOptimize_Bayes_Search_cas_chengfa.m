%% 自动参数寻优脚本 - 贝叶斯优化版 (AutoOptimize_Bayes_Search)
% 功能：
% 1. 使用贝叶斯优化 (Bayesian Optimization) 寻找最优参数。
% 2. 优化目标: **全局 AAE** (Total AAE, err_stats(:,1))。
%    - 兼顾运动段和静息段的整体精度。
% 3. 增强功能：并行计算 (Parallel Pool) + 多轮重启机制 (Restart Strategy)。
% 4. 流程：
%    - Round 1: 寻找 Fusion(HF) 的最小全局 AAE。
%    - Round 2: 寻找 Fusion(ACC) 的最小全局 AAE。
%
% 依赖：HeartRateSolver_cas_chengfa (返回结果已针对 Motion AAE 调整)

clc; clear; close all;

%% 1. 启动并行计算池 (Parallel Pool)
% 检查当前环境是否已有并行池，若无则自动启动
poolobj = gcp('nocreate'); 
if isempty(poolobj)
    fprintf('正在启动并行计算池 (可能需要几秒钟)...\n');
    parpool; % 使用默认核心数启动并行池，加速后续的贝叶斯优化评估
else
    fprintf('检测到已存在并行池: %d 个 Workers\n', poolobj.NumWorkers);
end

%% 2. 基础配置
% 数据文件路径 (请确认路径正确)
para_base.FileName = 'dataformatlab\multi_bobi2_processed.mat'; 
para_base.Time_Start = 1;
para_base.Time_Buffer = 10;
para_base.Calib_Time = 30;
para_base.Fs_Target = 25;
para_base.Motion_Th_Scale = 2.5;
para_base.Spec_Penalty_Enable = 1;
para_base.Spec_Penalty_Weight = 0.2;

% 专家模式配置: 直接从各运动场景的贝叶斯优化结果中加载前级参数
para_base.expert_mode = true;
% 分类器模式: 需与 QuickTest 中使用的 classifier_mode 保持一致
% 'window' = 每8s窗口独立推理 (响应快), 'segment' = 整个运动段统一概率 (更稳定)
para_base.classifier_mode = 'window';
para_base.model_path = 'models';
expert_files = struct( ...
    'arm_curl',   'dataformatlab\Best_Params_Result_multi_wanju1_processed.mat', ...
    'jump_rope',  'dataformatlab\Best_Params_Result_multi_tiaosheng2_processed.mat', ...
    'push_up',    'dataformatlab\Best_Params_Result_multi_fuwo2_processed.mat');
expert_names = fieldnames(expert_files);
para_base.expert_params = struct();
for ei = 1:length(expert_names)
    en = expert_names{ei};
    ef = expert_files.(en);
    if isfile(ef)
        tmp = load(ef);
        bp_hf = tmp.Best_Para_HF;
        para_base.expert_params.(en) = struct( ...
            'Fs_Target', bp_hf.Fs_Target, 'Max_Order', bp_hf.Max_Order, ...
            'LMS_Mu_Base', 0.01, 'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
    else
        warning('专家参数文件不存在: %s, 使用默认值', ef);
        para_base.expert_params.(en) = struct( ...
            'Fs_Target', 25, 'Max_Order', 16, 'LMS_Mu_Base', 0.01, ...
            'Num_Cascade_HF', 2, 'Num_Cascade_Acc', 3);
    end
end

% --- 贝叶斯优化核心设置 ---
% 单次优化的最大迭代次数，数值越大寻找最优解的概率越高，但耗时更长
Max_Iterations = 75; 
% 初始随机种子点数 (先随机跑10次打基础，防止模型一开始就陷入局部最优)
Num_Seed_Points = 10; 
% 重启次数 (每个方案独立运行几次，最终取所有运行中的历史最低误差)
Num_Repeats = 3;

%% 3. 定义搜索空间 (仅后级参数)
% 前级参数 (Fs_Target, Max_Order, LMS_Mu_Base, Num_Cascade) 已在各专家的 Best_Params 中固定,
% 专家模式下 HeartRateSolver 使用 expert_params.(name).Max_Order 而非 para.Max_Order,
% 因此这里只搜索后级处理 (频谱惩罚/谱峰追踪/平滑) 相关参数.
SearchSpace.Spec_Penalty_Width = [0.1, 0.2, 0.3];
SearchSpace.Spec_Penalty_Weight = [0.1, 0.3, 0.5];

SearchSpace.HR_Range_Hz = [15, 20, 25, 30, 35, 40] / 60;
SearchSpace.Slew_Limit_BPM = 8:15;        
SearchSpace.Slew_Step_BPM = [5, 7, 9];    

SearchSpace.HR_Range_Rest = [20, 25, 30, 35, 40, 50] / 60;
SearchSpace.Slew_Limit_Rest = 5:8;        
SearchSpace.Slew_Step_Rest = 3:5;         

SearchSpace.Smooth_Win_Len = [5, 7, 9];
SearchSpace.Time_Bias = [4, 5, 6];        

%% 4. 构建优化变量 (Index Mapping)
% 将每个参数列表映射为 [1, N] 的整数变量，供贝叶斯优化器作为输入维度使用
param_names = fieldnames(SearchSpace);
opt_vars = [];

fprintf('正在初始化贝叶斯优化变量空间...\n');
for i = 1:length(param_names)
    p_name = param_names{i};
    p_list = SearchSpace.(p_name);
    var_name = ['Idx_', p_name];
    % 创建 Integer 类型的优化变量，范围限定为 [1, 该参数列表的长度]
    opt_vars = [opt_vars, optimizableVariable(var_name, [1, length(p_list)], 'Type', 'integer')];
end

%% 5. 第一轮寻优: 针对 Fusion(HF) (3次独立运行取最优)
fprintf('\n======================================================\n');
fprintf('ROUND 1: 寻找 Fusion(HF) 的最优参数 (目标: Motion AAE)\n');
fprintf('======================================================\n');

% 定义目标函数，传入 'HF' 模式标识
ObjFcn_HF = @(T) Wrapper_CostFunction(T, SearchSpace, para_base, 'HF');

% 初始化全局最优记录
Global_Min_HF = Inf;
Best_Para_HF = [];

for run_idx = 1:Num_Repeats
    fprintf('\n--- Fusion(HF) Run %d / %d ---\n', run_idx, Num_Repeats);
    
    % 执行贝叶斯优化核心函数
    results_hf = bayesopt(ObjFcn_HF, opt_vars, ...
        'MaxObjectiveEvaluations', Max_Iterations, ...
        'NumSeedPoints', Num_Seed_Points, ...            % [策略] 增加初始点探索全局
        'AcquisitionFunctionName', 'expected-improvement-plus', ... % [策略] 兼顾探索与开发，自动跳出局部最优
        'IsObjectiveDeterministic', true, ...            % 目标函数是确定性的（相同输入得出相同输出）
        'UseParallel', true, ...                         % [加速] 开启并行计算评估代价函数
        'PlotFcn', {@plotMinObjective}, ...              % 仅绘制最小目标值下降曲线
        'Verbose', 0);                                   % 关闭详细的命令行输出以防刷屏
    
    % 检查本次运行结果是否打破了当前记录的全局最低误差
    current_min = results_hf.MinObjective;
    if current_min < Global_Min_HF
        fprintf('  >>> 发现更好的解! 全局AAE从 %.4f 降低到 %.4f\n', Global_Min_HF, current_min);
        Global_Min_HF = current_min;
        Best_Para_Expert_HF = Extract_Best_Params(results_hf, SearchSpace, para_base);
    else
        fprintf('  >>> 本次结果 (%.4f) 未超过历史最佳 (%.4f)\n', current_min, Global_Min_HF);
    end
end

fprintf('\n>> Round 1 (HF) 最终最低 Motion 误差: %.4f\n', Global_Min_HF);

%% 6. 第二轮寻优: 针对 Fusion(ACC) (3次独立运行取最优)
fprintf('\n======================================================\n');
fprintf('ROUND 2: 寻找 Fusion(ACC) 的最优参数 (目标: Motion AAE)\n');
fprintf('======================================================\n');

% 定义目标函数，传入 'ACC' 模式标识
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
    
    % 检查本次运行结果是否打破历史记录
    current_min = results_acc.MinObjective;
    if current_min < Global_Min_ACC
        fprintf('  >>> 发现更好的解! 全局AAE从 %.4f 降低到 %.4f\n', Global_Min_ACC, current_min);
        Global_Min_ACC = current_min;
        Best_Para_Expert_ACC = Extract_Best_Params(results_acc, SearchSpace, para_base);
    else
        fprintf('  >>> 本次结果 (%.4f) 未超过历史最佳 (%.4f)\n', current_min, Global_Min_ACC);
    end
end

fprintf('\n>> Round 2 (ACC) 最终最低 Motion 误差: %.4f\n', Global_Min_ACC);

%% 7. 保存结果
% 使用 _Expert 后缀避免与标准模式基线参数 (Best_Para_HF/ACC) 混淆
Min_Err_Expert_HF = Global_Min_HF;
Min_Err_Expert_ACC = Global_Min_ACC;

% 使用 fileparts 提取输入数据文件的路径和名称，以便动态生成保存文件名
[dataPath, dataFileName, ~] = fileparts(para_base.FileName);

% 生成新的文件名（加上 .mat 后缀以确保能被正常保存和读取）
newFileName = sprintf('Best_Params_Expert_Result_%s.mat', dataFileName);

% 利用 fullfile 将原路径与新文件名拼接。如果未指定路径，则默认保存在当前目录
if isempty(dataPath)
    SaveFileName = newFileName;
else
    SaveFileName = fullfile(dataPath, newFileName);
end

% 将寻优得到的最优参数组合和最低误差保存到本地文件
save(SaveFileName, 'Best_Para_Expert_HF', 'Best_Para_Expert_ACC', ...
    'Min_Err_Expert_HF', 'Min_Err_Expert_ACC', 'SearchSpace');

fprintf('\n=== 全部寻优结束 ===\n');
fprintf('结果已保存至: %s\n', SaveFileName);
fprintf('请运行 AutoOptimize_Result_Viewer.m 查看详细图表。\n');
% 出于效率考虑不自动关闭并行池，以免下次连续运行又要等待启动
% delete(gcp('nocreate')); 

%% 8. 参数敏感性分析与重要性评估 (以 HF 路径为例)
fprintf('\n======================================================\n');
fprintf('进行参数重要性评估 (Parameter Importance Analysis)\n');
fprintf('======================================================\n');

% 1. 提取贝叶斯优化过程中的所有历史评估数据
% 注意: 确保在第一轮寻优结束时，不要覆盖 results_hf 变量 (原代码需要微调保留该变量)
% 为了能在这一步访问，你需要在第5节中将 results_hf 暴露出来 (原代码已经是这样做的)

if exist('results_hf', 'var') && ~isempty(results_hf.XTrace)
    X_Data = results_hf.XTrace;                 % 输入参数矩阵 (Table格式)
    Y_Data = results_hf.ObjectiveTrace;         % 对应的误差结果 (Motion AAE)
    
    % 清洗数据：排除惩罚项(999)引发的极端异常值，以免影响模型训练
    valid_idx = Y_Data < 100; 
    X_Valid = X_Data(valid_idx, :);
    Y_Valid = Y_Data(valid_idx);

    if height(X_Valid) > 20 % 数据点足够多才进行训练
        % 2. 训练随机森林回归模型
        fprintf('正在训练随机森林模型以评估参数权重...\n');
        rf_model = fitrensemble(X_Valid, Y_Valid, 'Method', 'Bag', 'NumLearningCycles', 50);
        
        % 3. 提取特征重要性
        param_importance = predictorImportance(rf_model);
        
        % 4. 绘制重要性条形图
        figure('Name', 'Parameter Importance (Random Forest)', 'Color', 'w', 'Position', [100, 100, 800, 500]);
        bar(param_importance);
        xticklabels(X_Valid.Properties.VariableNames);
        xtickangle(45);
        ylabel('Importance Score (Higher is more critical)');
        title('Hyperparameter Importance for Motion AAE (HF Path)');
        grid on;
        
        % 打印到命令行
        fprintf('\n参数重要性得分:\n');
        for i = 1:length(param_importance)
            fprintf('%-25s : %.4f\n', X_Valid.Properties.VariableNames{i}, param_importance(i));
        end
        fprintf('建议：得分极低的参数可在后续优化中固定为默认值。\n');
    else
        fprintf('有效数据点不足，跳过随机森林重要性评估。\n');
    end
    
    % 5. 绘制偏依赖图 (Partial Dependence Plot)
    % 该图显示了每个参数在其取值范围内，对最终目标函数(误差)的边缘影响曲线。
    % 曲线越平，参数越不重要。
    figure('Name', 'Objective Model Partial Dependence', 'Color', 'w');
    plotObjectiveModel(results_hf, '1D');
else
    fprintf('未找到 results_hf 数据，无法进行敏感性分析。\n');
end

%% ========================================================================
%  辅助函数区域
%  ========================================================================

function Error_Val = Wrapper_CostFunction(Idx_Table, SearchSpace, para_base, Target_Mode)
    % 1. 解码参数 (将贝叶斯优化器传进来的索引 Index 转换回真实的参数 Value)
    current_para = para_base;
    param_names = fieldnames(SearchSpace);
    
    for i = 1:length(param_names)
        p_name = param_names{i};
        var_name = ['Idx_', p_name];
        idx = Idx_Table.(var_name); % 获取对应的离散索引值
        current_para.(p_name) = SearchSpace.(p_name)(idx); % 映射为真实数值
    end
    
    % 2. 运行核心解算算法
    try
        Res = HeartRateSolver_cas_chengfa(current_para);
        
        % 3. 根据目标模式返回对应的全局 AAE (Total AAE)
        if strcmp(Target_Mode, 'HF')
            Error_Val = Res.err_stats(4, 1);      % Fusion(HF) 全局 AAE
        elseif strcmp(Target_Mode, 'ACC')
            Error_Val = Res.err_stats(5, 1);       % Fusion(ACC) 全局 AAE
        else
            error('未知的优化目标模式');
        end
        
    catch
        % 惩罚项: 如果遇到引发报错的参数组合（例如产生 NaN），给予极大的惩罚误差
        % 引导优化器主动避开这些不稳定的参数区域
        Error_Val = 999; 
    end
end

function Best_Para = Extract_Best_Params(bayes_results, SearchSpace, para_base)
    % 功能：从 bayesopt 返回的综合结果对象中，提取出最优解的索引，并映射回真实值参数字典
    
    % 使用 XAtMinObjective 直接获取使得目标函数最小的最优参数 Table 行
    best_idx_table = bayes_results.XAtMinObjective;
    
    Best_Para = para_base;
    param_names = fieldnames(SearchSpace);
    
    for i = 1:length(param_names)
        p_name = param_names{i};
        var_name = ['Idx_', p_name];
        
        % [鲁棒性处理] 
        % 极端情况下，如果优化器找到了多个参数组合得到完全相同的最小误差，
        % XAtMinObjective 可能会返回包含多行的 Table。默认取第一行作为最优解即可。
        if height(best_idx_table) > 1
            % 如果是多行 Table，使用 {} 或 () 索引提取第一行对应列的值
            best_idx = best_idx_table{1, var_name};
        else
            % 如果是单行 Table，直接使用点索引
            best_idx = best_idx_table.(var_name);
        end
        
        % 映射回真实参数值并覆盖存入 Best_Para 结构体
        Best_Para.(p_name) = SearchSpace.(p_name)(best_idx);
    end
end