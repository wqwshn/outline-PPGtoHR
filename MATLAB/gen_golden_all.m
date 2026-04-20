function gen_golden_all()
% gen_golden_all - 一次性生成全部 Python 单元测试需要的 .mat 金标快照
%
% 用法：
%   1. 在 MATLAB 中 cd 到本仓库根目录
%   2. addpath('MATLAB');  cd MATLAB;
%   3. gen_golden_all
%   4. 生成的 .mat 会写入 ../python/tests/golden/
%
% 每个 .mat 文件包含同一函数的输入与输出，Python 测试通过 scipy.io.loadmat
% 加载后逐字段调用 Python 实现并 assert_allclose。

    % 输出目录
    out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'python', 'tests', 'golden');
    if ~isfolder(out_dir)
        mkdir(out_dir);
    end
    fprintf('金标输出目录: %s\n', out_dir);

    % 取一段真实 PPG/ACC/HF 数据做种子
    sample_mat = fullfile(fileparts(mfilename('fullpath')), '..', '20260418test_python', ...
                         'multi_tiaosheng1_processed.mat');
    if ~isfile(sample_mat)
        error('找不到样本数据 %s，无法生成金标', sample_mat);
    end
    S = load(sample_mat);
    raw = table2array(S.data);
    Fs = 100;

    % 取 8 秒（800 点）切片
    N = 8 * Fs;
    seg_ppg  = raw(1:N, 6);
    seg_hf1  = raw(1:N, 4);
    seg_hf2  = raw(1:N, 5);
    seg_accx = raw(1:N, 9);
    seg_accy = raw(1:N, 10);
    seg_accz = raw(1:N, 11);

    %% ---------- find_maxpeak ----------
    rng(42);
    case_freqs = [1.2, 2.4, 1.8, 0.9, 3.0]';
    case_amps  = [0.7, 0.9, 0.5, 0.3, 0.8]';
    expected_sorted = Find_maxpeak(case_freqs, case_freqs, case_amps);
    save(fullfile(out_dir, 'find_maxpeak.mat'), 'case_freqs', 'case_amps', 'expected_sorted');
    fprintf('  saved find_maxpeak.mat\n');

    %% ---------- find_real_hr ----------
    ref_data = S.ref_data;
    case_time_currents = [10; 30; 60; 90; 120];
    expected_hr_real = zeros(numel(case_time_currents), 1);
    for k = 1:numel(case_time_currents)
        expected_hr_real(k) = Find_realHR('dummy', case_time_currents(k), ref_data);
    end
    save(fullfile(out_dir, 'find_real_hr.mat'), 'ref_data', 'case_time_currents', 'expected_hr_real');
    fprintf('  saved find_real_hr.mat\n');

    %% ---------- find_near_biggest ----------
    case_fre  = [1.5, 1.6, 2.0, 2.5, 3.0, 1.2]';   % 6 个候选，函数只看前 5 个
    case_hr_prev = 1.55;
    case_range_plus  = 0.4;
    case_range_minus = -0.3;
    [expected_hr, expected_which] = Find_nearBiggest(case_fre, case_hr_prev, ...
                                                     case_range_plus, case_range_minus);
    save(fullfile(out_dir, 'find_near_biggest.mat'), ...
         'case_fre', 'case_hr_prev', 'case_range_plus', 'case_range_minus', ...
         'expected_hr', 'expected_which');
    fprintf('  saved find_near_biggest.mat\n');

    %% ---------- fft_peaks ----------
    case_signal = seg_ppg;
    case_fs     = Fs;
    case_percent = 0.3;
    [expected_fre, expected_famp] = FFT_Peaks(case_signal, case_fs, case_percent);
    save(fullfile(out_dir, 'fft_peaks.mat'), ...
         'case_signal', 'case_fs', 'case_percent', 'expected_fre', 'expected_famp');
    fprintf('  saved fft_peaks.mat\n');

    %% ---------- ppg_peace ----------
    case_signal_pp = seg_ppg;
    case_fs_pp     = Fs;
    expected_ratio = PpgPeace(case_signal_pp, case_fs_pp);
    save(fullfile(out_dir, 'ppg_peace.mat'), ...
         'case_signal_pp', 'case_fs_pp', 'expected_ratio');
    fprintf('  saved ppg_peace.mat\n');

    %% ---------- lms_filter (3 cases) ----------
    rng(7);
    case_mu = 0.005;
    case_M  = 3;
    case_K  = 0;
    case_u  = seg_accx;
    case_d  = seg_ppg;
    [exp_e_K0, exp_w_K0, exp_ee_K0] = lmsFunc_h(case_mu, case_M, case_K, case_u, case_d);

    case_K2 = 1;
    [exp_e_K1, exp_w_K1, exp_ee_K1] = lmsFunc_h(case_mu, case_M, case_K2, case_u, case_d);

    save(fullfile(out_dir, 'lms_filter.mat'), ...
         'case_mu', 'case_M', 'case_K', 'case_K2', 'case_u', 'case_d', ...
         'exp_e_K0', 'exp_w_K0', 'exp_ee_K0', ...
         'exp_e_K1', 'exp_w_K1', 'exp_ee_K1');
    fprintf('  saved lms_filter.mat\n');

    %% ---------- choose_delay ----------
    % 给 ChooseDelay1218 准备完整长度的 ppg/acc/hf 数组
    full_ppg  = raw(:, 6);
    full_hf1  = raw(:, 4);
    full_hf2  = raw(:, 5);
    full_accx = raw(:, 9);
    full_accy = raw(:, 10);
    full_accz = raw(:, 11);
    cd_time1 = 5.0;
    cd_fs    = Fs;
    [exp_mh, exp_ma, exp_td_h, exp_td_a] = ChooseDelay1218( ...
        cd_fs, cd_time1, full_ppg, ...
        {full_accx, full_accy, full_accz}, {full_hf1, full_hf2});
    save(fullfile(out_dir, 'choose_delay.mat'), ...
         'cd_fs', 'cd_time1', 'full_ppg', 'full_hf1', 'full_hf2', ...
         'full_accx', 'full_accy', 'full_accz', ...
         'exp_mh', 'exp_ma', 'exp_td_h', 'exp_td_a');
    fprintf('  saved choose_delay.mat\n');

    %% ---------- data_loader (整段 processed table 保存为 struct) ----------
    % 把 multi_tiaosheng1_processed.mat 中的 data table + ref_data 转为
    % Python scipy.io.loadmat 友好的 struct 格式。
    data_struct = table2struct(S.data, 'ToScalar', true);
    ref_data_dl = S.ref_data;
    sensor_csv_rel = fullfile('20260418test_python', 'multi_tiaosheng1.csv');
    gt_csv_rel     = fullfile('20260418test_python', 'multi_tiaosheng1_ref.csv');
    save(fullfile(out_dir, 'data_loader.mat'), ...
         'data_struct', 'ref_data_dl', 'sensor_csv_rel', 'gt_csv_rel');
    fprintf('  saved data_loader.mat\n');

    %% ---------- heart_rate_solver (端到端单一典型场景) ----------
    % 只跑 multi_tiaosheng1 —— 与用户约定：所有原始数据文件结构相同，
    % 一个场景即可充分验证重构等价性。
    scenarios = {'multi_tiaosheng1'};
    para = default_solver_params();
    for k = 1:numel(scenarios)
        sc = scenarios{k};
        para.FileName = fullfile('..', '20260418test_python', [sc '_processed.mat']);
        if ~isfile(para.FileName)
            warning('跳过场景 %s（mat 文件不存在）', sc); continue;
        end
        Res = HeartRateSolver_cas_chengfa(para);
        out_file = fullfile(out_dir, ['e2e_' sc '.mat']);
        save(out_file, 'para', 'Res');
        fprintf('  saved %s\n', ['e2e_' sc '.mat']);
    end

    fprintf('\n全部金标快照生成完毕。\n');
end


function p = default_solver_params()
    p.FileName              = '';
    p.Fs_Target             = 100;
    p.Max_Order             = 16;
    p.Time_Start            = 1;
    p.Time_Buffer           = 10;
    p.Calib_Time            = 30;
    p.Motion_Th_Scale       = 2.5;
    p.Spec_Penalty_Enable   = 1;
    p.Spec_Penalty_Weight   = 0.2;
    p.Spec_Penalty_Width    = 0.2;
    p.HR_Range_Hz           = 25 / 60;
    p.Slew_Limit_BPM        = 10;
    p.Slew_Step_BPM         = 7;
    p.HR_Range_Rest         = 30 / 60;
    p.Slew_Limit_Rest       = 6;
    p.Slew_Step_Rest        = 4;
    p.Smooth_Win_Len        = 7;
    p.Time_Bias             = 5;
end
