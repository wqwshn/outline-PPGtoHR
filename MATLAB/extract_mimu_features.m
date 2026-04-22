function features = extract_mimu_features(accx, accy, accz, gyrox, gyroy, gyroz, Fs)
% extract_mimu_features 提取 75 维 IMU 特征 (2s 窗口)
% 输入: 各轴信号片段 + 采样率
% 输出: (75, 1) 特征向量

    channels = {accx(:), accy(:), accz(:), gyrox(:), gyroy(:), gyroz(:)};
    features = zeros(75, 1);
    idx = 1;

    % === 6 channels x 10 features (9 time + 1 freq) = 60 ===
    for c = 1:6
        sig = channels{c};

        % 时域: 9 features
        features(idx)   = mean(sig);       idx = idx + 1;
        features(idx)   = std(sig);        idx = idx + 1;
        features(idx)   = min(sig);        idx = idx + 1;
        features(idx)   = max(sig);        idx = idx + 1;
        features(idx)   = max(sig) - min(sig); idx = idx + 1;
        features(idx)   = mean(sig.^2);    idx = idx + 1; % energy
        features(idx)   = sum(diff(sign(sig)) ~= 0) / (length(sig) - 1); idx = idx + 1; % zcr
        features(idx)   = skewness(sig);   idx = idx + 1;
        features(idx)   = kurtosis(sig);   idx = idx + 1;

        % 频域: dominant_freq
        nperseg = min(length(sig), 256);
        [f, P] = pwelch(sig, hamming(nperseg), [], [], Fs);
        non_dc = P(2:end);
        if ~isempty(non_dc) && max(non_dc) > 0
            features(idx) = f(2 + find(non_dc == max(non_dc), 1) - 1);
        else
            features(idx) = 0;
        end
        idx = idx + 1;
    end

    % === Magnitude features: acc_mag x 4 + gyro_mag x 4 = 8 ===
    acc_mag = sqrt(accx(:).^2 + accy(:).^2 + accz(:).^2);
    gyro_mag = sqrt(gyrox(:).^2 + gyroy(:).^2 + gyroz(:).^2);

    for mag = 1:2
        if mag == 1, sig = acc_mag; else, sig = gyro_mag; end
        features(idx) = mean(sig); idx = idx + 1;
        features(idx) = std(sig);  idx = idx + 1;
        features(idx) = mean(sig.^2); idx = idx + 1; % energy
        nperseg = min(length(sig), 256);
        [f, P] = pwelch(sig, hamming(nperseg), [], [], Fs);
        non_dc = P(2:end);
        if ~isempty(non_dc) && max(non_dc) > 0
            features(idx) = f(2 + find(non_dc == max(non_dc), 1) - 1);
        else
            features(idx) = 0;
        end
        idx = idx + 1;
    end

    % === Cross-correlation: 7 ===
    features(idx) = safe_corr(acc_mag, gyro_mag); idx = idx + 1;
    features(idx) = safe_corr(accx(:), accy(:));  idx = idx + 1;
    features(idx) = safe_corr(accx(:), accz(:));  idx = idx + 1;
    features(idx) = safe_corr(accy(:), accz(:));  idx = idx + 1;
    features(idx) = safe_corr(gyrox(:), gyroy(:)); idx = idx + 1;
    features(idx) = safe_corr(gyrox(:), gyroz(:)); idx = idx + 1;
    features(idx) = safe_corr(gyroy(:), gyroz(:)); idx = idx + 1;
end

function r = safe_corr(a, b)
    if std(a) == 0 || std(b) == 0
        r = 0;
    else
        r = corrcoef(a, b);
        r = r(1, 2);
    end
end
