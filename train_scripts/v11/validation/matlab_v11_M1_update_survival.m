%% M1：不动点的记忆性——20 个粒子的存档更新间隔是几何分布还是幂律？
% 运行方法：在 MATLAB 里把 Current Folder 设为本目录（train_scripts/v11/validation），
%           打开本文件，点 Run。只依赖 base MATLAB（自带 KM 生存分析，不用工具箱）。
%
% 问题（方向锁定 §11 M1）：如果「下一次找到更好的解」的等待时间是**无记忆的**
% （几何分布、风险率恒定），说明搜索只在**同一个吸引盆**里随机游走，那么 C3 的
% 40 步展开就是在同一个盆里做更长的顺序组合 —— 合理。反之若是**幂律/重尾**
% （风险率随时步衰减），则「无记忆」被否。**但幂律不等于分层盆地**：长尾还可能只是
% 「同一盆内改进幅度越来越小」。所以本脚本额外做两件事来区分：
%   (a) 存档更新带来的 SIZE 增量分布（若 +1 占绝对多数 => 同盆递减回报）；
%   (b) 用 KM 直接算 C3 展开窗口覆盖到一次改进事件的概率（40 步 vs 200 步）。
%
% 数据：data/trace_pevents.csv（id,K,particle,block,steps,size：每个粒子的存档更新时刻）
%       data/trace_pcensor.csv（id,K,particle,last_block,...：跑满 1 s 的最后记录，用于右删失）
% 方法：把每个粒子的更新序列变成一串「等待时长」W（单位 = block），最后一段是右删失；
%       汇合所有粒子做离散 Kaplan-Meier，得到 S(t)=P(W>t) 与风险率 h(t)。
%       两个候选模型各自最小二乘拟合 ln S(t)：
%         几何（无记忆）: S(t) = (1-p)^t      -> ln S = t*ln(1-p)        1 个参数
%         幂律（分层）  : S(t) = S1 * t^-a    -> ln S = ln S1 - a*ln t   2 个参数
%       用 AICc 比较（AIC = n*ln(RSS/n)+2k，AICc = AIC + 2k(k+1)/(n-k-1)）。
%
% 输出：figures/fig_M1_update_survival.png
%       results/matlab_M1_report.txt, results/matlab_M1_survival.csv

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
data = fullfile(here, 'data');
figs = fullfile(here, 'figures');
res  = fullfile(here, 'results');
if ~exist(figs, 'dir'), mkdir(figs); end
if ~exist(res, 'dir'), mkdir(res); end

E = v11_readtable_csv(fullfile(data, 'trace_pevents.csv'));
C = v11_readtable_csv(fullfile(data, 'trace_pcensor.csv'));
log_fid = fopen(fullfile(res, 'matlab_M1_report.txt'), 'w', 'n', 'UTF-8');
fprintf(log_fid, 'M1 存档更新间隔的生存分析（离散 KM，单位 block=10 步）\n');

arms = {'K200', 'K20'};
armName = {'K=200（窄、20 粒子）', 'K=20（宽、20 粒子）'};
figure('Color', 'w', 'Position', [40 60 1500 430]);
fitTab = table();

for ai = 1:2
    key = arms{ai};
    mE = strcmp(E.K, key);
    mC = strcmp(C.K, key);
    ids = unique(E.id(mE));
    dur = [];
    evt = [];
    nPart = 0;
    nNeverAfterFirst = 0;
    inc = [];
    parts_dur = {};
    parts_evt = {};
    parts_last = {};
    for gi = 1:numel(ids)
        gid = ids(gi);
        parts = unique(E.particle(mE & E.id == gid));
        for pi = 1:numel(parts)
            pid = parts(pi);
            m = mE & E.id == gid & E.particle == pid;
            b = sort(E.block(m));
            mc = mC & C.id == gid & C.particle == pid;
            if any(mc)
                lastb = C.last_block(mc);
            else
                lastb = max(b);
            end
            if numel(b) < 2
                nNeverAfterFirst = nNeverAfterFirst + 1;
                continue;
            end
            if numel(b) >= 2
                sz = E.size(m);
                [~, ord2] = sort(E.block(m));
                sz = sz(ord2);
                dsz = diff(sz);
                inc = [inc; dsz(:)];   %#ok<AGROW>
            end
            nPart = nPart + 1;
            w = diff(b);
            dur = [dur; w(:)];            %#ok<AGROW>
            evt = [evt; ones(numel(w), 1)]; %#ok<AGROW>
            % 最后一段：最后一个更新到跑满之间的删失等待
            wc = lastb - b(end);
            if wc > 0
                dur = [dur; wc];            %#ok<AGROW>
                evt = [evt; 0];             %#ok<AGROW>
            end
            % 每粒子单独存一份（用于排除「混合人群」假象：汇合 KM 会把
            % 快慢不同的粒子混在一起，即使每个粒子都是几何分布，汇合后
            % 风险率也会衰减。所以必须逐粒子各拟合一次。）
            if numel(b) >= 4
                parts_dur{end + 1} = [w(:); max(0, lastb - b(end))];  %#ok<SAGROW>
                parts_evt{end + 1} = [ones(numel(w), 1); double(lastb > b(end))]; %#ok<SAGROW>
                parts_last{end + 1} = lastb;                          %#ok<SAGROW>
            end
        end
    end
    tmaxEval = 60;                       % 评估到 60 block（600 步）
    [tt, S, atRisk, ndie] = v11_km(dur, evt, tmaxEval);
    % 只保留有意义的区间（生存概率 > 1% 且风险集 >= 15）
    keep = S > 0.01 & atRisk >= 15;
    keep(1) = true;
    tk = tt(keep); Sk = S(keep);
    haz = nan(1, numel(tt));
    for k = 1:numel(tt) - 1
        if S(k) > 0
            haz(k) = 1 - S(k + 1) / S(k);       % 窗口风险率
        end
    end

    % ---- 模型拟合 ----
    lnS = log(Sk);
    % 几何：lnS = a*t，a = ln(1-p) <= 0，1 参数，过原点
    aGeo = (tk(:) \ lnS(:));
    rGeo = lnS(:) - aGeo * tk(:);
    rssGeo = sum(rGeo .^ 2);
    pGeo = 1 - exp(aGeo);
    % 幂律：lnS = c - alpha*ln t，2 参数
    X = [ones(numel(tk), 1), log(tk(:))];
    beta = X \ lnS(:);
    rPow = lnS(:) - X * beta;
    rssPow = sum(rPow .^ 2);
    n = numel(tk);
    aicGeo = n * log(rssGeo / n) + 2 * 1;
    aicPow = n * log(rssPow / n) + 2 * 2;
    aiccGeo = aicGeo + 2 * 1 * 2 / (n - 1 - 1);
    aiccPow = aicPow + 2 * 2 * 3 / (n - 2 - 1);
    dAIC = aiccPow - aiccGeo;            % <0 支持幂律，>0 支持几何
    if dAIC > 2
        verdict = '几何（无记忆）：单一吸引盆，C3 直接展开即可';
    elseif dAIC < -2
        verdict = '幂律（风险率衰减）：无记忆被否；机制看增量分布，勿直接推分层';
    else
        verdict = '两者不可分（|dAICc|<=2）：按几何处理，但需注意';
    end

    fprintf('\n== %s ==\n', armName{ai});
    fprintf('粒子数 %d（更新次数 <2 的粒子 %d 个），等待样本 %d 段（事件 %d，删失 %d）\n', ...
            nPart, nNeverAfterFirst, numel(dur), sum(evt == 1), sum(evt == 0));
    fprintf('等待时长（block）：中位 %.1f，均值 %.2f，最大 %d\n', median(dur), mean(dur), max(dur));
    fprintf('KM 中位存活（S=0.5）：%s block\n', mat2str(kmedian(tt, S, 0.5)));
    fprintf('几何：p=%.4f（平均间隔 %.2f block），RSS=%.3f，AICc=%.1f\n', pGeo, 1 / max(pGeo, 1e-9), rssGeo, aiccGeo);
    fprintf('幂律：S1=%.4f, alpha=%.3f，RSS=%.3f，AICc=%.1f\n', exp(beta(1)), -beta(2), rssPow, aiccPow);
    fprintf('dAICc（幂律-几何）= %+.1f -> %s\n', dAIC, verdict);
    fprintf('风险率：第 1-5 block 均值 %.4f，第 20-30 block 均值 %.4f（比值 %.2f）\n', ...
            mean(haz(1:5), 'omitnan'), mean(haz(20:30), 'omitnan'), ...
            mean(haz(20:30), 'omitnan') / max(mean(haz(1:5), 'omitnan'), 1e-9));
    fprintf(log_fid, '\n== %s ==\n粒子 %d，等待样本 %d（事件 %d/删失 %d），中位 %.1f block\n', ...
            armName{ai}, nPart, numel(dur), sum(evt == 1), sum(evt == 0), median(dur));
    fprintf(log_fid, '几何 p=%.4f RSS=%.3f AICc=%.1f\n幂律 S1=%.4f alpha=%.3f RSS=%.3f AICc=%.1f\ndAICc=%+.1f -> %s\n', ...
            pGeo, rssGeo, aiccGeo, exp(beta(1)), -beta(2), rssPow, aiccPow, dAIC, verdict);
    fprintf(log_fid, '风险率 h(1-5)=%.4f h(20-30)=%.4f\n', ...
            mean(haz(1:5), 'omitnan'), mean(haz(20:30), 'omitnan'));

    % ---- C3 展开长度校准：一次「档案更新」事件要等多久，40 步窗口能覆盖多少 ----
    if numel(S) >= 4
        pwin40 = 1 - S(4);            % S(t)=P(W>t)，40 步 = 4 block
    else
        pwin40 = NaN;
    end
    pwin200 = 1 - S(min(numel(S), 20));   % 200 步 = 20 block
    pwin50 = 1 - S(min(numel(S), 5));     % 5 block = 50 步（= trace 的解码粒度）
    fprintf('平均等待 %.1f block = %.0f 步；P(窗口内有改进)：40 步 %.1f%% / 50 步 %.1f%% / 200 步 %.1f%%\n', ...
            mean(dur), 10 * mean(dur), 100 * pwin40, 100 * pwin50, 100 * pwin200);
    fprintf('  （注意：trace 每 5 block 才解码一次，所以 40 步窗口在 5-block 分辨率下根本观测不到事件）\n');
    fprintf(log_fid, '平均等待 %.1f block = %.0f 步；P(40 步内有改进)=%.3f，P(50 步内)=%.3f，P(200 步内)=%.3f\n', ...
            mean(dur), 10 * mean(dur), pwin40, pwin50, pwin200);
    fprintf(log_fid, '（解码粒度 5 block=50 步；40 步窗口在 5-block 分辨率下观测不到事件）\n');

    fitTab = [fitTab; table({key}, nPart, numel(dur), median(dur), pGeo, rssGeo, aiccGeo, ...
                            exp(beta(1)), -beta(2), rssPow, aiccPow, dAIC, {verdict}, ...
             'VariableNames', {'arm', 'n_particles', 'n_waits', 'median_block', 'p_geo', ...
                               'rss_geo', 'aicc_geo', 'S1_pow', 'alpha_pow', 'rss_pow', 'aicc_pow', ...
                               'dAICc', 'verdict'})];  %#ok<AGROW>

    % ---- 逐粒子复核（排除混合人群假象）----
    npp = numel(parts_dur);
    dpp = nan(npp, 1);
    ratioPP = nan(npp, 1);
    medPP = nan(npp, 1);
    for q = 1:npp
        dd = parts_dur{q}; ee = parts_evt{q};
        dd = dd(dd > 0);
        if numel(dd) < 5, continue; end
        [tq, Sq] = v11_km(dd, ee, max(parts_last{q}, 2));
        ok = Sq > 0.02;
        ok(1) = true;
        tq = tq(ok); Sq = Sq(ok);
        if numel(tq) < 6, continue; end
        lq = log(Sq(:));
        a1 = tq(:) \ lq;
        r1 = sum((lq - a1 * tq(:)) .^ 2);
        X1 = [ones(numel(tq), 1), log(tq(:))];
        b1 = X1 \ lq;
        r2 = sum((lq - X1 * b1) .^ 2);
        nn = numel(tq);
        aicc1 = nn * log(max(r1, 1e-12) / nn) + 2 + 2 * 2 / max(nn - 2, 1);
        aicc2 = nn * log(max(r2, 1e-12) / nn) + 4 + 2 * 6 / max(nn - 3, 1);
        dpp(q) = aicc2 - aicc1;
        h = 1 - Sq(2:end) ./ max(Sq(1:end - 1), 1e-12);
        half = floor(numel(h) / 2);
        if half >= 2
            ratioPP(q) = mean(h(half + 1:end)) / max(mean(h(1:half)), 1e-12);
        end
        medPP(q) = median(dd);
    end
    fprintf('存档更新带来的 SIZE 增量：+1 %d 次（%.0f%%），+2 %d 次，>=+3 %d 次，中位 %+.0f\n', ...
            sum(inc == 1), 100 * mean(inc == 1), sum(inc == 2), sum(inc >= 3), median(inc));
    fprintf('  -> 若绝对多数是 +1，说明长尾是「同一盆内缓慢改进的递减回报」，不是跳盆\n');
    fprintf(log_fid, '增量：+1 %.0f%% / +2 %d / >=+3 %d（共 %d 次更新）\n', ...
            100 * mean(inc == 1), sum(inc == 2), sum(inc >= 3), numel(inc));
    nPow = sum(dpp < -2); nGeo = sum(dpp > 2); nTie = sum(abs(dpp) <= 2);
    fprintf('逐粒子（n=%d，每个粒子单独 KM 拟合）：支持幂律 %d，支持几何 %d，不可分 %d；中位 dAICc %+.1f\n', ...
            sum(~isnan(dpp)), nPow, nGeo, nTie, median(dpp, 'omitnan'));
    fprintf('逐粒子风险率「后半/前半」中位比值 %.2f（几何应 ≈1.0；%d/%d 个粒子 < 0.8）\n', ...
            median(ratioPP, 'omitnan'), sum(ratioPP < 0.8), sum(~isnan(ratioPP)));
    fprintf('逐粒子中位等待时长：p10 %.0f / p50 %.0f / p90 %.0f block（跨粒子离散 → 混合人群效应）\n', ...
            prctile_simple(medPP, 10), prctile_simple(medPP, 50), prctile_simple(medPP, 90));
    fprintf(log_fid, '逐粒子：幂律 %d / 几何 %d / 不可分 %d，中位 dAICc %+.1f；风险率后半/前半中位 %.2f\n', ...
            nPow, nGeo, nTie, median(dpp, 'omitnan'), median(ratioPP, 'omitnan'));
    % 判据修正：若逐粒子多数仍支持幂律 → 分层真实；若多数支持几何而汇合支持幂律 → 混合人群假象
    if nPow > nGeo
        verdictPP = '逐粒子多数支持幂律：无记忆被否是真的；机制见增量/风险率两条';
    elseif nGeo > nPow
        verdictPP = '逐粒子多数支持几何：汇合的幂律是「快慢粒子混合」假象，C3 直接展开即可';
    else
        verdictPP = '逐粒子不可分：按几何处理，但汇合曲线重尾';
    end
    fprintf('逐粒子判据 -> %s\n', verdictPP);
    fprintf(log_fid, '逐粒子判据 -> %s\n', verdictPP);

    dppLast = dpp; nPowLast = nPow; nGeoLast = nGeo; nTieLast = nTie;
    % ---- 图 ----
    subplot(1, 3, ai);
    hold on; box on; grid on;
    stairs(tt, S, '-', 'LineWidth', 2, 'Color', [0.15 0.35 0.75]);
    plot(tk, exp(aGeo * tk), '--', 'LineWidth', 1.6, 'Color', [0.85 0.35 0.15]);
    plot(tk, exp(beta(1) + beta(2) * log(tk)), '-.', 'LineWidth', 1.6, 'Color', [0.15 0.60 0.30]);
    plot(tt(1:end - 1), haz(1:end - 1), ':', 'LineWidth', 1.2, 'Color', [0.45 0.45 0.45]);
    set(gca, 'XScale', 'log', 'YScale', 'log');
    ylim([1e-3 1.2]);
    xlim([1 60]);
    xlabel('等待时长 W（block，1 block = 10 步；对数轴）');
    ylabel('S(W) 与风险率（对数轴）');
    title(sprintf('%s：dAICc=%+.1f -> %s', key, dAIC, iff(dAIC > 2, '几何', iff(dAIC < -2, '幂律', '不可分'))));
    legend({'KM 生存', sprintf('几何 p=%.3f', pGeo), sprintf('幂律 alpha=%.2f', -beta(2)), '风险率 h(t)'}, ...
           'Location', 'southwest', 'FontSize', 7);
end
subplot(1, 3, 3);
hold on; box on; grid on;
if exist('dppLast', 'var') && ~isempty(dppLast)
    histogram(dppLast, -60:10:60, 'FaceColor', [0.40 0.45 0.75]);
    xline(0, 'k-'); xline(-2, 'k:'); xline(2, 'k:');
    xlabel('逐粒子 dAICc（幂律 − 几何；<0 支持幂律）');
    ylabel('粒子数');
    title(sprintf('逐粒子：幂律 %d / 几何 %d / 不可分 %d', nPowLast, nGeoLast, nTieLast));
end
sgtitle('M1 存档更新间隔：无记忆（几何）还是分层（幂律）？左/中=汇合 KM，右=逐粒子');

exportgraphics(gcf, fullfile(figs, 'fig_M1_update_survival.png'), 'Resolution', 150);
writetable(fitTab, fullfile(res, 'matlab_M1_survival.csv'));
fprintf('\n图已写到 %s\n', figs);
fclose(log_fid);

function v = iff(c, a, b)
if c, v = a; else, v = b; end
end

function v = prctile_simple(x, q)
x = x(~isnan(x));
if isempty(x), v = NaN; return; end
x = sort(x(:));
idx = max(1, min(numel(x), round(q / 100 * numel(x))));
v = x(idx);
end

function t = kmedian(tt, S, target)
% 找到 S 首次 <= target 的时刻
idx = find(S <= target, 1);
if isempty(idx), t = NaN; else, t = tt(idx); end
end
