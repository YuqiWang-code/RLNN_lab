%% M6：num_k 扫描（部署口径）——K 到底该用多少？
% 运行方法：在 MATLAB 里把 Current Folder 设为本目录（train_scripts/v11/validation），
%           打开本文件，点 Run。只依赖 base MATLAB。
%
% 背景：t4 早期从 36 图的**轨迹**里看到「K=20 比 K=200 强 +0.11」，但那个口径是
%   20 粒子档案 + 每 5 block 解码；部署口径是 K=200 档案 + decode_top=4 + 末端 PAVE。
%   两者不是同一个测量对象，所以在部署口径下重测，只有一个变量 num_k。
%
% 三个实验（全部同一配方：padded/captured、dense+fp16、seed 1017、
%   warm start best.pt、1.0 s deadline、decode_every=1000 + decode_top=4）：
%   (A) 时钟匹配：K ∈ {20,50,100,200} 各跑 1.0 s，200 张图；
%   (B) 步数匹配：K ∈ {20,200} 都恰好跑 808 步（= 部署 1 s 的实际步数），100 张图（脚本上限）；
%   (C) 档案解耦：用 36 图轨迹（data/trace_best.csv）比较「K=200 只看 20 个档案列」
%       与「K=20 看 20 个档案列」，说明 +0.11 来自档案可见性还是 K 本身。
%
% 数据：data/k_sweep/k{20,50,100,200}_n200_1s/*.csv（实验 A，server 产出）
%       data/k_sweep/steps808_k{20,200}_n200/*.csv（实验 B）
%       data/trace_best.csv（实验 C）
% 输出：figures/fig_M6_k_sweep.png
%       results/matlab_M6_report.txt, results/matlab_M6_summary.csv

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
data = fullfile(here, 'data');
figs = fullfile(here, 'figures');
res  = fullfile(here, 'results');
if ~exist(figs, 'dir'), mkdir(figs); end
if ~exist(res, 'dir'), mkdir(res); end
ksweep = fullfile(data, 'k_sweep');
log_fid = fopen(fullfile(res, 'matlab_M6_report.txt'), 'w', 'n', 'UTF-8');

%% ---------- 实验 A：时钟匹配 ----------
Ks = [20 50 100 200];
sizes = cell(1, numel(Ks));
gid = cell(1, numel(Ks));
meanSize = nan(1, numel(Ks));
meanSteps = nan(1, numel(Ks));
meanMs = nan(1, numel(Ks));
fprintf('== M6-A 时钟匹配（1.0 s，200 图，部署口径）==\n');
fprintf(log_fid, 'M6-A 时钟匹配 1.0 s 200 图\n');
for ki = 1:numel(Ks)
    d = fullfile(ksweep, sprintf('k%d_n200_1s', Ks(ki)));
    files = dir(fullfile(d, '*.csv'));
    if isempty(files)
        fprintf('  缺 %s（先跑 server 脚本）\n', d);
        continue;
    end
    T = v11_readtable_csv(fullfile(files(1).folder, files(1).name));
    T = T(T.repeat == 0, :);          % 部署口径是单次运行（repeat 0），repeat 1 留作噪声估计
    [g, keep] = unique(T.graph_id);
    sizes{ki} = T.final_size(keep);
    gid{ki} = g;
    meanSize(ki) = mean(T.final_size);
    meanSteps(ki) = mean(T.steps);
    meanMs(ki) = mean(T.ms_per_step);
    fprintf('  K=%-4d mean SIZE %8.4f  步数 %7.0f  ms/步 %6.3f  n=%d\n', ...
            Ks(ki), meanSize(ki), meanSteps(ki), meanMs(ki), height(T));
    fprintf(log_fid, 'K=%d mean %.4f steps %.0f ms/step %.3f n %d\n', ...
            Ks(ki), meanSize(ki), meanSteps(ki), meanMs(ki), height(T));
end

%% 配对自助法 vs K=200（图级配对，5000 次，固定种子）
refIdx = find(Ks == 200, 1);
fprintf('\n-- 图级配对 5000 次自助 95%% CI（对 K=200）--\n');
fprintf(log_fid, '配对自助 vs K=200\n');
ciLo = nan(1, numel(Ks));
ciHi = nan(1, numel(Ks));
for ki = 1:numel(Ks)
    if ki == refIdx || isempty(sizes{ki}) || isempty(sizes{refIdx}), continue; end
    d = pair_diff(sizes{ki}, gid{ki}, sizes{refIdx}, gid{refIdx});
    [lo, hi, mu] = boot_ci(d, 5000);
    ciLo(ki) = lo; ciHi(ki) = hi;
    fprintf('  K=%-4d - K=200：%+0.4f  CI [%+0.4f, %+0.4f]  %s\n', ...
            Ks(ki), mu, lo, hi, iff(lo > 0, 'CI 下界>0', iff(hi < 0, 'CI 上界<0', 'CI 跨 0')));
    fprintf(log_fid, 'K=%d - K200 %+.4f CI [%+.4f,%+.4f]\n', Ks(ki), mu, lo, hi);
end

%% 判据（按 captain 协议：CI 下界 > +0.05 → 有空间并扩 1000 图）
if all(isnan(ciLo)) || max([ciLo, -inf]) <= 0.05
    if ~isnan(meanSize(refIdx)) && meanSize(refIdx) == max(meanSize)
        verdictA = '部署口径下 K=200 最优（无更小 K 的 CI 下界超过 +0.05）';
    else
        verdictA = '部署口径下 K=200 不是最优，但没有更小 K 的 CI 下界超过 +0.05';
    end
else
    verdictA = '有小 K 的 CI 下界 > +0.05：K 值得改，需扩到 1000 图';
end
% 自检：K=200 臂的步数与 ms/步是否落在部署区间（1.179 ms/步、808 步的同机参考）
if ~isnan(meanMs(refIdx))
    fprintf('自检：K=200 臂 ms/步 %.2f（部署参考 1.18，本机当前负载可能偏高）、步数 %.0f（部署 808）\n', ...
            meanMs(refIdx), meanSteps(refIdx));
    fprintf(log_fid, '自检 K=200 ms/step %.2f steps %.0f\n', meanMs(refIdx), meanSteps(refIdx));
end
fprintf('判据 -> %s\n', verdictA);
fprintf(log_fid, '判据A -> %s\n', verdictA);

%% ---------- 实验 B：步数匹配 ----------
fprintf('\n== M6-B 步数匹配（都跑 808 步，200 图）==\n');
fprintf(log_fid, 'M6-B 步数匹配 808 步\n');
sB = cell(1, 2); gB = cell(1, 2); meanB = nan(1, 2); stepsB = nan(1, 2); msB = nan(1, 2);
kB = [20 200];
for ki = 1:2
    d = fullfile(ksweep, sprintf('steps808_k%d_n200', kB(ki)));
    files = dir(fullfile(d, '*.csv'));
    if isempty(files), fprintf('  缺 %s\n', d); continue; end
    T = v11_readtable_csv(fullfile(files(1).folder, files(1).name));
    if any(strcmp(T.Properties.VariableNames, 'repeat'))
        T = T(T.repeat == 0, :);
    end
    [g, keep] = unique(T.graph_id);
    sB{ki} = T.search_size(keep);
    gB{ki} = g;
    meanB(ki) = mean(T.search_size);
    stepsB(ki) = mean(T.steps);
    msB(ki) = mean(T.ms_per_step);
    fprintf('  K=%-4d search_size %8.4f  步数 %6.0f  ms/步 %6.3f  n=%d\n', ...
            kB(ki), meanB(ki), stepsB(ki), msB(ki), height(T));
    fprintf(log_fid, 'K=%d search_size %.4f steps %.0f ms/step %.3f\n', ...
            kB(ki), meanB(ki), stepsB(ki), msB(ki));
end
if ~any(isnan(meanB))
    d = pair_diff(sB{1}, gB{1}, sB{2}, gB{2});
    [lo, hi, mu] = boot_ci(d, 5000);
    fprintf('  K=20 - K=200（同步数 808）：%+0.4f  CI [%+0.4f, %+0.4f]\n', mu, lo, hi);
    fprintf('  解读：若这里差 ≈ 0 而时钟匹配差 ≠ 0，则差异来自「1 s 内能走多少步」，不是 K 本身\\n');
    fprintf(log_fid, 'steps-matched K20-K200 %+.4f CI [%+.4f,%+.4f]\n', mu, lo, hi);
    deltaSteps = meanSize(1) - meanSize(refIdx);      % 时钟匹配差（K20-K200）
    if deltaSteps < 0 && mu > -0.005
        verdictB = 'K 本身没有正贡献：1 s 内的差异全部来自步数（小 K 步多）';
    elseif deltaSteps < 0 && mu < -0.005
        verdictB = 'K 本身有负贡献（同步数下大 K 也更好）：小 K 只会更差';
    else
        verdictB = '见数字';
    end
    fprintf('  判据 -> %s\n', verdictB);
    fprintf(log_fid, '判据B -> %s\n', verdictB);
end

%% ---------- 实验 C：档案解耦（36 图轨迹）----------
fprintf('\n== M6-C 档案解耦（36 图轨迹，20 档案列口径）==\n');
Tb = v11_readtable_csv(fullfile(data, 'trace_best.csv'));
ids = unique(Tb.id);
fin = nan(numel(ids), 2);
for gi = 1:numel(ids)
    for ai = 1:2
        if ai == 1, key = 'K200'; else, key = 'K20'; end
        m = Tb.id == ids(gi) & strcmp(Tb.K, key);
        if any(m), fin(gi, ai) = max(Tb.best(m)); end
    end
end
ok = all(~isnan(fin), 2);
fprintf('  只看 20 个档案列时的 1.0 s 终点：K=200 %.4f，K=20 %.4f，差 %+0.4f（36 图）\n', ...
        mean(fin(ok, 1)), mean(fin(ok, 2)), mean(fin(ok, 2)) - mean(fin(ok, 1)));
fprintf(log_fid, 'trace 20 列口径：K200 %.4f K20 %.4f 差 %+.4f\n', ...
        mean(fin(ok, 1)), mean(fin(ok, 2)), mean(fin(ok, 2)) - mean(fin(ok, 1)));
fprintf('  解读：这个 +0.11 是「把 K=200 的 180 个档案列人为屏蔽掉」后的差，\n');
fprintf('        不等于部署口径（K=200 用全部 200 列）。所以它不能用来改部署 K。\n');
fprintf(log_fid, 'trace 口径 = 屏蔽 180 列，不等于部署\n');

%% 表
summary = table(Ks(:), meanSize(:), meanSteps(:), meanMs(:), ciLo(:), ciHi(:), ...
                'VariableNames', {'num_k', 'mean_size', 'mean_steps', 'ms_per_step', 'ci_lo_vs_k200', 'ci_hi_vs_k200'});
writetable(summary, fullfile(res, 'matlab_M6_summary.csv'));

%% 图（三栏：绝对水平 / 相对 K=200 的配对差 / 步数）
f = figure('Color', 'w', 'Position', [40 80 1400 400]);
okK = ~isnan(meanSize);

subplot(1, 3, 1);
hold on; box on; grid on;
plot(Ks(okK), meanSize(okK), '-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'Color', [0.15 0.35 0.75], 'MarkerFaceColor', [0.15 0.35 0.75]);
if any(~isnan(meanB))
    okB = ~isnan(meanB);
    plot(kB(okB), meanB(okB), 's--', 'LineWidth', 1.5, 'MarkerSize', 8, ...
         'Color', [0.85 0.40 0.10]);
end
set(gca, 'XScale', 'log');
xlim([16 250]);
yl = [min([meanSize(okK), meanB(~isnan(meanB))]) - 0.03, max([meanSize(okK), meanB(~isnan(meanB))]) + 0.03];
ylim(yl);
xlabel('num\_k'); ylabel('SIZE（200 图均值）');
title('时钟匹配 1.0 s（蓝）vs 步数匹配 810 步（橙）');
legend({'1.0 s 时钟匹配', '808 步匹配'}, 'Location', 'southeast', 'FontSize', 7);

subplot(1, 3, 2);
hold on; box on; grid on;
idxRel = find(~isnan(ciLo));
if ~isempty(idxRel)
    muRel = meanSize(idxRel) - meanSize(refIdx);      % 相对 K=200 的均值差（= CI 中心）
    err = [muRel - ciLo(idxRel); ciHi(idxRel) - muRel];
    errorbar(Ks(idxRel), muRel, err(1, :), err(2, :), 'o', ...
             'LineWidth', 1.5, 'MarkerSize', 7, 'Color', [0.55 0.20 0.55], ...
             'MarkerFaceColor', [0.55 0.20 0.55]);
    ylim([-0.30 0.10]);
end
yline(0, 'k--', 'LineWidth', 1.2);
yline(0.05, ':', 'Color', [0.5 0.5 0.5]);
set(gca, 'XScale', 'log');
xlim([16 250]);
xlabel('num\_k'); ylabel('mean SIZE − mean SIZE(K=200)');
title('相对 K=200 的配对 95% CI（点线 = +0.05 门槛）');
legend({'配对 CI', '0', '+0.05'}, 'Location', 'northeast', 'FontSize', 7);

subplot(1, 3, 3);
hold on; box on; grid on;
plot(Ks(okK), meanSteps(okK), '-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'Color', [0.20 0.60 0.35], 'MarkerFaceColor', [0.20 0.60 0.35]);
set(gca, 'XScale', 'log');
xlim([16 250]);
xlabel('num\_k'); ylabel('1.0 s 内的实际步数');
title(sprintf('步数随 K 下降：K=20 %.0f -> K=200 %.0f', meanSteps(1), meanSteps(end)));
exportgraphics(f, fullfile(figs, 'fig_M6_k_sweep.png'), 'Resolution', 150);
fprintf('\n图已写到 %s\n', figs);
fclose(log_fid);

%% ---------------- 局部函数 ----------------
function d = pair_diff(a, ga, b, gb)
% 按 graph_id 配对，返回 a - b（只保留两边都有的图）
ga = string(ga); gb = string(gb);
[dummy, ia, ib] = intersect(ga, gb); %#ok<ASGLU>
if isempty(ia)
    d = [];
else
    d = a(ia) - b(ib);
end
end

function [lo, hi, mu] = boot_ci(d, nboot)
d = d(:);
if isempty(d), lo = NaN; hi = NaN; mu = NaN; return; end
n = numel(d);
mu = mean(d);
rng(20261117, 'twister');
m = zeros(nboot, 1);
for b = 1:nboot
    m(b) = mean(d(randi(n, n, 1)));
end
m = sort(m);
lo = m(max(1, round(0.025 * nboot)));
hi = m(min(nboot, round(0.975 * nboot)));
end

function v = iff(c, a, b)
if c, v = a; else, v = b; end
end
