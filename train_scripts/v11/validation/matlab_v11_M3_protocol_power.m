%% M3：1000 图协议的功效——+0.010 / +0.020 / +0.0335 到底要多少张图才看得见？
% 运行方法：在 MATLAB 里把 Current Folder 设为本目录（train_scripts/v11/validation），
%           打开本文件，点 Run。只依赖 base MATLAB。
%
% 问题（方向锁定 §11 M3）：审稿人一定会问「+0.020 是不是噪声」。回答需要三样东西：
%   (1) 真实的**逐图配对噪声** σ_Δ（同一臂换种子、同一张图的差），这是「没有效应时
%       我们能测到的抖动」，只能用真实运行估，不能用公式假设；
%   (2) 给定 n 和 σ_Δ，小效应 Δ 的检验功效 power(n, Δ)；
%   (3) 反解：要在 80% 功效下看到 Δ，需要多少张图。
%
% 数据：data/protocol_1000.csv（1000 图 × 3 个 cell × 3 个种子，1.0 s 档：
%           captured/cached、captured/eager、eager/eager）
%       data/deployed_k200_1000.csv（部署口径 K=200 dense+fp16，1000 图，repeat 0/1）
% 方法：σ_Δ 用「同一 cell、同一张图、不同种子」的配对差估计（期望 0，只留噪声）；
%       功效用正态近似 power = Φ(Δ√n/σ_Δ − z_{1-α/2})，α=0.05 双侧，z=1.96；
%       需要的图数 n = (z_{1-α/2} + z_{power})² σ_Δ² / Δ²。
%
% 输出：figures/fig_M3_protocol_power.png
%       results/matlab_M3_report.txt, results/matlab_M3_power.csv

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
data = fullfile(here, 'data');
figs = fullfile(here, 'figures');
res  = fullfile(here, 'results');
if ~exist(figs, 'dir'), mkdir(figs); end
if ~exist(res, 'dir'), mkdir(res); end

P = v11_readtable_csv(fullfile(data, 'protocol_1000.csv'));
D = v11_readtable_csv(fullfile(data, 'deployed_k200_1000.csv'));
log_fid = fopen(fullfile(res, 'matlab_M3_report.txt'), 'w', 'n', 'UTF-8');
fprintf(log_fid, 'M3 1000 图协议功效分析\n');

%% 1) 三个 cell 的均值（复核 python 表）
cells = unique(P.cell);
fprintf('== 三个 cell 的 1.0 s 均值（1000 图 × 3 种子）==\n');
for ci = 1:numel(cells)
    m = strcmp(P.cell, cells{ci});
    fprintf('  %-18s mean %8.4f  逐图 sd %6.4f  n %d\n', ...
            cells{ci}, mean(P.final_size(m)), std(P.final_size(m)), sum(m));
    fprintf(log_fid, '%-18s mean %.4f sd %.4f n %d\n', ...
            cells{ci}, mean(P.final_size(m)), std(P.final_size(m)), sum(m));
end

%% 2) 逐图配对噪声 σ_Δ：同 cell、同图、不同种子
sigmas = [];
labels = {};
for ci = 1:numel(cells)
    m = strcmp(P.cell, cells{ci});
    ids = unique(P.graph_id(m));
    seeds = unique(P.seed(m));
    if numel(seeds) < 2, continue; end
    d = [];
    for ii = 1:numel(ids)
        mi = m & P.graph_id == ids(ii);
        vals = zeros(1, numel(seeds));
        ok = true;
        for si = 1:numel(seeds)
            v = P.final_size(mi & P.seed == seeds(si));
            if isempty(v), ok = false; break; end
            vals(si) = v(1);
        end
        if ~ok, continue; end
        d = [d; vals(1) - vals(2)];  %#ok<AGROW>
    end
    if isempty(d), continue; end
    sd_pair = std(d);
    sigmas(end + 1) = sd_pair;      %#ok<AGROW>
    labels{end + 1} = sprintf('%s 种子 %d-%d', cells{ci}, seeds(1), seeds(2)); %#ok<SAGROW>
    fprintf('配对噪声：%-34s mean %+0.4f  sd %6.4f  （n=%d 图）\n', ...
            labels{end}, mean(d), sd_pair, numel(d));
    fprintf(log_fid, '配对噪声 %s mean %+.4f sd %.4f n %d\n', ...
            labels{end}, mean(d), sd_pair, numel(d));
end
% 部署口径的 repeat 0/1 配对（同一 seed、同一图，重复跑）
m = true(height(D), 1);
idsD = unique(D.graph_id(m));
seedsD = unique(D.search_seed(m));
d = [];
for ii = 1:numel(idsD)
    mi = D.graph_id == idsD(ii);
    v = D.final_size(mi);
    if numel(v) >= 2
        d = [d; v(1) - v(2)]; %#ok<AGROW>
    end
end
if ~isempty(d)
    fprintf('配对噪声：部署 K=200 repeat 0-1            mean %+0.4f  sd %6.4f  （n=%d 图）\n', ...
            mean(d), std(d), numel(d));
    fprintf(log_fid, '配对噪声 部署 repeat0-1 mean %+.4f sd %.4f n %d\n', mean(d), std(d), numel(d));
    sigmas(end + 1) = std(d);      %#ok<AGROW>
    labels{end + 1} = '部署 K=200 repeat 0-1'; %#ok<SAGROW>
end
mD = D.search_seed == 1017 & D.repeat == 0;
fprintf('部署口径 K=200（seed 1017、repeat 0）1000 图均值 %.4f，逐图 sd %.4f，平均步数 %.0f\n', ...
        mean(D.final_size(mD)), std(D.final_size(mD)), mean(D.steps(mD)));
fprintf(log_fid, '部署 K=200 mean %.4f sd %.4f steps %.0f\n', ...
        mean(D.final_size(mD)), std(D.final_size(mD)), mean(D.steps(mD)));

%% 3) 三种噪声，别混用
% (a) 逐图 sd（σ_graph）：描述图的难度离散，不是效应估计的噪声；
% (b) 同种子配对噪声（σ_pair）：同一条轨迹下换臂的抖动，很小（≈0.03）；
% (c) **种子层面的均值噪声（σ_seedmean）**：同一臂换种子，1000 图均值会飘多少 ——
%     这才是「+0.020 是不是噪声」要用的量，因为论文主张是跨种子成立的。
% 本脚本用 3 cell × 3 seed 的 9 个均值，扣掉 cell 效应后合并估计 σ_seedmean。
cellMeans = nan(numel(cells), 3);
for ci = 1:numel(cells)
    m = strcmp(P.cell, cells{ci});
    ss = unique(P.seed(m));
    for si = 1:min(3, numel(ss))
        mm = m & P.seed == ss(si);
        cellMeans(ci, si) = mean(P.final_size(mm));
    end
end
dev = cellMeans - mean(cellMeans, 2);          % 扣掉每个 cell 的 3 种子均值
sigmaSeedMean = sqrt(sum(dev(:) .^ 2) / (numel(dev) - size(dev, 1)));
fprintf('\n== 三种噪声 ==\n');
fprintf('(a) 逐图 sd            %.4f（难度离散，不作效应噪声）\n', std(P.final_size));
fprintf('(b) 同种子配对噪声      %.4f（repeat 0-1，同一轨迹换臂）\n', min(sigmas));
fprintf('(c) 种子层面均值噪声    %.4f（同一臂换种子，1000 图均值的飘动量）\n', sigmaSeedMean);
fprintf(log_fid, 'sigma_graph %.4f sigma_pair %.4f sigma_seedmean %.4f\n', ...
        std(P.final_size), min(sigmas), sigmaSeedMean);

% 协议按方向锁定 = 3 种子合并 → 均值噪声再小 √3
nSeedsPool = 3;
sigmaEff = sigmaSeedMean / sqrt(nSeedsPool);
fprintf('3 种子合并后的检验噪声 σ_eff = %.4f\n', sigmaEff);
zA = 1.959964;
deltas = [0.005 0.010 0.020 0.0335 0.050];
rows = table();
fprintf('\n== 功效（α=0.05 双侧，3 种子合并）==\n');
for di = 1:numel(deltas)
    delta = deltas(di);
    pw = normcdf(delta / sigmaEff - zA);
    n80 = (zA + 0.8416) ^ 2 * sigmaSeedMean ^ 2 / delta ^ 2 * nSeedsPool;
    fprintf('Δ=%+.4f：power %.3f，80%% 功效需要 %d 张图 × 3 种子\n', delta, pw, ceil(n80));
    fprintf(log_fid, 'Δ %+.4f power %.3f n80 %d\n', delta, pw, ceil(n80));
    rows = [rows; table(delta, pw, ceil(n80), sigmaEff, sigmaSeedMean, ...
             'VariableNames', {'delta', 'power_3seed', 'n_for_80pct', 'sigma_eff', 'sigma_seedmean'})]; %#ok<AGROW>
end
deltaMin = (zA + 0.8416) * sigmaEff;
fprintf('n=1000×3 种子时 80%% 功效能检出的最小效应 = %+.4f\n', deltaMin);
deltaMin1 = (zA + 0.8416) * sigmaSeedMean;
fprintf('对照：只用 1 个种子（1000 图）时 = %+.4f（所以 +0.020 必须 3 种子合并才稳）\n', deltaMin1);
fprintf(log_fid, 'n1000x1 min detectable %+.4f\n', deltaMin1);
fprintf(log_fid, 'n1000x3 min detectable %+.4f\n', deltaMin);
writetable(rows, fullfile(res, 'matlab_M3_power.csv'));

%% 4) 判据
if deltaMin > 0.030
    verdict = '协议偏弱：连 +0.0335 都在边缘，需要多 seed 合并或扩图';
elseif deltaMin > 0.020
    verdict = '协议够用：+0.0335 可检出，+0.020 边缘，+0.010 在噪声带内';
else
    verdict = '协议很强：+0.020 可靠可检出';
end
fprintf('判据 -> %s\n', verdict);
fprintf(log_fid, '判据 -> %s\n', verdict);

%% 图
f = figure('Color', 'w', 'Position', [80 80 900 420]);
hold on; box on; grid on;
nGrid = [100 200 300 500 750 1000 1500 2000 3000 5000];
cols = lines(numel(deltas));
for di = 1:numel(deltas)
    pw = normcdf(deltas(di) * sqrt(nGrid / 1000) / sigmaEff - zA);
    plot(nGrid, pw, '-o', 'LineWidth', 1.6, 'MarkerSize', 4, 'Color', cols(di, :));
    text(nGrid(end), pw(end), sprintf(' %+.3f', deltas(di)), 'FontSize', 7, 'Color', cols(di, :));
end
yline(0.8, 'k--');
xline(1000, 'k:');
set(gca, 'XScale', 'log');
xlabel('图数 n（对数轴）');
ylabel('检验功效 power（α=0.05 双侧）');
title(sprintf('3 种子合并：n=1000 时最小可检出 %+.4f；+0.020 功效 %.2f', ...
      deltaMin, normcdf(0.020 / sigmaEff - zA)));
legend(arrayfun(@(d) sprintf('Δ=%+.4f', d), deltas, 'UniformOutput', false), ...
       'Location', 'southeast', 'FontSize', 7);
exportgraphics(f, fullfile(figs, 'fig_M3_protocol_power.png'), 'Resolution', 150);
fprintf('\n图已写到 %s\n', figs);
fclose(log_fid);
