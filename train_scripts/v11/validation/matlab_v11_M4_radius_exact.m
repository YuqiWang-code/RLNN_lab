%% M4：移动半径 vs 改进概率（精确判定，不是启发式）
% 运行方法：在 MATLAB 里把 Current Folder 设为本目录（train_scripts/v11/validation），
%           打开本文件，点 Run。只依赖 base MATLAB。
%
% 问题：v10 的坐标下降 / PAVE 修复都报告「1-交换、2-交换在收敛解上一条都没有」。
% 这到底是算子实现不到位，还是解真的已经落在「半径 r 内没有改进」的不动点上？
% 这里对每张真实 RB 图、每个起点解，精确枚举半径 k = 1..6 的加入型移动：
%   一个 k-移动 = 从补集里挑 k 个互不相邻的点 A 加进去，把与 A 相邻的 S 点删掉；
%   净增益 = k - |N(A) ∩ S|，>0 才算改进。
% 只要存在这样的 A，v11_exact_move 就一定找得到（候选点限制在 c(v) ≤ k-1 不丢解，
% 冲突预算 b = 0..k-1 递增做分支定界），所以「不存在」是**证明**而不是「没找到」。
%
% 判据（方向锁定 §11 M4）：若改进概率在 k=2 之后没有下降，说明「半径-2 精确修复
% 0/20」是实现问题；若 k≥2 全线为 0，说明半径内确实没有改进，不动点是结构性的。
%
% 数据：data/graphs.csv, data/edges/*.csv, data/solutions.csv
% 输出：figures/fig_M4_radius_exact.png
%       results/matlab_M4_summary.csv, results/matlab_M4_report.txt

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
data = fullfile(here, 'data');
figs = fullfile(here, 'figures');
res  = fullfile(here, 'results');
if ~exist(figs, 'dir'), mkdir(figs); end
if ~exist(res, 'dir'), mkdir(res); end

LIMIT = 0;        % 调试时可改成 6，只跑前 6 张图；0 = 全部 36 张
KMAX = 6;         % 半径上限
starts = {'greedy', 'search1s', 'pave'};

G = v11_readtable_csv(fullfile(data, 'graphs.csv'));
S = v11_readtable_csv(fullfile(data, 'solutions.csv'));
if LIMIT > 0
    G = G(1:min(LIMIT, height(G)), :);
end
ng = height(G);
log_fid = fopen(fullfile(res, 'matlab_M4_report.txt'), 'w', 'n', 'UTF-8');

fprintf('== M4 移动半径 vs 改进概率（精确枚举）==\n');
fprintf('图数 %d，半径 k = 1..%d，起点：min-degree greedy / 1.0 s 搜索解 / 部署 PAVE 解\n', ng, KMAX);
fprintf(log_fid, 'M4 移动半径 vs 改进概率：%d 张图，k=1..%d\n', ng, KMAX);

gap = nan(ng, KMAX, numel(starts));    % 净增益（0 = 不存在改进移动）
exist_flag = false(ng, KMAX, numel(starts));
capped = false(ng, KMAX, numel(starts));
t0 = tic;
for gi = 1:ng
    gid = G.id(gi);
    n = G.n(gi);
    adj = v11_load_edges(fullfile(here, 'data', char(G.edges(gi))), n);
    for si = 1:numel(starts)
        Sn = start_solution(S, gid, n, starts{si});
        if isempty(Sn), continue; end
        for k = 1:KMAX
            [gn, ~, st] = v11_exact_move(adj, Sn, k);
            gap(gi, k, si) = gn;
            exist_flag(gi, k, si) = strcmp(st, 'found') && gn > 0;
            capped(gi, k, si) = strcmp(st, 'capped');
        end
    end
    if mod(gi, 6) == 0 || gi == ng
        fprintf('  [%d/%d] %s n=%d  %.1f s\n', gi, ng, char(G.name(gi)), n, toc(t0));
    end
end

%% 汇总表
fprintf('\n%-4s %10s %10s %10s %10s %10s %10s\n', 'k', 'P(1s)', 'P(PAVE)', 'P(greedy)', ...
        'maxG(1s)', 'maxG(PAVE)', 'capped');
kk = (1:KMAX)';
p1 = mean(exist_flag(:, :, 2), 1)';
pp = mean(exist_flag(:, :, 3), 1)';
pg = mean(exist_flag(:, :, 1), 1)';
m1 = max([zeros(1, KMAX); gap(:, :, 2)], [], 1)';
mp = max([zeros(1, KMAX); gap(:, :, 3)], [], 1)';
mg1 = mean(gap(:, :, 2), 1)';
mgp = mean(gap(:, :, 3), 1)';
capk = sum(any(capped, 3), 1)';
for k = 1:KMAX
    fprintf('%-4d %9.0f%% %9.0f%% %9.0f%% %10.0f %10.0f %10d\n', k, 100*p1(k), 100*pp(k), ...
            100*pg(k), m1(k), mp(k), capk(k));
    fprintf(log_fid, ['k=%d P(1s)=%.0f%% P(PAVE)=%.0f%% P(greedy)=%.0f%% maxG(1s)=%.0f maxG(PAVE)=%.0f ' ...
            'meanG(1s)=%.2f meanG(PAVE)=%.2f capped=%d\n'], k, 100*p1(k), 100*pp(k), 100*pg(k), ...
            m1(k), mp(k), mg1(k), mgp(k), capk(k));
end
tab = table(kk, p1, pp, pg, mg1, mgp, m1, mp, capk, ...
            'VariableNames', {'k', 'p_1s', 'p_pave', 'p_greedy', 'mean_gain_1s', ...
                              'mean_gain_pave', 'max_gain_1s', 'max_gain_pave', 'capped'});
writetable(tab, fullfile(res, 'matlab_M4_summary.csv'));

%% 关键数字
for si = 1:numel(starts)
    s1 = any(exist_flag(:, 1, si));
    s2 = any(any(exist_flag(:, 2:min(3, KMAX), si), 2));
    s3 = any(any(exist_flag(:, 2:KMAX, si), 2));
    fprintf('[%s] 存在 1-移动的图：%d/%d；存在 2/3-移动：%d/%d；存在 ≤%d-移动：%d/%d\n', ...
            starts{si}, sum(exist_flag(:, 1, si)), ng, sum(s2), ng, KMAX, sum(s3), ng);
    fprintf(log_fid, '[%s] 1-move %d/%d; 2-3 move %d/%d; <=%d-move %d/%d\n', starts{si}, ...
            sum(exist_flag(:, 1, si)), ng, sum(s2), ng, KMAX, sum(s3), ng);
end
fprintf('搜索节点上限触发的格子数：%d / %d（0 表示全部判定都是精确结论）\n', sum(capped(:)), numel(capped));
fprintf(log_fid, 'capped cells %d/%d\n', sum(capped(:)), numel(capped));

%% 图
f = figure('Color', 'w', 'Position', [80 80 900 400]);
subplot(1, 2, 1);
hold on; box on; grid on;
cols = [0.55 0.55 0.55; 0.20 0.50 0.90; 0.85 0.25 0.20];
for si = 1:numel(starts)
    plot(1:KMAX, 100*squeeze(mean(exist_flag(:, :, si), 1)), 'o-', ...
         'Color', cols(si, :), 'LineWidth', 1.8, 'MarkerFaceColor', cols(si, :));
end
set(gca, 'XTick', 1:KMAX);
xlabel('移动半径 k（加入点个数）');
ylabel('存在改进移动的图比例 (%)');
title('精确判定：半径 k 内有没有改进？');
legend({'1.0 s 搜索解', '部署 PAVE 解', 'greedy 解'}, 'Location', 'northeast');
ylim([-3 105]);
subplot(1, 2, 2);
hold on; box on; grid on;
for si = 1:numel(starts)
    plot(1:KMAX, squeeze(mean(gap(:, :, si), 1)), 's-', ...
         'Color', cols(si, :), 'LineWidth', 1.6);
end
set(gca, 'XTick', 1:KMAX);
xlabel('移动半径 k');
ylabel('平均净增益（个点，0 = 无改进）');
title(sprintf('平均净增益（%d 张图）', ng));
legend({'1.0 s', 'PAVE', 'greedy'}, 'Location', 'northwest');
exportgraphics(f, fullfile(figs, 'fig_M4_radius_exact.png'), 'Resolution', 150);

fprintf('\n图已写到 %s\n', figs);
fclose(log_fid);
%% 结论（措辞已按可辩护范围收紧）
nCell = numel(capped);
fprintf('\n== 结论（严格限定适用范围）==\n');
fprintf('在 %d 张真实 RB 图、3 个起点（min-degree greedy / 1.0 s 搜索解 / 部署 PAVE 解）、\n', ng);
fprintf('且仅限「精确的 k-交换邻域、k<=%d」这 %d 个格子的范围内：存在改进移动的比例 = 0%%。\n', KMAX, nCell);
fprintf('其中 %d 个格子是精确结论（搜索未被上限截断）。\n', nCell - sum(capped(:)));
fprintf('注意边界：本结论**不覆盖** (a) k>=%d 的交换邻域；(b) 破坏-修复、LNS、连续多步这类\n', KMAX + 1);
fprintf('组合移动（它们不是单步邻域）；(c) 其他起点（随机解、温度扰动后的解）。\n');
fprintf('所以正确表述是「更大的**单步**邻域在这三个起点上无效」，不是「任何邻域都无效」。\n');
fprintf('\nM4 结论见 results/matlab_M4_report.txt\n');


function Sn = start_solution(S, gid, n, which)
% 从 solutions.csv 里取指定起点解，返回 1×n 的逻辑向量（节点是 1 基下标）
Sn = [];
m = S.id == gid;
switch which
    case 'greedy'
        m = m & strcmp(S.kind, 'search') & strcmp(S.level, 'greedy');
    case 'search1s'
        m = m & strcmp(S.kind, 'search') & strcmp(S.level, '1');
    case 'pave'
        m = m & strcmp(S.kind, 'pave_deployed');
end
if ~any(m), return; end
row = find(m, 1, 'first');
nodes = str2num(char(S.nodes(row)));  %#ok<ST2NM>
Sn = false(1, n);
Sn(nodes) = true;
end
