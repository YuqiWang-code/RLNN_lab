%% M5：1 秒预算的两段式分配——oracle 上界到底有多大
% 运行方法：在 MATLAB 里把 Current Folder 设为本目录（train_scripts/v11/validation），
%           打开本文件，点 Run。只依赖 base MATLAB。
%
% 问题（方向锁定 §11 M5 / C1 保底线）：1 s 里如果可以「先跑 K=200 窄搜索 t 秒，
% 再跑 K=20 宽搜索 (1-t) 秒」，最优切换点 t* 在哪、能多赚多少？
% 用真实轨迹给**上界**：两条臂都在真实 RB 图上跑过，每 5 个 block 记一次 20 粒子
% 档案的最好解（data/trace_best.csv），于是
%     portfolio(t) = max( s200(t), s20(1-t) )   ← 两臂各 20 粒子，取更好的那个
%     sum(t)       =      s200(t) + s20(1-t)    ← 方向锁定原文的「求和」口径（不可实现，仅参考）
% oracle 上界 = 每张图各自挑最好的 t*（事后诸葛亮）→ 任何固定 t* 策略的天花板。
%
% 口径声明：轨迹只导出 20 个固定粒子列，所以这里的基线是「20 粒子档案」口径，
% 不是部署的 top-4 + PAVE 那条线（那条在 data/deployed_k200_1000.csv，见 M3）。
% 两段式是**同口径配对**比较，只用来判断「1 s 分配还有没有空间」。
%
% 数据：data/trace_best.csv（id,K,rec,steps,t,best），K ∈ {K200, K20}
% 输出：figures/fig_M5_two_phase_budget.png
%       results/matlab_M5_curve.csv, results/matlab_M5_summary.csv, results/matlab_M5_report.txt

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
data = fullfile(here, 'data');
figs = fullfile(here, 'figures');
res  = fullfile(here, 'results');
if ~exist(figs, 'dir'), mkdir(figs); end
if ~exist(res, 'dir'), mkdir(res); end

T = v11_readtable_csv(fullfile(data, 'trace_best.csv'));
ids = unique(T.id);
ng = numel(ids);
GRID = 0:0.01:1.0;
ngr = numel(GRID);
log_fid = fopen(fullfile(res, 'matlab_M5_report.txt'), 'w', 'n', 'UTF-8');

S200 = nan(ng, ngr);
S20 = nan(ng, ngr);
for gi = 1:ng
    gid = ids(gi);
    for arm = 1:2
        if arm == 1, key = 'K200'; else, key = 'K20'; end
        m = T.id == gid & strcmp(T.K, key);
        if ~any(m), continue; end
        t = T.t(m); b = T.best(m);
        [t, ord] = sort(t); b = b(ord);
        [t, iu] = unique(t, 'last'); b = b(iu);     % interp1 要求严格递增
        v = interp1(t, b, GRID, 'previous', 'extrap');
        v(isnan(v)) = b(1);
        v(GRID < t(1)) = b(1);                       % t* 早于首条记录 → 用首值
        if arm == 1, S200(gi, :) = v; else, S20(gi, :) = v; end
    end
end

fprintf('== M5 两段式预算 oracle 上界（20 粒子档案口径）==\n');
fprintf('图数 %d，切换点网格 %d 个（0..1 s，步长 0.01 s）\n', ng, ngr);

%% 单臂基线
base200 = mean(S200(:, end));
base20 = mean(S20(:, end));
best_single = max(base200, base20);
fprintf('单臂跑满 1 s：K=200 %.4f；K=20 %.4f（K=20 更强，差 %+.4f）\n', base200, base20, base20 - base200);
fprintf(log_fid, 'M5 两段式预算（20 粒子档案口径，%d 图）\nK200(1s)=%.4f K20(1s)=%.4f\n', ng, base200, base20);

%% 组合曲线
port = nan(ng, ngr);
sumc = nan(ng, ngr);
for i = 1:ngr
    port(:, i) = max(S200(:, i), S20(:, ngr + 1 - i));
    sumc(:, i) = S200(:, i) + S20(:, ngr + 1 - i);
end
mean_port = mean(port, 1);
mean_sum = mean(sumc, 1);
[best_port, iport] = max(mean_port);
t_star = GRID(iport);

%% 逐图 oracle
[oracle_per_graph, ior] = max(port, [], 2);
oracle = mean(oracle_per_graph);
t_oracle = GRID(ior);
n_boundary = sum(t_oracle <= 1e-9 | t_oracle >= 1 - 1e-9);
per_graph_single = max(S200(:, end), S20(:, end));   % 每张图挑更强的单臂（需要预知答案）
d_oracle_vs_single = oracle_per_graph - per_graph_single;

fprintf('\n全局最优固定 t* = %.2f s：组合均值 %.4f\n', t_star, best_port);
fprintf('逐图 oracle 上界 = %.4f；其中 %d/%d 张图的最优 t* 落在边界（t*=0 或 t*=1，即整秒跑单臂）\n', ...
        oracle, n_boundary, ng);
fprintf('逐图 oracle 相对「每张图预知更强单臂」的增量 = %+.4f（max %+.0f，仅 %d 张图非零）\n', ...
        mean(d_oracle_vs_single), max(d_oracle_vs_single), sum(d_oracle_vs_single ~= 0));
fprintf('原文「求和」口径：max_t mean(sum) = %.4f at t*=%.2f s（两个 arm 相加，物理上不可实现，仅参考）\n', ...
        max(mean_sum), GRID(find(mean_sum == max(mean_sum), 1)));
fprintf(log_fid, '全局固定 t*=%.2f 组合=%.4f\n逐图 oracle=%.4f（边界解 %d/%d 图）\n', ...
        t_star, best_port, oracle, n_boundary, ng);
fprintf(log_fid, 'oracle 相对逐图更强单臂 %+.4f（非零 %d 图）\n', mean(d_oracle_vs_single), sum(d_oracle_vs_single ~= 0));
fprintf(log_fid, 'sum 口径 max=%.4f at t*=%.2f（不可实现）\n', max(mean_sum), GRID(find(mean_sum == max(mean_sum), 1)));

%% 判据
% 决定性量是「切换点本身的收益」：oracle 相对「每张图预知更强单臂」的增量。
% 而 gain_over_single 含 Jensen 效应（逐图 max 的均值 ≥ 两臂均值的 max），
% 那一部分只能靠「逐图选臂」拿到，属于选择器问题，不是预算切分问题。
gain_over_single = best_port - best_single;
switch_gain = mean(d_oracle_vs_single);
if switch_gain <= 0.005
    verdict = '没有空间：切换点本身零收益，两段式退化为整秒跑单臂';
elseif switch_gain < 0.015
    verdict = '弱：切换收益不到 +0.015（方向锁定下界）';
else
    verdict = '有空间：需要再用多种子确认';
end
fprintf('\n组合 vs 最强单臂 %.4f：%+.4f（其中切换点收益 %+.4f，其余是逐图选臂的 Jensen 效应）\n', ...
        best_single, gain_over_single, switch_gain);
fprintf('方向锁定预期 t*∈[0.55,0.70] 且 +0.015~+0.025；实测 t*=%.2f ⇒ 结论：%s\n', t_star, verdict);
fprintf('注意：与 K=200 单臂比会看到 %+.4f，那不是两段式收益，而是 K=20 臂本身更强。\n', best_port - base200);
fprintf(log_fid, '组合 vs 最强单臂 %.4f: %+.4f（切换点收益 %+.4f）；实测 t*=%.2f -> %s\n', ...
        best_single, gain_over_single, switch_gain, t_star, verdict);

%% 表
writetable(table(GRID(:), mean_port(:), mean_sum(:), 'VariableNames', {'t_star', 'portfolio_mean', 'sum_mean'}), ...
           fullfile(res, 'matlab_M5_curve.csv'));
writetable(table(ids(:), oracle_per_graph(:), t_oracle(:), per_graph_single(:), ...
                 'VariableNames', {'id', 'oracle_portfolio', 't_oracle', 'best_single_arm'}), ...
           fullfile(res, 'matlab_M5_summary.csv'));

%% 图
f = figure('Color', 'w', 'Position', [80 80 980 400]);
subplot(1, 2, 1);
hold on; box on; grid on;
plot(GRID, mean_port, '-', 'LineWidth', 2, 'Color', [0.20 0.50 0.90]);
plot(GRID, mean_sum - mean(mean_sum) + mean(best_port), '--', 'LineWidth', 1.2, 'Color', [0.70 0.70 0.70]);
yline(base200, 'k-.', 'LineWidth', 1.2);
yline(base20, 'k:', 'LineWidth', 1.2);
xline(t_star, 'k:');
xlabel('切换点 t^*（先 K=200 用 t^* 秒，再 K=20 用 1-t^* 秒）');
ylabel('20 粒子档案最好解（图均值）');
title(sprintf('两段式：t^*=%.2f 达 %.4f（最强单臂 %.4f，%+.4f）', t_star, best_port, best_single, gain_over_single));
legend({'portfolio max(s_{200},s_{20})', '求和口径（已平移，参考）', 'K=200 满 1 s', 'K=20 满 1 s', 't^*'}, ...
       'Location', 'southwest', 'FontSize', 7);
subplot(1, 2, 2);
hold on; box on; grid on;
histogram(t_oracle * 100, -2.5:5:102.5, 'FaceColor', [0.35 0.55 0.85]);
xlabel('逐图最优 t^* 的位置（秒 ×100）');
ylabel('图数');
title(sprintf('逐图 oracle 切换点：%d/%d 落在边界', n_boundary, ng));
exportgraphics(f, fullfile(figs, 'fig_M5_two_phase_budget.png'), 'Resolution', 150);

fprintf('\n图已写到 %s\n', figs);
fclose(log_fid);
