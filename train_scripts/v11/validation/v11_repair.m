function [S, rounds_used] = v11_repair(adj, S, order, rounds, maxk)
%V11_REPAIR PAVE 修复算子的 MATLAB 复刻：先 extend（贪心补满），再做至多 rounds 次
%   「改进交换」。改进交换用 v11_exact_move 精确枚举：
%       k=2 → 相当于 PAVE 的 1-removal 交换（删 1 个、加 ≥2 个）
%       k=3 → 相当于 2-removal 交换（删 2 个、加 ≥3 个）
%   这正是 models/mis_v3_search.py:222 improving_actions 的两档动作集。
if nargin < 5 || isempty(maxk)
    maxk = 3;
end
S = v11_greedy_fill(adj, S, order);
rounds_used = 0;
for r = 1:rounds
    moved = false;
    for k = 2:maxk
        [gain, add, status] = v11_exact_move(adj, S, k);
        if strcmp(status, 'found') && gain > 0
            S(add) = true;
            covered = any(adj(add, :), 1);   % 新加入点的所有图邻居
            S(covered) = false;              % 删掉与它们冲突的 S 点（无自环，add 自身不受影响）
            rounds_used = rounds_used + 1;
            moved = true;
            break;
        end
    end
    if ~moved
        break;
    end
end
S = v11_greedy_fill(adj, S, order);
end
