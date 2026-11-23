function [gain, best, status, visits] = v11_exact_move(adj, S, k, cap)
%V11_EXACT_MOVE 精确判定「半径 k 的改进移动是否存在」（只含加入，删除取最少）。
%
%   一个 k-移动 = 从补集里挑 k 个点 A 加入（A 内部必须独立），并把与 A 相邻的
%   S 点全部删掉（删掉的数量是 N(A)∩S，取最少即最优）。净增益
%       gain = k - |N(A) ∩ S|          （>0 才算改进）
%   因此「存在改进的 k-移动」等价于「存在独立集 A ⊆ V\\S，|A| = k，|N(A)∩S| ≤ k-1」。
%
%   这是精确判定（不是启发式）：任何参与 A 的点都必须满足 c(v)=|N(v)∩S| ≤ k-1，
%   所以候选集限制在 c ≤ k-1 不丢解；再按冲突预算 b = 0..k-1 递增做分支定界，
%   第一个成功的 b 就给出最大增益 k-b。
%
%   cap 是可选的搜索节点上限（默认 4e5），超限时 status='capped'。
%   返回：gain = 最大净增益（0 表示不存在改进移动）；best = 对应的加入点下标。
if nargin < 4 || isempty(cap)
    cap = 4e5;
end
n = numel(S);
S = logical(S(:)');
conflict = (adj * double(S(:)))';      % c(v) = 与 S 的邻居数（行向量，避免 n×1 与 1×n 隐式扩展）
gain = 0;
best = [];
status = 'none';
visits = 0;
for budget = 0:k - 1
    cand = find(~S & conflict <= budget);
    if numel(cand) < k
        continue;
    end
    % 冲突少、下标小的先试，尽早命中
    [~, order] = sortrows([conflict(cand), cand]);
    cand = cand(order);
    [found, picked, used, capped] = search(cand, budget);
    visits = visits + used;
    if found
        gain = k - budget;
        best = picked;
        status = 'found';
        return;
    end
    if capped
        status = 'capped';
        return;
    end
end

    function [found, picked, used, capped] = search(cand, budget)
        found = false;
        picked = zeros(1, k);
        used = 0;
        capped = false;
        depth = 0;
        chosen = zeros(1, k);
        marked = false(1, n);      % 已被 N(A)∩S 覆盖的点
        covered = 0;
        dfs(1);

        function dfs(pos)
            if found || capped
                return;
            end
            used = used + 1;
            if used > cap
                capped = true;
                return;
            end
            if depth == k
                found = true;
                picked = chosen;
                return;
            end
            left = k - depth;
            for t = pos:numel(cand)
                if numel(cand) - t + 1 < left
                    return;
                end
                v = cand(t);
                if any(adj(v, chosen(1:depth)))
                    continue;
                end
                % 新覆盖的冲突点
                fresh = adj(v, :) & S & ~marked;
                fresh_count = nnz(fresh);
                if covered + fresh_count > budget
                    continue;
                end
                marked(fresh) = true;
                covered = covered + fresh_count;
                depth = depth + 1;
                chosen(depth) = v;
                dfs(t + 1);
                if found || capped
                    return;
                end
                depth = depth - 1;
                marked(fresh) = false;
                covered = covered - fresh_count;
            end
        end
    end
end
