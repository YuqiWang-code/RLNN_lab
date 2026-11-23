function [S, added] = v11_greedy_fill(adj, S, order, forbidden)
%V11_GREEDY_FILL 贪心补满，等价于 PAVE 的 ``extend`` 算子（models/mis_v2_exchange.py:37）：
%   按固定顺序 order（v11_greedy_order，度数升序）扫一遍补集，能加就加。
%   forbidden（可选）= tabu，禁止把破坏掉的点原样放回。
n = numel(S);
S = logical(S(:)');
if nargin < 4 || isempty(forbidden)
    forbidden = false(1, n);
else
    forbidden = logical(forbidden(:)');
end
added = 0;
for v = order
    if S(v) || forbidden(v)
        continue;
    end
    if ~any(adj(v, S))
        S(v) = true;
        added = added + 1;
    end
end
end
