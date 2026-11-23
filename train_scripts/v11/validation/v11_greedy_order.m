function order = v11_greedy_order(adj)
%V11_GREEDY_ORDER PAVE 的 ``extend`` 扫描顺序（models/mis_v2_exchange.py:27）：
%   度数升序，并列按下标升序。每张图算一次，之后所有补满/重建都复用它。
n = size(adj, 1);
deg = sum(adj, 2);
[~, order] = sortrows([deg, (1:n)']);
order = order(:)';
end
