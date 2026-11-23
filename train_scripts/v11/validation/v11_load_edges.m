function A = v11_load_edges(path, n)
%V11_LOAD_EDGES 读边表 CSV（u,v，1 基）成 n×n 逻辑邻接矩阵。
%   只用 base MATLAB：readmatrix + sub2ind。无向边按对称处理，自环丢弃。
if ~isfile(path)
    error('v11:missingEdgeFile', '找不到边表文件：%s', path);
end
M = readmatrix(path);
A = false(n, n);
if isempty(M)
    return;
end
u = M(:, 1);
v = M(:, 2);
keep = u ~= v;
u = u(keep);
v = v(keep);
lin = unique(sub2ind([n n], [u; v], [v; u]));
A(lin) = true;
end
