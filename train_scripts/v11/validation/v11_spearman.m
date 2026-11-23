function [rho, p_perm] = v11_spearman(x, y, nperm)
%V11_SPEARMAN 斯皮尔曼秩相关（不依赖 Statistics 工具箱）。
%   rho = 1 - 6*sum(d^2)/(n(n^2-1)) 只在无并列时成立，这里用「平均秩 + 皮尔逊」
%   的定义式；p 值用置换检验（默认 2000 次），因此也不需要 t 分布工具箱。
if nargin < 3 || isempty(nperm)
    nperm = 2000;
end
x = x(:);
y = y(:);
rx = v11_rank(x);
ry = v11_rank(y);
rho = v11_pearson(rx, ry);
if nargout < 2
    return;
end
rng(20261117, 'twister');   % 固定种子，结果可复现
count = 0;
for k = 1:nperm
    perm = randperm(numel(ry));
    if abs(v11_pearson(rx, ry(perm))) >= abs(rho) - 1e-12
        count = count + 1;
    end
end
p_perm = (count + 1) / (nperm + 1);
end
