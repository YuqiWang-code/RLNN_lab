function r = v11_pearson(x, y)
%V11_PEARSON 皮尔逊相关系数（base MATLAB，自己写，避免 corr 的工具箱依赖）。
x = x(:) - mean(x);
y = y(:) - mean(y);
den = sqrt(sum(x .^ 2) * sum(y .^ 2));
if den <= 0
    r = NaN;
else
    r = sum(x .* y) / den;
end
end
