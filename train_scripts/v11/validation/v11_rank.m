function r = v11_rank(x)
%V11_RANK 平均秩（并列取平均），等价于常用的 rankdata，不依赖工具箱。
x = x(:);
[~, order] = sort(x);
r = zeros(numel(x), 1);
sorted = x(order);
i = 1;
while i <= numel(x)
    j = i;
    while j < numel(x) && sorted(j + 1) == sorted(i)
        j = j + 1;
    end
    r(order(i:j)) = (i + j) / 2;
    i = j + 1;
end
end
