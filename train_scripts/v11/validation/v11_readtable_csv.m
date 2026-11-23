function T = v11_readtable_csv(path)
%V11_READTABLE_CSV 读 CSV 成 table，绕开 readtable 的一个坑：
%   像 data/ladder.csv 的 level 列是 greedy / 0.05 / 0.1 ... 混排，readtable 会把
%   整列猜成 double，'greedy' 变成 NaN，于是 level=='greedy' 永远匹配不到。
%   这里用 readcell 逐格读原始值：全列都是数 → double；否则 → string（数字按 %g 转文本）。
%   返回值里数值列仍是 double，可以直接 mean/散点；文本列是 string，可 strcmp。
if ~isfile(path)
    error('v11:missingFile', '找不到数据文件：%s', path);
end
C = readcell(path, 'Delimiter', ',');
if isempty(C)
    T = table();
    return;
end
names = C(1, :);
body = C(2:end, :);
nv = numel(names);
cols = cell(1, nv);
for j = 1:nv
    col = body(:, j);
    isnum = cellfun(@isnumeric, col);
    if all(isnum)
        v = cell2mat(col);
        if isempty(v)
            v = zeros(0, 1);
        end
        cols{j} = v;
        continue;
    end
    if all(isnum | cellfun(@(x) (ischar(x) && isempty(x)) || (isa(x, 'missing')), col))
        v = nan(numel(col), 1);
        v(isnum) = cell2mat(col(isnum));
        cols{j} = v;
        continue;
    end
    s = strings(numel(col), 1);
    for i = 1:numel(col)
        x = col{i};
        if isnumeric(x) && ~isempty(x) && ~any(isnan(x))
            s(i) = string(num2str(x, '%g'));
        elseif ischar(x)
            s(i) = string(x);
        elseif isstring(x)
            s(i) = x;
        else
            s(i) = missing;
        end
    end
    cols{j} = s;
end
T = table();
for j = 1:nv
    nm = names{j};
    if ~ischar(nm), nm = char(string(nm)); end
    nm = matlab.lang.makeValidName(nm);
    T.(nm) = cols{j};
end
end
