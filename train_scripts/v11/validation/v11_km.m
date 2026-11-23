function [tt, S, atRisk, d] = v11_km(dur, evt, tmax)
% V11_KM 离散时间 Kaplan-Meier 生存函数（base MATLAB，无工具箱）
%   dur  等待时长（正整数，单位 = block 数）
%   evt  1 = 观察到了事件（存档更新），0 = 右删失（跑完了还没再更新）
%   tmax 评估到 tmax
% 返回 tt (1..tmax)、S 生存概率 S(t)=P(W>=t+1)、atRisk、d（该时刻事件数）
dur = dur(:); evt = evt(:);
tt = 1:max(1, round(tmax));
S = ones(1, numel(tt));
atRisk = zeros(1, numel(tt));
d = zeros(1, numel(tt));
s = 1;
for k = 1:numel(tt)
    t = tt(k);
    nrisk = sum(dur >= t);
    ndie = sum(dur == t & evt == 1);
    atRisk(k) = nrisk;
    d(k) = ndie;
    if nrisk > 0
        s = s * (1 - ndie / nrisk);
    end
    S(k) = s;
end
end
