function ent = b_ent(p)

if p == 0
    ent = zeros(length(p),1);
elseif p == 1
    ent = zeros(length(p),1);
else
    ent = p.*log2(p) + (1-p).*log2(1-p);
    ent = -ent;
end 