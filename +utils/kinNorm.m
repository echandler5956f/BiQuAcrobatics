function res = kinNorm(vec)
    if ~isnumeric(vec)
        flag = 0;
        for i = 1 : size(vec)
            if vec(i) ~= 0
                if flag == 1
                    res = vpa(sqrt(2))*-vec(i);
                    break;
                else
                    res = vpa(vec(i));
                    flag = 1;
                end
            end
        end
        if flag == 0
            res = 0;
        end
    else
        res = 0;
        for i = 1 : size(vec)
            res = res + vec(i)^2;
        end
        res = sqrt(res);
    end
end