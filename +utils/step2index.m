%% Helper function for converting a step in a linear vector to an index
function index = step2index(vec,step)
    rows = size(vec,1);
    cols = size(vec,2);
    if xor(rows == 1, cols == 1)
        if rows >= cols
            if vec(end,1)-vec(1,1) == 0
                index = 1;
            else
                if step == 0
                    index = 1;
                else
                    if step > 0
                        index = floor((rows-1)*step/(vec(end,1)-vec(1,1)));
                    else
                        index = floor((rows-1)*-step/(vec(1,1)-vec(end,1)));
                    end
                end
            end
        else
            if vec(1,end)-vec(1,1) == 0
                index = 1;
            else
                if step == 0
                    index = 1;
                else 
                    if step > 0
                        index = floor((cols-1)*step/(vec(1,end)-vec(1,1)));
                    else
                        index = floor((cols-1)*-step/(vec(1,1)-vec(1,end)));
                    end
                end
            end
        end
    else
        error('Cannot compute index. Array must be a row or column vector.')
    end
end