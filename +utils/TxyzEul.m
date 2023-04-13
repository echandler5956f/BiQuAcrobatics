function res = TxyzEul(t1, t2, t3)
    res = [1, 0,        sin(t2);
           0, cos(t1), -sin(t1)*cos(t2);
           0, sin(t1),  cos(t1)*cos(t2)];
end