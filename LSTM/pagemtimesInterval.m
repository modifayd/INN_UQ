function [zu,zl] = pagemtimesInterval(x_lower,x_upper,y_lower,y_upper)

[mx, nx, p]=size(x_lower);
[ny, py, q]=size(y_lower);

if ny==nx

    xlower1 = reshape(x_lower, mx, nx, 1, size(x_lower,3));
    ylower1 = reshape(y_lower, ny, 1, py, size(x_lower,3));
    ylower2 = permute(ylower1,[2,1,3,4]);

    xupper1 = reshape(x_upper, mx, nx, 1, size(x_upper,3));
    yupper1 = reshape(y_upper, ny, 1, py, size(x_upper,3));
    yupper2 = permute(yupper1,[2,1,3,4]);
   
    % c1 = bsxfun(@times,xlower1,ylower2);
    % c2 = bsxfun(@times,xlower1,yupper2);
    % c3 = bsxfun(@times,xupper1,ylower2);
    % c4 = bsxfun(@times,xupper1,yupper2);

    c1 = times(xlower1,ylower2);
    c2 = times(xlower1,yupper2);
    c3 = times(xupper1,ylower2);
    c4 = times(xupper1,yupper2);

    
    c_upper = max(max(c1,c2),max(c3,c4));
    c_lower = min(min(c1,c2),min(c3,c4));

    zu = reshape(sum(c_upper,2),mx,py,size(x_lower,3));
    zl = reshape(sum(c_lower,2),mx,py,size(x_lower,3));

else

    error('Pagetimes Error: pages of "x" and "y" are not matched!')

end

end








