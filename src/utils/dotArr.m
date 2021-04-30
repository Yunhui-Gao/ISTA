function val = dotArr(x, y)

% *************************************************************************
% * This function calculates the inner product (dot product) of two
%   multi-dimensional arrays.
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- x, y  : multi-dimensional array
%             The input arrays.
%
%   ===== Outputs =========================================================
%
%   - val : float
%           The calculated product.
%
% *************************************************************************

val = dot(x(:),y(:));

end

