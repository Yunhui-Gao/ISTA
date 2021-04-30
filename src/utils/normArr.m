function val = normArr(x)

% *************************************************************************
% * This function calculates the l2 norm of a multi-dimensional array.
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- x   : multi-dimensional array
%           The input array.
%
%   ===== Outputs =========================================================
%
%   - val : float
%           The calculated norm.
%
% *************************************************************************

val = sqrt(dot(x(:),x(:)));

end

