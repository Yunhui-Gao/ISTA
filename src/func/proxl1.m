function prox = proxl1(x,gamma)
% *************************************************************************
% * This function calculates the proximity operator of x with respect to
%   the l1 norm, which is also known as the soft-thresholding operator:
%
%           min { || y ||_1 + 1 / (2*gamma) *  || y - x ||_2^2 }.
%            y
%
% *************************************************************************
% * Author : Yunhui Gao
% * Date   : 2021/04/20
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- x     : 1D / 2D / 3D array
%             The input array.
%
%   - gamma : float
%             Parameter for the proximity operator.
%
%   ===== Outputs =========================================================
%
%   - prox : 1D / 2D / 3D array
%            The proximity operator for x with respect to the l1 norm.
%
% *************************************************************************

prox = max(0,x-gamma) + min(0,x+gamma);

end

