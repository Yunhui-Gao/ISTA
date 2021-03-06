function norm = norml1(x)
% *************************************************************************
% * This function calculates the l1 norm of an input array x. 
% *************************************************************************
%
%   ===== Required inputs =================================================
%
%	- x     : 1D / 2D / 3D array
%             The input array.
%
%   ===== Outputs =========================================================
%
%   - val :  float
%            The calculated l1 norm of x.
%
% *************************************************************************

norm = sum(abs(x),[1,2,3]);

end

