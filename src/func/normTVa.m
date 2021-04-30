function norm = normTVa(x)
% *************************************************************************
% * This function calculates the anisotropic TV norm of an input array x. 
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
%            The calculated anisotropic TV norm of x.
%
% *************************************************************************

grad = grads(x);
norm = sum(sum(abs(grad),4),[1,2,3]);

end

