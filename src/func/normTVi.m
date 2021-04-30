function val = normTVi(x)
% *************************************************************************
% * This function calculates the isotropic TV norm of an input array x. 
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
%            The calculated isotropic TV norm of x.
%
% *************************************************************************

grad = grads(x);
val = sum(sqrt(sum(grad.^2,4)),[1,2,3]);

end

