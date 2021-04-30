function prox = proxTVi(x,gamma,iter)
% *************************************************************************
% * This function applies the fast projected gradient (FPG) algorithm to
%   the following denoising problem: 
%
%           min { || y ||_TVi + 1 / (2*gamma) *  || y - x ||_2^2 },
%            y
%   
%   where   || y ||_TVi stands for the isotropic TV norm of y.
% 
% * References:
%   [1] A. Beck and M. Teboulle, "Fast Gradient-Based Algorithms for 
%       Constrained Total Variation Image Denoising and Deblurring 
%       Problems," IEEE Transactions on Image Processing 18, 2419-2434 
%       (2009).
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
%   - iter  : int
%             Number of iterations for solving the denoising problem.
%
%   ===== Outputs =========================================================
%
%   - prox : 1D / 2D / 3D array
%            The proximity operator for x with respect to the isotropic
%            TV norm.
%
% *************************************************************************

t_prev = 1;

[n1,n2,n3] = size(x);
grad_next = zeros(n1,n2,n3,3);
grad_prev = zeros(n1,n2,n3,3);
temp = zeros(n1,n2,n3,3);
for i = 1:iter
    grad_next = temp + 1/8/gamma*grads(x - gamma*divs(temp));
    deno = zeros(n1,n2,n3,3);
    deno(:,:,:,1) = max(1,sqrt(grad_next(:,:,:,1).^2 + grad_next(:,:,:,2).^2 + grad_next(:,:,:,3).^2));
    deno(:,:,:,2) = deno(:,:,:,1);
    deno(:,:,:,3) = deno(:,:,:,1);
    grad_next = grad_next./deno;
    t_next = (1+sqrt(1+4*t_prev^2))/2;
    temp = grad_next + (t_prev-1)/t_next*(grad_next-grad_prev);
    grad_prev = grad_next;
    t_prev = t_next;
end

prox = x - gamma*divs(grad_next);

end

