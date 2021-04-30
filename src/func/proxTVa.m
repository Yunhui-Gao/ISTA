function prox = proxTVa(x,gamma,iter)

t_prev = 1;
    
[n1,n2,n3] = size(x);
grad_next = zeros(n1,n2,n3,3);
grad_prev = zeros(n1,n2,n3,3);
temp = zeros(n1,n2,n3,3);

for i = 1:iter
    grad_next = temp + 1/8/gamma*grads(x - gamma*divs(temp));
    deno = zeros(n1,n2,n3,3);
    deno(:,:,:,1) = max(1,abs(grad_next(:,:,:,1)));
    deno(:,:,:,2) = max(1,abs(grad_next(:,:,:,2)));
    deno(:,:,:,3) = max(1,abs(grad_next(:,:,:,3)));
    grad_next = grad_next./deno;
    t_next = (1+sqrt(1+4*t_prev^2))/2;
    temp = grad_next + (t_prev-1)/t_next*(grad_next-grad_prev);
    grad_prev = grad_next;
    t_prev = t_next;
end  

prox = x - gamma*divs(grad_next);

end

