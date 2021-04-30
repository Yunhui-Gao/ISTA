function norm = normTVi(x)

grad = grads(x);
norm = sum(sqrt(sum(grad.^2,4)),[1,2,3]);

end

