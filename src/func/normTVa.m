function norm = normTVa(x)

grad = grads(x);
norm = sum(sum(abs(grad),4),[1,2,3]);

end

