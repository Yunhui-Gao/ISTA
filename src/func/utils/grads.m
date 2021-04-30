function grad = grads(x)

[n1,n2,n3] = size(x);
grad = zeros(n1,n2,n3,3);
grad(:,:,:,1) = x - circshift(x,[-1,0,0]);
grad(n1,:,:,1) = 0;
grad(:,:,:,2) = x - circshift(x,[0,-1,0]);
grad(:,n2,:,2) = 0;
grad(:,:,:,3) = x - circshift(x,[0,0,-1]);
grad(:,:,n3,3) = 0;

end

