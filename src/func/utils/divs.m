function div = divs(grad)

[n1,n2,n3,~] = size(grad);

shift = circshift(grad(:,:,:,1),[1,0,0,0]);
div1 = grad(:,:,:,1) - shift;
div1(1,:,:) = grad(1,:,:,1);
div1(n1,:,:) = -shift(n1,:,:);

shift = circshift(grad(:,:,:,2),[0,1,0,0]);
div2 = grad(:,:,:,2) - shift;
div2(:,1,:) = grad(:,1,:,2);
div2(:,n2,:) = -shift(:,n2,:);

shift = circshift(grad(:,:,:,3),[0,0,1,0]);
div3 = grad(:,:,:,3) - shift;
div3(:,:,1) = grad(:,:,1,3);
div3(:,:,n3) = -shift(:,:,n3);

div = div1 + div2 + div3;

end

