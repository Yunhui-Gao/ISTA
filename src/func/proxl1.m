function prox = proxl1(x,gamma)

prox = max(0,x-gamma) + min(0,x+gamma);

end

