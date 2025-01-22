function out = BFR(y,ypred)

out=100*(1- (norm(ypred-y, 2))/(norm(y-mean(y),2)));

end