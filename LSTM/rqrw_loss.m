
function loss = rqrw_loss(pred_upper, pred_lower, out, coverage, penalty)
%RQR-W loss
c=coverage;%coverage
l=penalty;%penalty
error1 = out-pred_lower;
error2 = out-pred_upper;
errors = pred_upper-pred_lower;
loss1 = max(error1.*error2.*(c+2*l),error2.*error1.*(c+2*l-1));
loss2 = l*(errors).^2*0.5;
loss = mean(loss1+loss2,"all");
end



