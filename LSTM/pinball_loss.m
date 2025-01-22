
function loss = pinball_loss(pred_upper, pred_lower, out, coverage, penalty)
% Pinball loss
tau1=0.5+0.5*coverage;
tau2=0.5-0.5*coverage;
l=penalty;%penalty

loss_upper = (max((out-pred_upper)*tau1 , (pred_upper-out)*(1-tau1)));
loss_lower = (max((out-pred_lower)*tau2 , (pred_lower-out)*(1-tau2)));
errors = pred_upper-pred_lower;
loss2 = l*(errors).^2*0.5;
loss = mean((loss_upper + loss_lower + loss2),"all");


end

