function loss = pinball_loss(pred_upper, pred_lower, out, coverage, penalty)

%pinball loss
tau1=0.5+0.5*coverage;
tau2=0.5-0.5*coverage;
l=penalty;%penalty

loss_upper1 = max( (out(:,:,2:end)-permute(pred_upper(:,:,:),[2 3 1])) *tau1 , (permute(pred_upper(:,:,:),[2 3 1])-out(:,:,2:end))*(1-tau1));
loss_lower1 = max( (out(:,:,2:end)-permute(pred_lower(:,:,:),[2 3 1])) *tau2 , (permute(pred_lower(:,:,:),[2 3 1])-out(:,:,2:end))*(1-tau2));
errors = permute(pred_upper(:,:,:),[2 3 1])-permute(pred_lower(:,:,:),[2 3 1]);
loss2 = l*(errors).^2*0.5;
loss = mean( (loss_upper1) + (loss_lower1) + (loss2),"all");

end

