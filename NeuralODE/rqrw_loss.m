function loss = rqrw_loss(pred_upper,pred_lower,out, coverage, penalty)
%RQR-W loss
c=coverage;%coverage
l=penalty;%penalty
error1 = (out(:,:,2:end)-permute(pred_lower(:,:,:),[2 3 1]));
error2 = (out(:,:,2:end)-permute(pred_upper(:,:,:),[2 3 1]));
errors = permute(pred_upper(:,:,:),[2 3 1])-permute(pred_lower(:,:,:),[2 3 1]);
loss1 = max(error1.*error2.*(c+2*l),error2.*error1.*(c+2*l-1));
loss2 = l*(errors).^2*0.5;
loss = mean(loss1+loss2,"all");%+0*0.1*(sum(param.bias_lower_delta1.^2)+sum(param.bias_lower_delta2.^2)+sum(param.bias_upper_delta1.^2)+sum(param.bias_upper_delta2.^2));
end


