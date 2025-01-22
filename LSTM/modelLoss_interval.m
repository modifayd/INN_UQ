function [loss, gradients] = modelLoss_interval(param, pre_net_param,pre_net, in, out, loss_type, coverage, penalty ,output_lag, activation)

% Get initial output
out0=out(:,:,1);

% Forward propogation
[pred_upper, pred_lower] = forward_interval(param, pre_net_param,pre_net, in, out0, output_lag, activation);

% Loss calculation
if strcmp(loss_type, 'pinball')
    % Call the pinball loss function
    loss = pinball_loss(pred_upper,pred_lower,out(:,:,2:end), coverage, penalty);
elseif strcmp(loss_type, 'rqr')
    % Call the rqrw loss function
    loss = rqrw_loss(pred_upper,pred_lower,out(:,:,2:end), coverage, penalty);
else
    % Handle cases where an invalid loss_type is provided
    error('Unknown loss type: %s. Please use ''pinball'' or ''rqr''.', loss_type);
end

% Gradient calculation
gradients = dlgradient(loss,param);

end

