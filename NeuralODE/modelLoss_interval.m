function [loss,gradients] = modelLoss_interval(tspan,in,neuralOdeParameters,pre_net,targets, loss_type, coverage, penalty, output_lag, activation)

%Intervalize pre trained network
neuralOdeParameters_new = combined_param(neuralOdeParameters,pre_net,activation);

% Get initial output data
X0 = squeeze(targets(:,:,1));

% Get predictions.
[Xu , Xl] = model(tspan, X0, in, neuralOdeParameters_new,pre_net, output_lag);

% Loss calculation
if strcmp(loss_type, 'pinball')
    % Call the pinball loss function
    loss = pinball_loss(Xu, Xl, targets, coverage, penalty);
elseif strcmp(loss_type, 'rqr')
    % Call the rqrw loss function
    loss = rqrw_loss(Xu, Xl, targets, coverage, penalty);
else
    % Handle cases where an invalid loss_type is provided
    error('Unknown loss type: %s. Please use ''pinball'' or ''rqr''.', loss_type);
end

% Compute gradients.
gradients = dlgradient(loss,neuralOdeParameters);

end


function [Y_upper, Y_lower] = model(tspan, X0, in,neuralOdeParameters,net, output_lag)

tol=1e-5;
[Y_upper, Y_lower, ~] = euler_forward_interval(@odeModelInterval,@odeModel, tspan, X0, in, neuralOdeParameters, net, tol, output_lag);

end







