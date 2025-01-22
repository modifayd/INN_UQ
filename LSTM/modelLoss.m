function [loss,gradients] = modelLoss(net,input,targets,output_lag)

% Get initial output
target0= targets(:,:,1);

% Compute predictions.
out_pred = forward_custom(net,input,target0,output_lag);

% Compute L2 loss.
loss = l2loss(out_pred,targets(:,:,2:end),'DataFormat',"CBT",NormalizationFactor="all-elements" );

% Compute gradients.
gradients = dlgradient(loss,net.Learnables);

end

