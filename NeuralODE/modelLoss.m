function [loss,gradients] = modelLoss(tspan,input,neuralOdeParameters,targets, output_lag)

% Compute predictions.
X0=targets(:,:,1);
X = model(tspan,X0,input,neuralOdeParameters, output_lag);

% Compute L2 loss.
loss = l2loss(X,permute(stripdims(targets(:,:,2:end)),[3 1 2]),'DataFormat',"TCB","NormalizationFactor","all-elements");

% Compute gradients.
gradients = dlgradient(loss,neuralOdeParameters);

end