
function [y_upper, y_lower] = odeModelInterval(in_upper, in_lower, theta)
% Initialize the outputs
y_upper = in_upper;
y_lower = in_lower;

% Get the field names of theta, which correspond to the layers
layerNames = fieldnames(theta);
numLayers = numel(layerNames);  % Number of layers based on the number of fields

% Iterate through each layer except the last one
for i = 1:numLayers-1
    % Get the name of the current layer
    layer = layerNames{i};

    % Apply the interval fully connected operation followed by tanh activation
    [pred_upper, pred_lower] = Interval_fullyconnect(...
        theta.(layer).Weights_upper, theta.(layer).Weights_lower, ...
        y_upper, y_lower, ...
        theta.(layer).Bias_upper, theta.(layer).Bias_lower);

    % Apply tanh activation to the upper and lower predictions
    y_upper = tanh(pred_upper);
    y_lower = tanh(pred_lower);
end

% Handle the last layer (no tanh activation)
lastLayer = layerNames{numLayers};  % Get the name of the last layer
[delta_upper, delta_lower] = Interval_fullyconnect(...
    theta.(lastLayer).Weights_upper, theta.(lastLayer).Weights_lower, ...
    y_upper, y_lower, ...
    theta.(lastLayer).Bias_upper, theta.(lastLayer).Bias_lower);

% Add input to output for residual connection (matching size of Bias in the last layer)
y_upper = delta_upper + in_upper(1:numel(theta.(lastLayer).Bias_lower), :);
y_lower = delta_lower + in_lower(1:numel(theta.(lastLayer).Bias_lower), :);
end



