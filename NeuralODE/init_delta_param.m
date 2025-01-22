
function neuralOdeParameters_delta = init_delta_param(X, U, pre_trained_net, hidden_uncertainty_rate, output_uncertainty_rate)

neuralOdeParameters_delta = struct;

inSize = size(X, 1) + size(U, 1); % Input size
outSize = size(X, 1);            % Output size

% Get the number of layers dynamically
layerNames = fieldnames(pre_trained_net);
numLayers = length(layerNames);

% Initialize sizes for the first layer
prevSize = inSize;

% Iterate through each layer
for i = 1:numLayers
    layerName = sprintf('fc%d', i); % Generate the layer name dynamically
    currentSize = size(pre_trained_net.(layerName).Weights, 1); % Number of units in the current layer

    % Determine uncertainty rate
    if i == numLayers
        % Output layer
        uncertainty_rate = output_uncertainty_rate;
        outputSize = outSize;
    else
        % Hidden layers
        uncertainty_rate = hidden_uncertainty_rate;
    end

    % Initialize parameters for the current layer
    neuralOdeParameters_delta.(layerName) = struct;
    sz = [currentSize prevSize]; % Size of weight matrix

    neuralOdeParameters_delta.(layerName).Weights_upper = abs(pre_trained_net.(layerName).Weights) * uncertainty_rate;
    neuralOdeParameters_delta.(layerName).Bias_upper = abs(pre_trained_net.(layerName).Bias) * uncertainty_rate;
    neuralOdeParameters_delta.(layerName).Weights_lower = abs(pre_trained_net.(layerName).Weights) * uncertainty_rate;
    neuralOdeParameters_delta.(layerName).Bias_lower = abs(pre_trained_net.(layerName).Bias) * uncertainty_rate;

    % Update the previous size for the next layer
    prevSize = currentSize;
end

end
