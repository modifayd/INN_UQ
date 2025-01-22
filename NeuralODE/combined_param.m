
function neuralOdeParameters_new = combined_param(neuralOdeParameters, pre_net, activation)

% Get the number of layers from the fields in the structure
numLayers = length(fieldnames(neuralOdeParameters));

% Iterate through each layer
for i = 1:numLayers
    layerName = sprintf('fc%d', i); % Generate the layer name dynamically

    if activation =="relu"
    % Update the parameters for the current layer
    neuralOdeParameters_new.(layerName).Weights_upper = relu(neuralOdeParameters.(layerName).Weights_upper) + pre_net.(layerName).Weights;
    neuralOdeParameters_new.(layerName).Bias_upper = relu(neuralOdeParameters.(layerName).Bias_upper) + pre_net.(layerName).Bias;
    neuralOdeParameters_new.(layerName).Weights_lower = -relu(neuralOdeParameters.(layerName).Weights_lower) + pre_net.(layerName).Weights;
    neuralOdeParameters_new.(layerName).Bias_lower = -relu(neuralOdeParameters.(layerName).Bias_lower) + pre_net.(layerName).Bias;

    end

        if activation =="abs"
    % Update the parameters for the current layer
    neuralOdeParameters_new.(layerName).Weights_upper = abs(neuralOdeParameters.(layerName).Weights_upper) + pre_net.(layerName).Weights;
    neuralOdeParameters_new.(layerName).Bias_upper = abs(neuralOdeParameters.(layerName).Bias_upper) + pre_net.(layerName).Bias;
    neuralOdeParameters_new.(layerName).Weights_lower = -abs(neuralOdeParameters.(layerName).Weights_lower) + pre_net.(layerName).Weights;
    neuralOdeParameters_new.(layerName).Bias_lower = -abs(neuralOdeParameters.(layerName).Bias_lower) + pre_net.(layerName).Bias;

    end

end

end
