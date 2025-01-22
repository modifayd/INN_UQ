function y = odeModel(~, in, theta)
% Initialize the input
y = in;

% Get the field names of theta, which correspond to the layers
layerNames = fieldnames(theta);
numLayers = numel(layerNames);  % Number of layers based on the number of fields

for i = 1:numLayers-1
    % Apply the linear transformation followed by tanh activation for all but the last layer
    y = tanh(theta.(sprintf('fc%d', i)).Weights * y + theta.(sprintf('fc%d', i)).Bias);
end

% Apply the last layer (without tanh)
y = theta.(sprintf('fc%d', numLayers)).Weights * y + theta.(sprintf('fc%d', numLayers)).Bias;

% Add input to output for residual connection (matching size of Bias in the last layer)
y = y + in(1:numel(theta.(sprintf('fc%d', numLayers)).Bias), :);
end


