
function param = delta_param_init(pre_net_param,output_uncertainity_rate,hidden_uncertainity_rate)

% Initialize the param struct
param = struct();

% Identify the number of LSTMs by checking fields in pre_net_param
lstmFields = fieldnames(pre_net_param);
numLSTMs = sum(contains(lstmFields, 'lstm_w'));

% Loop through each LSTM layer and set the upper and lower deltas
for i = 1:numLSTMs
    % Get the LSTM parameters from pre_net_param
    lstm_w = pre_net_param.(['lstm_w' num2str(i)]);
    lstm_rw = pre_net_param.(['lstm_rw' num2str(i)]);
    lstm_b = pre_net_param.(['lstm_b' num2str(i)]);

    % Set the upper delta values for weights, recurrent weights, and biases
    param.(['Wlstm_upper_delta' num2str(i)]) = dlarray(abs(lstm_w) * hidden_uncertainity_rate);
    param.(['Ulstm_upper_delta' num2str(i)]) = dlarray(abs(lstm_rw) * hidden_uncertainity_rate);
    param.(['blstm_upper_delta' num2str(i)]) = dlarray(abs(lstm_b) * hidden_uncertainity_rate);

    % Set the lower delta values for weights, recurrent weights, and biases
    param.(['Wlstm_lower_delta' num2str(i)]) = dlarray(abs(lstm_w) * hidden_uncertainity_rate);
    param.(['Ulstm_lower_delta' num2str(i)]) = dlarray(abs(lstm_rw) * hidden_uncertainity_rate);
    param.(['blstm_lower_delta' num2str(i)]) = dlarray(abs(lstm_b) * hidden_uncertainity_rate);
end

% Set the deltas for the fully connected layer weights and biases
param.weights_upper_delta = dlarray(output_uncertainity_rate * abs(pre_net_param.w));
param.bias_upper_delta = dlarray(output_uncertainity_rate * abs(pre_net_param.b));

param.weights_lower_delta = dlarray(output_uncertainity_rate * abs(pre_net_param.w));
param.bias_lower_delta = dlarray(output_uncertainity_rate * abs(pre_net_param.b));

end
