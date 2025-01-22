
function pre_net_param = pre_trained_parameters(net)
% Initialize pre_net_param struct
pre_net_param = struct();

% Calculate the number of LSTMs in the network
numLearnables = size(net.Learnables, 1);
numLSTMs = (numLearnables - 2) / 3;  % Each LSTM has 3 parts, last layer has 2 parts

% Loop through each LSTM and store its parameters
for i = 1:numLSTMs
    % Calculate indices for weights, recurrent weights, and biases
    w_idx = 3 * (i - 1) + 1;
    rw_idx = 3 * (i - 1) + 2;
    b_idx = 3 * (i - 1) + 3;

    % Store LSTM weights, recurrent weights, and biases in the struct
    pre_net_param.(['lstm_w' num2str(i)]) = cell2mat(net.Learnables{w_idx, 3});
    pre_net_param.(['lstm_rw' num2str(i)]) = cell2mat(net.Learnables{rw_idx, 3});
    pre_net_param.(['lstm_b' num2str(i)]) = cell2mat(net.Learnables{b_idx, 3});
end

% Store final layer weights and biases
pre_net_param.w = cell2mat(net.Learnables{end-1, 3});
pre_net_param.b = cell2mat(net.Learnables{end, 3});
end
