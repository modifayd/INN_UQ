function [pred_upper, pred_lower] = forward_interval(param, pre_net_param, net, in, out0, output_lag, activation)
    % Adaptive LSTM + FC implementation

    in = stripdims(in);
    out0 = stripdims(out0);

    if activation=="relu"
    % Combine parameters for FC layer
    upper_w = pre_net_param.w + relu(param.weights_upper_delta);
    lower_w = pre_net_param.w - relu(param.weights_lower_delta);
    upper_b = pre_net_param.b + relu(param.bias_upper_delta);
    lower_b = pre_net_param.b - relu(param.bias_lower_delta);

    % Define number of LSTM layers dynamically
    numLSTMLayers = (numel(fieldnames(param)) - 4) / 6; % Assuming 6 params per LSTM
    lstm_params = cell(1, numLSTMLayers);
    hiddenSizes = zeros(1, numLSTMLayers);

    % Dynamically initialize LSTM parameters
    for i = 1:numLSTMLayers
        lstm_params{i}.w_upper = relu(param.(['Wlstm_upper_delta' num2str(i)])) + pre_net_param.(['lstm_w' num2str(i)]);
        lstm_params{i}.rw_upper = relu(param.(['Ulstm_upper_delta' num2str(i)])) + pre_net_param.(['lstm_rw' num2str(i)]);
        lstm_params{i}.bias_upper = relu(param.(['blstm_upper_delta' num2str(i)])) + pre_net_param.(['lstm_b' num2str(i)]);

        lstm_params{i}.w_lower = -relu(param.(['Wlstm_lower_delta' num2str(i)])) + pre_net_param.(['lstm_w' num2str(i)]);
        lstm_params{i}.rw_lower = -relu(param.(['Ulstm_lower_delta' num2str(i)])) + pre_net_param.(['lstm_rw' num2str(i)]);
        lstm_params{i}.bias_lower = -relu(param.(['blstm_lower_delta' num2str(i)])) + pre_net_param.(['lstm_b' num2str(i)]);

        hiddenSizes(i) = length(param.(['Ulstm_upper_delta' num2str(i)])(:, 1)) / 4;
    end
    end

    if activation=="abs"
    % Combine parameters for FC layer
    upper_w = pre_net_param.w + abs(param.weights_upper_delta);
    lower_w = pre_net_param.w - abs(param.weights_lower_delta);
    upper_b = pre_net_param.b + abs(param.bias_upper_delta);
    lower_b = pre_net_param.b - abs(param.bias_lower_delta);

    % Define number of LSTM layers dynamically
    numLSTMLayers = (numel(fieldnames(param)) - 4) / 6; % Assuming 6 params per LSTM
    lstm_params = cell(1, numLSTMLayers);
    hiddenSizes = zeros(1, numLSTMLayers);

    % Dynamically initialize LSTM parameters
    for i = 1:numLSTMLayers
        lstm_params{i}.w_upper = abs(param.(['Wlstm_upper_delta' num2str(i)])) + pre_net_param.(['lstm_w' num2str(i)]);
        lstm_params{i}.rw_upper = abs(param.(['Ulstm_upper_delta' num2str(i)])) + pre_net_param.(['lstm_rw' num2str(i)]);
        lstm_params{i}.bias_upper = abs(param.(['blstm_upper_delta' num2str(i)])) + pre_net_param.(['lstm_b' num2str(i)]);

        lstm_params{i}.w_lower = -abs(param.(['Wlstm_lower_delta' num2str(i)])) + pre_net_param.(['lstm_w' num2str(i)]);
        lstm_params{i}.rw_lower = -abs(param.(['Ulstm_lower_delta' num2str(i)])) + pre_net_param.(['lstm_rw' num2str(i)]);
        lstm_params{i}.bias_lower = -abs(param.(['blstm_lower_delta' num2str(i)])) + pre_net_param.(['lstm_b' num2str(i)]);

        hiddenSizes(i) = length(param.(['Ulstm_upper_delta' num2str(i)])(:, 1)) / 4;
    end
        end



    % Initialize outputs and hidden states
    pred_upper = dlarray(zeros([size(out0, 1), size(in, 2), size(in, 3) - 1]));
    pred_lower = dlarray(zeros([size(out0, 1), size(in, 2), size(in, 3) - 1]));
    out = dlarray(zeros([size(out0, 1), size(in, 2), size(in, 3) - 1]));

    % Initialize hidden and cell states
    H0_upper = cell(1, numLSTMLayers);
    H0_lower = cell(1, numLSTMLayers);
    C0_upper = cell(1, numLSTMLayers);
    C0_lower = cell(1, numLSTMLayers);

    for i = 1:numLSTMLayers
        H0_upper{i} = dlarray(zeros(hiddenSizes(i), size(in, 2)));
        H0_lower{i} = dlarray(zeros(hiddenSizes(i), size(in, 2)));
        C0_upper{i} = dlarray(zeros(hiddenSizes(i), size(in, 2)));
        C0_lower{i} = dlarray(zeros(hiddenSizes(i), size(in, 2)));
    end

    % Extend initial outputs with lag
    out0 = cat(3, zeros([size(out, 1), size(out, 2), output_lag - 1]), out0);
    out0 = permute(out0, [3 2 1]);

    % Forward pass through time
    for t = 1:(size(in, 3) - 1)
        model_in = cat(1, in(:, :, t), out0);

        % Pass through LSTM layers dynamically
        h_upper = model_in;
        h_lower = model_in;
        for j = 1:numLSTMLayers
            [h_upper, h_lower, ~, ~, ~, ~] = Interval_lstmForward(h_upper, h_lower, lstm_params{j}, ...
                                                                  hiddenSizes(j), H0_upper{j}, H0_lower{j}, ...
                                                                  C0_upper{j}, C0_lower{j});
        end

        % Final FC layer
        [pred_upper(:, :, t), pred_lower(:, :, t)] = Interval_fullyconnect(upper_w, lower_w, h_upper, h_lower, upper_b, lower_b);

        % Mean prediction
        [out(:, :, t), state] = predict(net, dlarray(model_in, "CBT"));
        net.State = state;

        % Update output lag
        if output_lag > t
            out0 = cat(3, zeros([size(out, 1), size(out, 2), output_lag - t]), out(:, :, 1:t));
        else
            out0 = out(:, :, t - output_lag + 1:t);
        end
        out0 = permute(out0, [3 2 1]);

        % Update hidden and cell states
        for j = 1:numLSTMLayers
            H0_upper{j} = cell2mat(table2array(state(2 * j - 1, 3)));
            H0_lower{j} = cell2mat(table2array(state(2 * j - 1, 3)));
            C0_upper{j} = cell2mat(table2array(state(2 * j, 3)));
            C0_lower{j} = cell2mat(table2array(state(2 * j, 3)));
        end
    end
end



