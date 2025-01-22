function [h_upper, h_lower,  h_upper_t, h_lower_t, c_upper, c_lower] = Interval_lstmForward(Upper_in, Lower_in, lstm_param, hiddenSize, h_upper_t, h_lower_t, c_upper, c_lower)


h_upper = dlarray(zeros([length(lstm_param.rw_upper(1,:)) length(Upper_in(1,:,1))  length(Upper_in(1,1,:))]));
h_lower = dlarray(zeros([length(lstm_param.rw_upper(1,:)) length(Upper_in(1,:,1))  length(Upper_in(1,1,:))]));

% Iterate over time dimension/LSTM hidden state update
time_steps = length(Upper_in(1,1,:));

for t=1:time_steps


    upper_in = Upper_in(:,:,t);
    lower_in = Lower_in(:,:,t);

    % Concatenate the weights for input, forget, output, and cell gates
    % z = param.W * x + param.U * h_prev + param.b;
    [x_upper,x_lower] = Interval_fullyconnect(lstm_param.w_upper,lstm_param.w_lower,upper_in,lower_in,0,0);%W * x
    [r_upper,r_lower] = Interval_fullyconnect(lstm_param.rw_upper,lstm_param.rw_lower,h_upper_t,h_lower_t,0,0);%U * h_prev
    z_upper = x_upper + r_upper + lstm_param.bias_upper;
    z_lower = x_lower + r_lower + lstm_param.bias_lower;

    % Split the concatenated results into individual gate components
    i_upper = sigmoid(z_upper(1:hiddenSize, :));              % forget gate upper
    i_lower = sigmoid(z_lower(1:hiddenSize, :));              % forget gate lower

    f_upper = sigmoid(z_upper(hiddenSize+1:2*hiddenSize, :));  % input gate upper
    f_lower = sigmoid(z_lower(hiddenSize+1:2*hiddenSize, :));  % input gate lower

    c_tilde_upper  = tanh(z_upper(2*hiddenSize+1:3*hiddenSize, :));% output gate upper
    c_tilde_lower  = tanh(z_lower(2*hiddenSize+1:3*hiddenSize, :));% output gate lower

    o_upper = sigmoid(z_upper(3*hiddenSize+1:end, :));      % cell proposal upper
    o_lower = sigmoid(z_lower(3*hiddenSize+1:end, :));      % cell proposal lower


    % Update cell state
    % c_upper = f_upper .* c_upper + i_upper .* c_tilde_upper;
    % c_lower = f_lower .* c_lower + i_lower .* c_tilde_lower;
    [a1,a2] = Interval_elementwiseProduct(f_upper,f_lower,c_upper,c_lower);
    [b1,b2] = Interval_elementwiseProduct(i_upper,i_lower,c_tilde_upper,c_tilde_lower);

    c_upper = a1 + b1;
    c_lower = a2 + b2;

    % Update hidden state
    % h_upper = o_upper .* tanh(c_upper);
    % h_lower = o_lower .* tanh(c_lower);
    [h_upper_t,h_lower_t] = Interval_elementwiseProduct(o_upper,o_lower,tanh(c_upper), tanh(c_lower));
    h_upper(:,:,t)=h_upper_t;
    h_lower(:,:,t)=h_lower_t;

end

end

