clear all
clc
close all



% ny-nx-nd
output_lag = 1;
input_lag = 2;
dead_time = 0;

% Split rates
p_train = 2/3*0.8;
p_val = 2/3*0.2;

% Activation -- relu, abs
activation="abs";

% Normalization Type
normalization_type = "z_score";


%load pre-net
seed=1;
best_net = load(['pre_net_seed_', num2str(seed), '_noise_0.mat']);
% Get Net&params
net = best_net.best_net;
% Pre Trained parameters
pre_net_param = pre_trained_parameters(net);

% Interval param
param_delta = load("Interval_net_delta_seed_1_noise_0_loss_rqr_coverage_0.9_penalty_0.5.mat");
param_delta = param_delta.best_net;


% Hair Dryer
% load dryer2;
% U = u2';
% X = y2';

% Heat exchanger
% load("exchanger.mat");

% MR-Damper
load('mrdamper.mat');
U=V';
X=F';


% Dead time nd
U=[zeros(1,dead_time) U(1,1:end-dead_time)];
% Input Lag u(k),u(k-1),...
U=input_lagged(U,input_lag);
% Data gen
[train_in,train_out,train_in_all,train_out_all,val_in,val_out,test_in,test_out,statistics,idx_train,data_in,data_out] = data_gen(U,X,window_length,p_train,p_val,normalization_type);





% Pre-net-test

% Separate targets for prediction in each set
train_target0 = train_out_all(:,:,1);
val_target0 = val_out(:,:,1);
test_target0 = test_out(:,:,1);

% Predictions for each set
out_pred_train = forward_custom(net, train_in_all, train_target0,output_lag);
out_pred_val = forward_custom(net, val_in, val_target0,output_lag);
out_pred_test = forward_custom(net, test_in, test_target0,output_lag);

% Compute L2 loss (MSE) for each set -- normalized data result
l2_train = l2loss(out_pred_train,train_out_all(:,:,2:end),'DataFormat',"CBT",NormalizationFactor="all-elements" );
l2_val = l2loss(out_pred_val,val_out(:,:,2:end),'DataFormat',"CBT",NormalizationFactor="all-elements" );
l2_test = l2loss(out_pred_test,test_out(:,:,2:end),'DataFormat',"CBT",NormalizationFactor="all-elements" );


% Plot output for all data --> train val test
plot_var = struct();
plot_var.train_out_all=train_out_all;
plot_var.val_out=val_out;
plot_var.test_out=test_out;

plot_var.out_pred_train=out_pred_train;
plot_var.out_pred_val=out_pred_val;
plot_var.out_pred_test=out_pred_test;

plot_var.mse_train=l2_train;
plot_var.mse_val=l2_val;
plot_var.mse_test=l2_test;

plot_custom_crisp(plot_var)


% BFR result
bfr = BFR(squeeze(extractdata(test_out(:,:,2:end))),squeeze(extractdata(out_pred_test)));


% Denormalized mse results

% Denormalized mse results

% z-score denormalization
if normalization_type == "z_score"

    X_test = ((X(:,floor((p_train+p_val)*end)+1:floor(end))))';
    test_out1 = dlarray(X_test,"TCB");
    mse_test = l2loss(out_pred_test*statistics.std_out+statistics.mean_out,test_out1(:,:,2:end),'DataFormat',"CBT",NormalizationFactor="all-elements" );
    sqrt(mse_test)
end

% Min-max denormalization
if normalization_type == "min_max"

    X_test = ((X(:,floor((p_train+p_val)*end)+1:floor(end))))';
    test_out1 = dlarray(X_test,"TCB");
    mse_test = l2loss(out_pred_test*(statistics.upper_out-statistics.lower_out)+statistics.lower_out,test_out1(:,:,2:end),'DataFormat',"CBT",NormalizationFactor="all-elements" );
    sqrt(mse_test)

end

% Interval test

% Test on all data -- train val test

plot_var = struct(); % Store plot variables

% Separate targets for prediction in each set
train_target0 = train_out_all(:,:,1);
val_target0 = val_out(:,:,1);
test_target0 = test_out(:,:,1);

% Mean Predictions
plot_var.pred_mean_train = forwardMean(net,train_in_all,train_target0,output_lag);
plot_var.pred_mean_val = forwardMean(net,val_in,val_target0,output_lag);
plot_var.pred_mean_test = forwardMean(net,test_in,test_target0,output_lag);

% Predictions for each set
[plot_var.pred_upper_train, plot_var.pred_lower_train] = forward_interval(param_delta, pre_net_param, net, train_in_all, train_target0,output_lag,activation);
[plot_var.pred_upper_val, plot_var.pred_lower_val] = forward_interval(param_delta, pre_net_param, net, val_in, val_target0,output_lag,activation);
[plot_var.pred_upper_test, plot_var.pred_lower_test] = forward_interval(param_delta, pre_net_param, net, test_in, test_target0,output_lag,activation);



% Plot

plot_var.train_out_all = train_out_all;
plot_var.val_out = val_out;
plot_var.test_out = test_out;

plot_custom_interval(plot_var)




