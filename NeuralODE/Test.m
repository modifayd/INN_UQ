clear all
clc
close all

% ny-nx-nd-Activation-WindowLength-Normalization
output_lag = 3;
input_lag = 2;
dead_time = 2;
activation = "abs";
window_length = 30;
normalization_type="z_score";

%load pre-net
seed=1;
best_net = load("pre_net_seed_"+num2str(seed)+"_noise_"+num2str(0)+".mat");
% Get Net&params
pre_net = best_net.best_net;
% Pre Trained parameters
pre_net_param = pre_net;

% Interval param
param_delta = load("Interval_net_delta_seed_1_noise_0_loss_rqr_coverage_0.9_penalty_0.5.mat");
param_delta = param_delta.best_net;

% Split rates
p_train = 0.5*0.8;
p_val = 0.5*0.2;

% Load Data
% Hair Dryer
load dryer2;
U = u2';
X = y2';

% Heat exchanger
% load("exchanger.mat");

% MR-Damper
% load('mrdamper.mat');
% U=V';
% X=F';

% Dead time nd
U=[zeros(1,dead_time) U(1,1:end-dead_time)];
% Input Lag u(k),u(k-1),...
U=input_lagged(U,input_lag);

[train_in,train_out,train_in_all,train_out_all,val_in,val_out,test_in,test_out,statistics,idx_train,data_in,data_out] = data_gen(U,X,window_length,p_train,p_val,normalization_type);


% Pre-net-test
% % Get Network/parameters
neuralOdeParameters = best_net;
%

plot_var = struct(); % Store plot variables

% Separate targets for prediction in each set
train_target0 = train_out_all(:,:,1);

val_target0 = val_out(:,:,1);
test_target0 = test_out(:,:,1);

% Predictions for each set
plot_var.out_pred_train = euler_forward(@odeModel,1:1:size(train_out_all,3),dlarray((permute(stripdims(train_target0),[3 1 2])),"TCB"),train_in_all,pre_net_param, 1e-1, output_lag);
plot_var.out_pred_val = euler_forward(@odeModel,1:1:size(val_out,3),dlarray((permute(stripdims(val_target0),[3 1 2])),"TCB"),val_in,pre_net_param, 1e-1, output_lag);
plot_var.out_pred_test = euler_forward(@odeModel,1:1:size(test_out,3),dlarray((permute(stripdims(test_target0),[3 1 2])),"TCB"),test_in,pre_net_param, 1e-1, output_lag);

% Compute L2 loss (MSE) for each set
plot_var.mse_train = l2loss(plot_var.out_pred_train,permute(stripdims(train_out_all(:,:,2:end)),[3 1 2]),'DataFormat',"TCB","NormalizationFactor","all-elements");
plot_var.mse_val = l2loss(plot_var.out_pred_val,permute(stripdims(val_out(:,:,2:end)),[3 1 2]),'DataFormat',"TCB","NormalizationFactor","all-elements");
plot_var.mse_test = l2loss(plot_var.out_pred_test,permute(stripdims(test_out(:,:,2:end)),[3 1 2]),'DataFormat',"TCB","NormalizationFactor","all-elements");

plot_var.train_out_all=train_out_all;
plot_var.val_out=val_out;
plot_var.test_out=test_out;

plot_custom_crisp(plot_var)


% BFR result
bfr = BFR(squeeze(extractdata(test_out(:,:,2:end))),squeeze(extractdata(plot_var.out_pred_test)));

% Denormalized mse results

% z-score denormalization
if normalization_type == "z_score"
    X_test1 = ((X(:,floor((p_train+p_val)*end)+1:floor(end))))';
    mse_test = l2loss(plot_var.out_pred_test*(statistics.std_out)+statistics.mean_out,X_test1(2:end,:),'DataFormat',"CBT",NormalizationFactor="all-elements" );
    sqrt(mse_test)
end

if normalization_type == "min_max"
    % Min-max denormalization
    X_test1 = ((X(:,floor((p_train+p_val)*end)+1:floor(end))))';
    mse_test = l2loss(plot_var.out_pred_test*(statistics.upper_out-statistics.lower_out)+statistics.lower_out,X_test1(2:end,:),'DataFormat',"CBT",NormalizationFactor="all-elements" );
    sqrt(mse_test)
end



% Interval test

% Test on all data -- train val test


% Combine param_delta and pre_net
neuralOdeParameters_new_test = combined_param(param_delta,pre_net,activation);

% Separate targets for prediction in each set
train_target0 = train_out_all(:,:,1);
val_target0 = val_out(:,:,1);
test_target0 = test_out(:,:,1);

plot_var = struct(); % Store plot variables

% Predictions for each set
[plot_var.pred_upper_train, plot_var.pred_lower_train,plot_var.pred_mean_train] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(train_out_all,3), (train_target0),train_in_all, neuralOdeParameters_new_test,pre_net, 1e-5, output_lag);

[plot_var.pred_upper_val, plot_var.pred_lower_val,plot_var.pred_mean_val] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(val_out,3), (val_target0),val_in, neuralOdeParameters_new_test,pre_net, 1e-5, output_lag);

[plot_var.pred_upper_test, plot_var.pred_lower_test,plot_var.pred_mean_test] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(test_out,3), (test_target0),test_in, neuralOdeParameters_new_test,pre_net, 1e-5, output_lag);



% Plot

plot_var.train_out_all =train_out_all;
plot_var.val_out =val_out;
plot_var.test_out =test_out;

plot_custom_interval(plot_var)