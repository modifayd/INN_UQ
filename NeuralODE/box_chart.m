
clear all
close all
clc


% Model features
seedx = [1 2 3 4 5];
noiseLevelx = 0;%[0.0 0.2];
loss_typex ="rqr";%["pinball","rqr"];
Penaltyx = 0.5;%[0.0 0.5];
Coveragex = 0.9;%[0.9 0.95];
normalization_type = "z_score";
window_length = 30;

% ny-nx-nd
output_lag = 3;
input_lag = 2;
dead_time = 2;

% Split rates
p_train = 0.5*0.8;
p_val = 0.5*0.2;

% Activation -- relu, abs
activation="abs";

% Initialize figure and data
figure;
hold on;
x = []; % x-values for boxchart
data = {}; % Cell array to store data for each x
categories = {}; % Cell array to store dynamic labels
index=0;

for nn=1:length(noiseLevelx)
    for cc=1:length(loss_typex)
        for ii=1:length(Coveragex)
            for jj=1:length(Penaltyx)
                for mm=1:length(seedx)

                    % Calculate model for each seed in the most inner loop

                    rng(15) % Same input for all models
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

                    % Input config
                    U=[zeros(1,dead_time) U(1,1:end-dead_time)];
                    U=input_lagged(U,input_lag);

                    % Model parameters
                    coverage=Coveragex(ii);
                    penalty=Penaltyx(jj);
                    loss_type=loss_typex(cc);
                    noiseLevel = noiseLevelx(nn);
                    seed=seedx(mm);

                    % Add noise to model
                    X=X+randn(size(X))*noiseLevel;

                    %% Train-Val-Test Split

                    [train_in,train_out,train_in_all,train_out_all,val_in,val_out,test_in,test_out,statistics,idx_train,data_in,data_out] = data_gen(U,X,window_length,p_train,p_val,normalization_type);

                    %%
                    % Evaluate on all data train val test

                    % Load pre net
                    net = load("pre_net_seed_"+num2str(seed)+"_noise_"+num2str(noiseLevel)+".mat");
                    pre_trained_net = net.best_net;

                    % Load delta param
                    param_delta = load("Interval_net_delta_seed_"+num2str(seed)+"_noise_"+num2str(noiseLevel)+"_loss_"+char(loss_type)+"_coverage_"+num2str(coverage)+"_penalty_"+num2str(penalty)+".mat");
                    % param_delta = load("Interval_net_delta_seed_2_noise_0_loss_pinball_coverage_0.9_penalty_0.mat");

                    param_delta = param_delta.best_net;

                    % Combine param_delta and pre_net
                    neuralOdeParameters_new_test = combined_param(param_delta,pre_trained_net, activation);

                    % Separate targets0 for prediction in each set
                    train_target0 = train_out_all(:,:,1);
                    val_target0 = val_out(:,:,1);
                    test_target0 = test_out(:,:,1);


                    % Predictions for each set
                    % [pred_upper_train, pred_lower_train,pred_mean_train] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(train_out_all,3), (train_target0),train_in_all, neuralOdeParameters_new_test,pre_trained_net, 1e-5,output_lag);
                    %
                    % [pred_upper_val, pred_lower_val,pred_mean_val] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(val_out,3), (val_target0),val_in, neuralOdeParameters_new_test,pre_trained_net, 1e-5,output_lag);

                    [pred_upper_test, pred_lower_test,pred_mean_test] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(test_out,3), (test_target0),test_in, neuralOdeParameters_new_test,pre_trained_net, 1e-5,output_lag);


                    % % Plot
                    % plot_var = struct();
                    %
                    % plot_var.train_out_all =train_out_all;
                    % plot_var.val_out =val_out;
                    % plot_var.test_out =test_out;
                    %
                    % plot_var.pred_mean_train = pred_mean_train;
                    % plot_var.pred_mean_val=pred_mean_val;
                    % plot_var.pred_mean_test=pred_mean_test;
                    %
                    % plot_var.pred_upper_train = pred_upper_train;
                    % plot_var.pred_upper_val = pred_upper_val;
                    % plot_var.pred_upper_test = pred_upper_test;
                    %
                    % plot_var.pred_lower_train = pred_lower_train;
                    % plot_var.pred_lower_val = pred_lower_val;
                    % plot_var.pred_lower_test = pred_lower_test;
                    %
                    % plot_custom_interval(plot_var)



                    % Compute PICP for each set
                    % picp_train(seed) = PICP(squeeze(extractdata(train_out_all(:,:,2:end))), ...
                    %     squeeze(pred_lower_train(:,:)), ...
                    %     squeeze(pred_upper_train(:,:))); % Assuming you have upper/lower for train
                    % picp_val(seed) = PICP(squeeze(extractdata(val_out(:,:,2:end))), ...
                    %     squeeze(pred_lower_val(:,:)), ...
                    %     squeeze(pred_upper_val(:,:))); % Assuming you have upper/lower for val
                    picp_test(seed) = PICP(squeeze(extractdata(test_out(:,:,2:end))), ...
                        squeeze(pred_lower_test(:,:)), ...
                        squeeze(pred_upper_test(:,:))); % Assuming you have upper/lower for test

                    % Compute PINAW for each set
                    % pinaw_train = PINAW(squeeze(extractdata(train_out_all(i,:,2:end))), ...
                    %                   squeeze(pred_lower_train(i,:,:)), ...
                    %                   squeeze(pred_upper_train(i,:,:))); % Assuming you have upper/lower for train
                    % pinaw_val = PINAW(squeeze(extractdata(val_out(i,:,2:end))), ...
                    %                 squeeze(pred_lower_val(i,:,:)), ...
                    %                 squeeze(pred_upper_val(i,:,:))); % Assuming you have upper/lower for val
                    % pinaw_test(seed) = PINAW(squeeze(extractdata(test_out(:,:,2:end))), ...
                    %          squeeze(pred_lower_test(:,:)), ...
                    %          squeeze(pred_upper_test(:,:))); % Assuming you have upper/lower for test



                end

                % Interval box chart

                % Iterate and add box charts

                index=index+1;

                % Generate some data for this iteration (replace with your actual data)
                currentData = extractdata(picp_test);
                % Create the box chart for this iteration

                % Simulate new data for the current x value
                x(end+1) = index; % New x-point
                data{end+1} = currentData; % Add to data list
                mean_(index) = mean(currentData);
                std_(index) = std(currentData,1);
                loss_=char(loss_type);
                categories{end+1} = ['N',num2str(noiseLevel),'Loss',upper(loss_(1)),'-Cov',num2str(coverage),'Pen',num2str(penalty)];

                % Clear and replot boxchart for all x-values
                cla; % Clear current axes
                for j = 1:length(x)
                    boxchart(x(j) * ones(size(data{j})), data{j});

                end

                % Update xticks and xticklabels
                xticks(x);
                xticklabels(categories);
                set(gca, 'XTickLabelRotation', 90); % Rotate labels vertically

                % Pause to simulate live plotting
                pause(1);

            end
        end
    end
end

% Add mean to plot
% hold on;plot(mean_,"-o")

