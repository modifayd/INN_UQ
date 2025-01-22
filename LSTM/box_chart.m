
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

                    % Calculate model for each seed

                    rng(15) 

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
                    
                    % Data generation
                    [train_in,train_out,train_in_all,train_out_all,val_in,val_out,test_in,test_out,statistics,idx_train,data_in,data_out] = data_gen(U,X,window_length,p_train,p_val,normalization_type);

                    %%
                    % Evaluate on all data train val test

                    %load pre net
                    net = load("pre_net_seed_"+num2str(seed)+"_noise_"+num2str(noiseLevel)+".mat");
                    net = net.best_net;

                    % Pre Trained parameters
                    pre_net_param = pre_trained_parameters(net);

                    %load delta param
                    param_delta = load("Interval_net_delta_seed_"+num2str(seed)+"_noise_"+num2str(noiseLevel)+"_loss_"+char(loss_type)+"_coverage_"+num2str(coverage)+"_penalty_"+num2str(penalty)+".mat");
                    param_delta = param_delta.best_net;

                    % Test on all data -- train val test

                    % Separate targets for prediction in each set
                    train_target0 = train_out_all(:,:,1);
                    val_target0 = val_out(:,:,1);
                    test_target0 = test_out(:,:,1);

                    % Mean Predictions
                    % pred_mean_train = forwardMean(net,train_in_all,train_target0,output_lag);
                    % pred_mean_val = forwardMean(net,val_in,val_target0,output_lag);
                    pred_mean_test = forwardMean(net,test_in,test_target0,output_lag);

                    % Predictions for each set
                    % [pred_upper_train, pred_lower_train] = forward_interval(param_delta, pre_net_param, net, train_in_all, train_target0,output_lag, activation);
                    % [pred_upper_val, pred_lower_val] = forward_interval(param_delta, pre_net_param, net, val_in, val_target0,output_lag, activation);
                    [pred_upper_test, pred_lower_test] = forward_interval(param_delta, pre_net_param, net, test_in, test_target0,output_lag, activation);



                    %%

                    % Compute PICP for each set
                    % picp_train(seed) = PICP(squeeze(extractdata(train_out_all(1,:,2:end))), ...
                    %     squeeze(pred_lower_train(1,:,:)), ...
                    %     squeeze(pred_upper_train(1,:,:))); % Assuming you have upper/lower for train
                    % picp_val(seed) = PICP(squeeze(extractdata(val_out(1,:,2:end))), ...
                    %     squeeze(pred_lower_val(1,:,:)), ...
                    %     squeeze(pred_upper_val(1,:,:))); % Assuming you have upper/lower for val
                    picp_test(seed) = PICP(squeeze(extractdata(test_out(1,:,2:end))), ...
                        squeeze(pred_lower_test(1,:,:)), ...
                        squeeze(pred_upper_test(1,:,:))); % Assuming you have upper/lower for test

                    % Compute PINAW for each set
                    % pinaw_train = PINAW(squeeze(extractdata(train_out_all(1,:,2:end))), ...
                    %                   squeeze(pred_lower_train(1,:,:)), ...
                    %                   squeeze(pred_upper_train(1,:,:))); % Assuming you have upper/lower for train
                    % pinaw_val = PINAW(squeeze(extractdata(val_out(1,:,2:end))), ...
                    %                 squeeze(pred_lower_val(1,:,:)), ...
                    %                 squeeze(pred_upper_val(1,:,:))); % Assuming you have upper/lower for val
                    % pinaw_test(seed) = PINAW(squeeze(extractdata(test_out(1,:,2:end))), ...
                    %          squeeze(pred_lower_test(1,:,:)), ...
                    %          squeeze(pred_upper_test(1,:,:))); % Assuming you have upper/lower for test


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
                loss_=char(loss_type);
                mean_(index) = mean(currentData);
                std_(index) = std(currentData,1);
                categories{end+1} = ['N',num2str(noiseLevel),'Loss',upper(loss_(1)),'-Cov',num2str(coverage),'Pen',num2str(penalty*1)];

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

