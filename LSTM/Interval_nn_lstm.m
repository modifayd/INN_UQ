clear all
clc
close all


% Model features
seedx = [1 2 3 4 5];
noiseLevelx = 0;%[0.0 0.2];
loss_typex ="rqr";%["pinball","rqr"];
Penaltyx = 0.5;%[0.0 0.5];
Coveragex = 0.9;%[0.9 0.95];
normalization_type = "z_score";

% ny-nx-nd
output_lag = 1;
input_lag = 2;
dead_time = 0;

% Split rates
p_train = 2/3*0.8;
p_val = 2/3*0.2;

% Activation -- relu, abs
activation="abs";

% Create upper lower Weights&Biases
output_uncertainity_rate=1; % Output layer delta magnitude level
hidden_uncertainity_rate=0.2; % Hidden layers delta magnitude level

% Window length
window_length=40;


for nn=1:length(noiseLevelx)
    for mm=1:length(seedx)
        for cc=1:length(loss_typex)
            for ii=1:length(Coveragex)
                for jj=1:length(Penaltyx)


                    rng(15) % Same input for all models

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

                    % dead time and input lag
                    U=[zeros(1,dead_time) U(1,1:end-dead_time)];
                    U=input_lagged(U,input_lag);

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
                    % Pre Trained Net -- dlnetwork
                    net = load("pre_net_seed_"+num2str(seed)+"_noise_"+num2str(noiseLevel)+".mat");
                    net = net.best_net;
                    % Pre Trained parameters
                    pre_net_param = pre_trained_parameters(net);

                    % Init Param delta
                    param_delta = delta_param_init(pre_net_param,output_uncertainity_rate,hidden_uncertainity_rate);

                    %% Training/fitting

                    rng(seed); % fixed randomness for training

                    % Hyperparameters
                    velocity=[];
                    gradients=[];
                    averageGrad=[];
                    averageSqGrad=[];
                    learnRate=5*1e-3;
                    Batch_size=64;

                    %early stopping paramters
                    counter=0;
                    min_val_loss=inf;
                    patience=5;
                    min_delta=0.0005;

                    iter=1;

                    % figure();
                    % h = animatedline();
                    % h2 = animatedline("Color","r");
                    % legend("train","validation")
                    % xlabel("iter")
                    % ylabel('Loss');
                    % ax.YGrid = 'on';
                    % ax.XGrid = 'on';
                    % title("Interval Loss")

                    for epoch = 1:20

                        batch_loss=0;

                        % Learning rate scheduler
                        if(mod(epoch, 10) == 0)
                            learnRate = learnRate*0.5;
                        end

                        % Shuffle training data
                        idx_train = idx_train(randperm(length(idx_train)));
                        train_in = (dlarray(data_in(idx_train,:,:),"BCT"));
                        train_out = (dlarray(data_out(idx_train,:,:),"BCT"));

                        for i=1:floor(length(train_in(1,:,1)))/(Batch_size)

                            %loss grad
                            if i*Batch_size>(length(train_in(1,:)))
                                % [loss, gradients] = dlfeval(@modelLoss_interval, param, pre_net_param, net,(train_in(:,1+(i-1)*Batch_size:i*Batch_size,:)), (train_out(:,1+(i-1)*Batch_size:i*Batch_size,:)), loss_type);
                            else
                                [loss, gradients] = dlfeval(@modelLoss_interval, param_delta, pre_net_param, net,(train_in(:,1+(i-1)*Batch_size:i*Batch_size,:)), (train_out(:,1+(i-1)*Batch_size:i*Batch_size,:)), loss_type, coverage, penalty, output_lag,activation);
                            end

                            %update net
                            [param_delta,averageGrad,averageSqGrad] = adamupdate(param_delta,gradients,averageGrad,averageSqGrad,iter,learnRate);

                            % addpoints(h, iter, (loss));

                            batch_loss = batch_loss + loss;

                            iter = iter+1;

                        end

                        batch_loss = batch_loss/floor((length(train_in(1,:,1)))/(Batch_size));
                        % addpoints(h, epoch, (batch_loss));
                        % drawnow;

                        %validation set
                        out0_val=val_out(:,:,1);

                        % Forward propogation
                        [pred_upper_val, pred_lower_val] = forward_interval(param_delta, pre_net_param, net, val_in, out0_val, output_lag,activation);

                        %Loss calculation
                        if strcmp(loss_type, 'pinball')
                            % Call the pinball loss function
                            loss_val = pinball_loss(pred_upper_val, pred_lower_val, val_out(:,:,2:end), coverage, penalty);
                        elseif strcmp(loss_type, 'rqr')
                            % Call the rqrw loss function
                            loss_val = rqrw_loss(pred_upper_val, pred_lower_val, val_out(:,:,2:end), coverage, penalty);
                        else
                            % Handle cases where an invalid loss_type is provided
                            error('Unknown loss type: %s. Please use ''pinball'' or ''rqr''.', loss_type);
                        end
                        %
                        % addpoints(h2, epoch, (loss_val));
                        % drawnow;


                        LOSStrain(epoch)=batch_loss;
                        LOSSval(epoch)=loss_val;

                        if(mod(epoch,1)==0)
                            disp(['|epoch--- ',num2str(epoch),'    |loss--- ',num2str(double(extractdata(gather(batch_loss)))),'    |val_loss--- ',num2str(double(extractdata(gather(loss_val))))])
                        end

                        % Early stopping
                        [counter,min_val_loss,early_indicator] = early_stopping(loss_val,patience,min_delta,counter,min_val_loss);

                        if min_val_loss == loss_val
                            best_net=param_delta;
                        end

                        if early_indicator==1
                            break;
                        end

                    end

                    LOSS_INTERVAL = struct();
                    LOSS_INTERVAL.train = LOSStrain;
                    LOSS_INTERVAL.val = LOSSval;

                    %%

                    % Test on all data -- train val test
                    % param_delta = load("Interval_net_delta_seed_1_noise_0_loss_rqr_coverage_0.9_penalty_0.5.mat");
                    % param_delta = param_delta.best_net;

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




                    %%
                    % Save Nets%Plots

                    % Directory to save
                    % outputDir = activation;
                    % %
                    % % Save best_net
                    % saveNetwork(outputDir, seed, noiseLevel, best_net, loss_type, coverage, penalty)
                    %
                    %
                    % % Save Loss
                    % saveLOSS(outputDir, seed, noiseLevel, LOSS_INTERVAL, loss_type, coverage, penalty)

                    % Save the first figure
                    % figTitle = 'Interval_test_plot';
                    %  saveFigure(outputDir, seed, noiseLevel, figTitle, loss_type, coverage, penalty)
                    % close(2);
                    %
                    % % Save the second figure
                    % figTitle = 'Interval_Loss';
                    % saveFigure(outputDir, seed, noiseLevel, figTitle, loss_type, coverage, penalty)
                    % close(1);




                end
            end
        end
    end
end

%% Helper functions for Interval LSTM


function saveNetwork(outputDir, seed, noiseLevel, best_net, loss_type, coverage, penalty)

% Create the filename based on seed and noise level
filename = fullfile(outputDir, ['Interval_net_delta_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '_loss_', char(loss_type), '_coverage_', num2str(coverage), '_penalty_', num2str(penalty), '.mat']);

% Save the network (overwrites if file with the same name already exists)
save(filename, 'best_net');

end

function saveFigure(outputDir, seed, noiseLevel, figTitle, loss_type, coverage, penalty)

% Format the figure title for the filename
figTitle = strrep(figTitle, ' ', '_'); % Replace spaces with underscores

% Create the filename based on figTitle, seed, and noise level
filename = fullfile(outputDir, [figTitle, num2str(seed), '_noise_', num2str(noiseLevel), '_loss_', char(loss_type), '_coverage_', num2str(coverage), '_penalty_', num2str(penalty), '.fig']);

% Save the current figure as a .fig file (overwrites if file with the same name already exists)
saveas(gcf, filename);
end

function saveLOSS(outputDir, seed, noiseLevel, LOSS_INTERVAL, loss_type, coverage, penalty)

% Create the filename based on seed and noise level
filename = fullfile(outputDir, ['loss_interval_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '_loss_', char(loss_type), '_coverage_', num2str(coverage), '_penalty_', num2str(penalty), '.mat']);

% Save the network (overwrites if file with the same name already exists)
save(filename, 'LOSS_INTERVAL');

end


