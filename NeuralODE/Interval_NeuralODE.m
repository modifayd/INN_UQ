
clear all
close all
clc

% Model features
seedx = [1 2 3 4 5];
noiseLevelx = 0;%[0.0 0.2];
loss_typex = "rqr";%["pinball","rqr"];
Penaltyx = 0.5;%[0 0.5];
Coveragex = 0.9;%[0.9 0.95];
window_length=40;
normalization_type ="z_score";

% ny-nx-nd
output_lag = 1;
input_lag = 2;
dead_time=0;

% Activation -- relu, abs
activation="abs";

% Split rates
p_train = 2/3*0.8;
p_val = 2/3*0.2;

% LP Uncertainity rate
output_uncertainity_rate=0.75; % Output layer delta magnitude level
hidden_uncertainity_rate=0.75; % Hidden layers delta magnitude level


for nn=1:length(noiseLevelx)
    for mm=1:length(seedx)
        for cc=1:length(loss_typex)
            for ii=1:length(Coveragex)
                for jj=1:length(Penaltyx)

                    rng(15) % 
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

                    % Data gen
                    [train_in,train_out,train_in_all,train_out_all,val_in,val_out,test_in,test_out,statistics,idx_train,data_in,data_out] = data_gen(U,X,window_length,p_train,p_val,normalization_type);

                    %%

                    % Load pretrained param
                    net = load("pre_net_seed_"+num2str(seed)+"_noise_"+num2str(noiseLevel)+".mat");
                    pre_trained_net = net.best_net;

                    % Create time axis for neural ode
                    neuralOdeTimesteps = window_length;
                    dt = 1;%t(2);
                    timesteps = (0:neuralOdeTimesteps-1)*dt;

                    % Create upper lower Weights&Biases

                    neuralOdeParameters_delta = init_delta_param(X, U, pre_trained_net, hidden_uncertainity_rate, output_uncertainity_rate);

                    %%

                    rng(seed); % fixed randomness for training

                    %Training parameters
                    gradDecay = 0.99;
                    sqGradDecay = 0.999;
                    numIter = 1200;
                    plotFrequency = 50;


                    % Hyperparameters
                    learnRate=1*1e-2;
                    velocity=[];
                    Batch_size=64;
                    gradients=[];
                    averageGrad=[];
                    averageSqGrad=[];

                    %early stopping paramters
                    counter=0;
                    min_val_loss=inf;
                    patience=5;
                    min_delta=0.005;
                    %%
                    %
                    % figure();
                    % h = animatedline();
                    % h2 = animatedline("Color","r");
                    % legend("train","validation")
                    % xlabel("iter")
                    % ylabel('Loss');
                    % ax.YGrid = 'on';
                    % ax.XGrid = 'on';

                    iter=1;

                    for epoch = 1:30

                        batch_loss=0;

                        % Learning rate scheduler
                        if(mod(epoch, 20) == 0)
                            learnRate = learnRate*0.1;
                        end

                        % Shuffle training data
                        idx_train = idx_train(randperm(length(idx_train)));
                        train_in = (dlarray(data_in(idx_train,:,:),"BCT"));
                        train_out = (dlarray(data_out(idx_train,:,:),"BCT"));

                        for i=1:floor(length(train_in(1,:,1)))/(Batch_size)

                            %loss grad
                            if i*Batch_size>(length(train_in(1,:)))  % Do not include last part if < batch_size
                                % [loss,gradients] = dlfeval(@modelLoss,timesteps,(train_in(:,1+(i-1)*Batch_size:i*Batch_size,:)),neuralOdeParameters_delta,pre_trained_net,(train_out(:,1+(i-1)*Batch_size:i*Batch_size,:)), loss_type);
                            else
                                % Evaluate network and compute loss and gradients
                                [loss,gradients] = dlfeval(@modelLoss_interval,timesteps,(train_in(:,1+(i-1)*Batch_size:i*Batch_size,:)),neuralOdeParameters_delta,pre_trained_net,(train_out(:,1+(i-1)*Batch_size:i*Batch_size,:)), loss_type, coverage, penalty,output_lag, activation);
                            end

                            %update net
                            [neuralOdeParameters_delta,averageGrad,averageSqGrad] = adamupdate(neuralOdeParameters_delta,gradients,averageGrad,averageSqGrad,iter,learnRate);

                            batch_loss = batch_loss + loss;

                            iter = iter+1;

                        end

                        batch_loss = batch_loss/floor((length(train_in(1,:,1)))/(Batch_size));

                        % addpoints(h, epoch, (batch_loss));

                        %validation set - early stopping
                        neuralOdeParameters_new = combined_param(neuralOdeParameters_delta,pre_trained_net,activation);
                        y0=val_out(:,:,1);
                        [xval_upper, xval_lower, ~] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(val_in,3), y0,val_in, neuralOdeParameters_new,pre_trained_net, 1e-5, output_lag);

                        %Loss calculation
                        if strcmp(loss_type, 'pinball')
                            % Call the pinball loss function
                            loss_val = pinball_loss(xval_upper, xval_lower, val_out, coverage, penalty);
                        elseif strcmp(loss_type, 'rqr')
                            % Call the rqrw loss function
                            loss_val = rqrw_loss(xval_upper, xval_lower, val_out, coverage, penalty);
                        else
                            % Handle cases where an invalid loss_type is provided
                            error('Unknown loss type: %s. Please use ''pinball'' or ''rqr''.', loss_type);
                        end

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
                            best_net=neuralOdeParameters_delta;
                        end

                        if early_indicator==1
                            break;
                        end

                    end


                    LOSS_INTERVAL = struct();
                    LOSS_INTERVAL.train = LOSStrain;
                    LOSS_INTERVAL.val = LOSSval;


                    %%
                    % Evaluate on all data train val test

                    % Combine param_delta and pre_net
                    neuralOdeParameters_new_test = combined_param(best_net,pre_trained_net,activation);

                    % Separate targets for prediction in each set
                    train_target0 = train_out_all(:,:,1);
                    val_target0 = val_out(:,:,1);
                    test_target0 = test_out(:,:,1);

                    plot_var = struct(); % Store plot variables

                    % Predictions for each set
                    [plot_var.pred_upper_train, plot_var.pred_lower_train,plot_var.pred_mean_train] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(train_out_all,3), (train_target0),train_in_all, neuralOdeParameters_new_test,pre_trained_net, 1e-5, output_lag);

                    [plot_var.pred_upper_val, plot_var.pred_lower_val,plot_var.pred_mean_val] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(val_out,3), (val_target0),val_in, neuralOdeParameters_new_test,pre_trained_net, 1e-5, output_lag);

                    [plot_var.pred_upper_test, plot_var.pred_lower_test,plot_var.pred_mean_test] = euler_forward_interval(@odeModelInterval,@odeModel, 1:1:size(test_out,3), (test_target0),test_in, neuralOdeParameters_new_test,pre_trained_net, 1e-5, output_lag);


                    % Plot

                    plot_var.train_out_all =train_out_all;
                    plot_var.val_out =val_out;
                    plot_var.test_out =test_out;

                    plot_custom_interval(plot_var)

                    %%
                    % Save Nets%Plots

                    % Directory to save
                    outputDir = activation;

                    % Save best_net
                    saveNetwork(outputDir, seed, noiseLevel, best_net, loss_type, coverage, penalty)


                    % Save Loss
                    saveLOSS(outputDir, seed, noiseLevel, LOSS_INTERVAL, loss_type, coverage, penalty)
                    %
                    % % Save the first figure
                    % figTitle = 'Interval_test_plot';
                    %  saveFigure(outputDir, seed, noiseLevel, figTitle, loss_type, coverage, penalty)
                    % % close(2);
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

%% Helper Functions




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
filename = fullfile(outputDir, [figTitle,'_seed_' ,num2str(seed), '_noise_', num2str(noiseLevel), '_loss_', char(loss_type), '_coverage_', num2str(coverage), '_penalty_', num2str(penalty), '.fig']);

% Save the current figure as a .fig file (overwrites if file with the same name already exists)
saveas(gcf, filename);
end





function saveLOSS(outputDir, seed, noiseLevel, LOSS_INTERVAL, loss_type, coverage, penalty)


% Create the filename based on seed and noise level
filename = fullfile(outputDir, ['loss_interval_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '_loss_', char(loss_type), '_coverage_', num2str(coverage), '_penalty_', num2str(penalty), '.mat']);

% Save the network (overwrites if file with the same name already exists)
save(filename, 'LOSS_INTERVAL');

end

% Loss Functions

