
clear all
close all
clc

% Model features
seedx = [1 2 3 4 5];
noiseLevelx = 0;%[0.0 0.2];
normalization_type = "z_score";

% ny-nx-nd
output_lag = 1;
input_lag = 2;
dead_time = 0;

% Split rates
p_train = 2/3*0.8;
p_val = 2/3*0.2;

% Window Length
window_length=80;

for nn=1:length(noiseLevelx)
    for mm=1:length(seedx)

        seed = seedx(mm);
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

        % Dead time nd
        U=[zeros(1,dead_time) U(1,1:end-dead_time)];
        % Input Lag u(k),u(k-1),...
        U=input_lagged(U,input_lag);

        X=X+randn(size(X))*0;

        %% Train-Val-Test Split

        [train_in,train_out,train_in_all,train_out_all,val_in,val_out,test_in,test_out,statistics,idx_train,data_in,data_out] = data_gen(U,X,window_length,p_train,p_val,normalization_type);

        %%

        % NN create/train

        hidden_size=48;

        net = [sequenceInputLayer(size(train_in,1)+output_lag)
            lstmLayer(hidden_size,OutputMode="sequence")
            lstmLayer(hidden_size,OutputMode="sequence")
            % lstmLayer(hidden_size*0.5,OutputMode="sequence",Name="lstm")
            % lstmLayer(hidden_size,OutputMode="sequence",Name="lstm")
            fullyConnectedLayer(size(train_out,1),Name="mean")
            ];

        net = dlnetwork(net);
        net=initialize(net);


        %% Training/fitting

        % Create for loop for seed iterations

        rng(seed);

        %Training parameters
        gradDecay = 0.99;
        sqGradDecay = 0.999;
        numIter = 1200;
        plotFrequency = 50;

        % Hyperparameters
        learnRate=5*1e-3;
        velocity=[];
        Batch_size=128;
        gradients=[];
        averageGrad=[];
        averageSqGrad=[];

        %early stopping paramters
        counter=0;
        min_val_loss=inf;
        patience=20;
        min_delta=0.001;

        %
        figure();
        h = animatedline();
        h2 = animatedline("Color","r");
        legend("train","validation")
        xlabel("iter")
        ylabel('Loss');
        ax.YGrid = 'on';
        ax.XGrid = 'on';


        %animated line for live plot of test
        % Initialize the figures and animated lines
        N = size(X,1); % Output size
        % figure;
        % for i=1:N
        %     subplot(N,1,i);
        %     % Dynamically create the animated line objects for each subplot
        %     eval(['h1_true_' num2str(i) ' = animatedline(''Color'', ''k'', ''LineWidth'', 1.2);']);
        %     eval(['h1_pred_' num2str(i) ' = animatedline(''Color'', [0, 0, 0.5], ''LineWidth'', 1.2);']);
        %     legend("true", "pred");
        %     title("y"+i);
        %     hold on;
        % end


        iter=1;
        for epoch = 1:50
            batch_loss=0;

            % Learning rate scheduler
            if(mod(epoch, 20) == 0)
                learnRate = learnRate/2;
            end

            % Shuffle training data
            idx_train = idx_train(randperm(length(idx_train)));
            train_in = (dlarray(data_in(idx_train,:,:),"BCT"));
            train_out = (dlarray(data_out(idx_train,:,:),"BCT"));

            for i=1:floor(length(train_in(1,:,1)))/(Batch_size)

                % Loss and grad calculation
                if i*Batch_size>(length(train_in(1,:))) % Do not include last part if < batch_size
                    % [loss,gradients] = dlfeval(@modelLoss,net,(train_in(:,1+(i-1)*Batch_size:i*Batch_size,:)),(train_out(:,1+(i-1)*Batch_size:i*Batch_size,:)));
                else
                    % Evaluate network and compute loss and gradients
                    [loss,gradients] = dlfeval(@modelLoss,net,(train_in(:,1+(i-1)*Batch_size:i*Batch_size,:)),(train_out(:,1+(i-1)*Batch_size:i*Batch_size,:)),output_lag);
                end

                % Update net
                [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iter,learnRate);

                batch_loss = batch_loss + loss;

                iter = iter+1;

            end

            batch_loss = batch_loss/floor((length(train_in(1,:,1)))/(Batch_size));

            addpoints(h, epoch, (batch_loss));

            %validation set
            val_target0= val_out(:,:,1);

            % Compute predictions.
            out_pred_val = forward_custom(net,val_in,val_target0,output_lag);

            % Compute L2 loss.
            loss_val = l2loss(out_pred_val,val_out(:,:,2:end),'DataFormat',"CBT",NormalizationFactor="all-elements" );

            addpoints(h2, epoch, (loss_val));
            drawnow;

            LOSStrain(epoch)=batch_loss;
            LOSSval(epoch)=loss_val;

            % Display train val loss
            if(mod(epoch,1)==0)
                disp(['|epoch--- ',num2str(epoch),'    |loss--- ',num2str(double(extractdata(gather(batch_loss)))),'    |val_loss--- ',num2str(double(extractdata(gather(loss_val))))])
            end

            % Early stopping
            [counter,min_val_loss,early_indicator] = early_stopping(loss_val,patience,min_delta,counter,min_val_loss);

            if min_val_loss == loss_val
                best_net=net;
            end

            if early_indicator==1
                break;
            end

            % Plot validation data

            % % Evaluate at the current epoch
            % y0_test = val_out(:,:,1);
            % y_test = forward_custom(best_net,val_in,y0_test);
            %
            % % Extract true values and predictions
            % y_true = squeeze(extractdata(val_out(:,:,2:end)));
            % y_pred = squeeze(extractdata(y_test(:,:,:)));
            %
            %
            % % Update the subplot
            % for j=1:N
            %
            %     subplot(N,1,j); % Switch subplot
            %     eval(['clearpoints(h1_true_' num2str(j) ');']);
            %     eval(['clearpoints(h1_pred_' num2str(j) ');']);
            %     % Add data points for the true and predicted values
            %     eval(['addpoints(h1_true_' num2str(j) ', 1:size(y_true,1), y_true(:,j));']);
            %     eval(['addpoints(h1_pred_' num2str(j) ', 1:size(y_pred,1), y_pred(:,j));']);
            %     drawnow;
            %
            % end


        end

        % close;
        %%

        %Loss plot

        % figure('Visible', 'off');
        % plot(LOSS);
        % hold on;
        % plot(LOSSval);
        % legend("training loss","val loss")
        % title("Loss plot")
        % xlabel("epoch")
        LOSS = struct();
        LOSS.train = LOSStrain;
        LOSS.val = LOSSval;

        %% Test Network on all data

        % Get Net&params
        net_final = best_net;

        % Separate targets for prediction in each set
        train_target0 = train_out_all(:,:,1);
        val_target0 = val_out(:,:,1);
        test_target0 = test_out(:,:,1);

        % Predictions for each set
        out_pred_train = forward_custom(net_final, train_in_all, train_target0,output_lag);
        out_pred_val = forward_custom(net_final, val_in, val_target0,output_lag);
        out_pred_test = forward_custom(net_final, test_in, test_target0,output_lag);

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


        %%

        % Save Nets%Plots

        % Directory to save
        % outputDir = 'results&nets';
        % %
        % % % Save best_net
        % saveNetwork(outputDir, seed, noiseLevel, best_net);
        % %
        % % %save loss
        % saveLOSS(outputDir, seed, noiseLevel, LOSS)

        % % Save the first figure
        % figTitle = 'Crisp_test_plot';
        % saveFigure(outputDir, seed, noiseLevel, figTitle)
        % close(2);
        %
        % % Save the second figure
        % figTitle = 'Crisp_Loss';
        % saveFigure(outputDir, seed, noiseLevel, figTitle)
        % close(1);


    end

end
%% Helper Functions for LSTM



function saveNetwork(outputDir, seed, noiseLevel, best_net)

% Create the filename based on seed and noise level
filename = fullfile(outputDir, ['pre_net_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '.mat']);

% Save the network (overwrites if file with the same name already exists)
save(filename, 'best_net');

end

function saveFigure(outputDir, seed, noiseLevel, figTitle)

% Format the figure title for the filename
figTitle = strrep(figTitle, ' ', '_'); % Replace spaces with underscores

% Create the filename based on figTitle, seed, and noise level
filename = fullfile(outputDir, [figTitle, '_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '.fig']);

% Save the current figure as a .fig file (overwrites if file with the same name already exists)
saveas(gcf, filename);
end

function saveLOSS(outputDir, seed, noiseLevel, LOSS)

% Create the filename based on seed and noise level
filename = fullfile(outputDir, ['loss_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '.mat']);

% Save the network (overwrites if file with the same name already exists)
save(filename, 'LOSS');

end








