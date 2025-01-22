
clear all
close all
clc

% Model
seedx = [1 2 3 4 5];
noiseLevelx = 0;%[0.0 0.2];
normalization_type = "z_score";

% ny-nx-nd
output_lag = 1;
input_lag = 2;
dead_time=0;

% Window length
window_length=40;

% Split rates
p_train = 2/3*0.8;
p_val = 2/3*0.2;


for nn=1:length(noiseLevelx)
    for mm=1:length(seedx)

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

        U=[zeros(1,dead_time) U(1,1:end-dead_time)];
        U=input_lagged(U,input_lag);

        % Model parameters
        noiseLevel = noiseLevelx(nn); % Noise level
        seed = seedx(mm);

        % Add noise to model
        X=X+randn(size(X))*noiseLevel;

        %% Train-Val-Test Split


        [train_in,train_out,train_in_all,train_out_all,val_in,val_out,test_in,test_out,statistics,idx_train,data_in,data_out] = data_gen(U,X,window_length,p_train,p_val,normalization_type);

        %%
        %Model parameters init
        neuralOdeTimesteps = window_length;
        dt = 1;
        timesteps = 1:1:neuralOdeTimesteps;%(0:neuralOdeTimesteps-1)*dt;

        neuralOdeParameters = struct;

        inSize = input_lag+1+output_lag;%size(X,1)+size(U,1);
        out_size = size(X,1);
        hiddenSize = 32;
        % hiddenSize2 = 10;
        % hiddenSize3 = 32;

        % Create Network
        neuralOdeParameters.fc1 = struct;
        sz = [hiddenSize inSize];
        neuralOdeParameters.fc1.Weights = initializeGlorot(sz, hiddenSize, inSize);
        neuralOdeParameters.fc1.Bias = initializeZeros([hiddenSize 1]);

        neuralOdeParameters.fc2 = struct;
        sz = [out_size hiddenSize];
        neuralOdeParameters.fc2.Weights = initializeGlorot(sz, out_size, hiddenSize);
        neuralOdeParameters.fc2.Bias = initializeZeros([out_size 1]);
        %
        % neuralOdeParameters.fc3 = struct;
        % sz =  [out_size hiddenSize2];
        % neuralOdeParameters.fc3.Weights = initializeGlorot(sz, out_size, hiddenSize2);
        % neuralOdeParameters.fc3.Bias = initializeZeros([out_size 1]);

        % neuralOdeParameters.fc4 = struct;
        % sz = [out_size hiddenSize3];
        % neuralOdeParameters.fc4.Weights = initializeGlorot(sz, out_size, hiddenSize3);
        % neuralOdeParameters.fc4.Bias = initializeZeros([out_size 1]);



        %% Training/fitting

        % Create for loop for seed iterations

        rng(seed);

        %Training parameters
        gradDecay = 0.99;
        sqGradDecay = 0.999;
        numIter = 1200;
        plotFrequency = 50;


        % Hyperparameters
        learnRate=1*1e-3;
        velocity=[];
        Batch_size=64;
        gradients=[];
        averageGrad=[];
        averageSqGrad=[];

        %early stopping paramters
        counter=0;
        min_val_loss=inf;
        patience=100;
        min_delta=0.0005;

        %
        figure();
        h = animatedline();
        h2 = animatedline("Color","r");
        legend("train","validation")
        xlabel("iter")
        ylabel('Loss');
        ax.YGrid = 'on';
        ax.XGrid = 'on';


        % animated line for live plot of test
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
        for epoch = 1:100
            batch_loss=0;

            % Learning rate scheduler
            % if(mod(epoch, 75) == 0)
            %     learnRate = learnRate/2;
            % end

            % Shuffle training data
            idx_train = idx_train(randperm(length(idx_train)));
            train_in = (dlarray(data_in(idx_train,:,:),"BCT"));
            train_out = (dlarray(data_out(idx_train,:,:),"BCT"));

            for i=1:floor(length(train_in(1,:,1)))/(Batch_size)

                % Loss and grad calculation
                if i*Batch_size>(length(train_in(1,:,1))) % Do not include last part if < batch_size
                    % [loss,gradients] = dlfeval(@modelLoss,timesteps,(train_in(:,1+(i-1)*Batch_size:i*Batch_size,:)),neuralOdeParameters,(train_out(:,1+(i-1)*Batch_size:i*Batch_size,:)));
                else
                    % Evaluate network and compute loss and gradients
                    [loss,gradients] = dlfeval(@modelLoss,timesteps,(train_in(:,1+(i-1)*Batch_size:i*Batch_size,:)),neuralOdeParameters,(train_out(:,1+(i-1)*Batch_size:i*Batch_size,:)), output_lag);
                end

                % Update net
                [neuralOdeParameters,averageGrad,averageSqGrad] = adamupdate(neuralOdeParameters,gradients,averageGrad,averageSqGrad,iter,learnRate);

                batch_loss = batch_loss + loss;

                iter = iter+1;

            end

            batch_loss = batch_loss/floor((length(train_in(1,:,1)))/(Batch_size));

            addpoints(h, epoch, (batch_loss));
            % drawnow;

            %validation set - early stopping
            y0 = val_out(:,:,1);
            y_val = euler_forward(@odeModel,1:1:size(val_in,3),y0,val_in,neuralOdeParameters, 1e-1, output_lag);

            loss_val = l2loss(y_val,permute(stripdims(val_out(:,:,2:end)),[3 1 2]),'DataFormat',"TCB","NormalizationFactor","all-elements");
            %
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
                best_net=neuralOdeParameters;
            end

            if early_indicator==1
                break;
            end


            % % Evaluate at the current epoch
            % y0_test = test_out(:,:,1);
            % y_test = euler_forward(@odeModel,1:1:size(test_in,3), (y0_test), test_in, neuralOdeParameters, 1e-1);
            %
            % % Extract true values and predictions
            % y_true = squeeze(extractdata(test_out(:,:,2:end)));
            % y_pred = squeeze(extractdata(y_test(:,:)));
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



        %%

        %Loss plot

        % figure;
        % plot(LOSStrain);
        % hold on;
        % plot(LOSSval);
        % legend("training loss","val loss")
        % title("Loss plot")
        % xlabel("epoch")
        LOSS = struct();
        LOSS.train = LOSStrain;
        LOSS.val = LOSSval;

        %% Test Network on all data

        % % Get Network/parameters
        neuralOdeParameters = best_net;
        %

        plot_var = struct(); % Store plot variables

        % Separate targets for prediction in each set
        train_target0 = train_out_all(:,:,1);

        val_target0 = val_out(:,:,1);
        test_target0 = test_out(:,:,1);

        % Predictions for each set
        plot_var.out_pred_train = euler_forward(@odeModel,1:1:size(train_out_all,3),dlarray((permute(stripdims(train_target0),[3 1 2])),"TCB"),train_in_all,neuralOdeParameters, 1e-1, output_lag);
        plot_var.out_pred_val = euler_forward(@odeModel,1:1:size(val_out,3),dlarray((permute(stripdims(val_target0),[3 1 2])),"TCB"),val_in,neuralOdeParameters, 1e-1, output_lag);
        plot_var.out_pred_test = euler_forward(@odeModel,1:1:size(test_out,3),dlarray((permute(stripdims(test_target0),[3 1 2])),"TCB"),test_in,neuralOdeParameters, 1e-1, output_lag);

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







        %%
        % Save Nets%Plots

        % Directory to save
        outputDir = 'results&nets';

        % Save best_net
        saveNetwork(outputDir, seed, noiseLevel, best_net);

        %save loss
        saveLOSS(outputDir, seed, noiseLevel, LOSS)

        % Save the first figure
        % figTitle = 'Crisp_test_plot';
        % saveFigure(outputDir, seed, noiseLevel, figTitle)
        % close(2);

        % Save the second figure
        % figTitle = 'Crisp_Loss';
        % saveFigure(outputDir, seed, noiseLevel, figTitle)
        % close(1);

    end
end

%% Helper Functions

function saveNetwork(outputDir, seed, noiseLevel, best_net)

% Create the filename based on seed and noise level
filename = fullfile(outputDir, ['pre_net_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '.mat']);

% Save the network (overwrites if file with the same name already exists)
save(filename, 'best_net');

end


function saveLOSS(outputDir, seed, noiseLevel, LOSS)

% Create the filename based on seed and noise level
filename = fullfile(outputDir, ['loss_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '.mat']);

% Save the network (overwrites if file with the same name already exists)
save(filename, 'LOSS');

end

function saveFigure(outputDir, seed, noiseLevel, figTitle)

% Format the figure title for the filename
figTitle = strrep(figTitle, ' ', '_'); % Replace spaces with underscores

% Create the filename based on figTitle, seed, and noise level
filename = fullfile(outputDir, [figTitle, '_seed_', num2str(seed), '_noise_', num2str(noiseLevel), '.fig']);

% Save the current figure as a .fig file (overwrites if file with the same name already exists)
saveas(gcf, filename);
end


