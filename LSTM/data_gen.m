
function [train_in,train_out,train_in_all,train_out_all,val_in,val_out,test_in,test_out,statistics,idx_train,data_in,data_out] = data_gen(U,X,window_length,p_train,p_val,normalization_type)


        statistics = struct();
      
        if normalization_type == "min_max"
        % Min-max normalization
        % Upper lower calulation
        lower_in = min(U(:,1:floor(p_train*end)),[],"all");
        upper_in = max(U(:,1:floor(p_train*end)),[],"all");
        lower_out = min(X(:,1:floor(p_train*end)),[],"all");
        upper_out = max(X(:,1:floor(p_train*end)),[],"all");


         % Train set
        X_train = ((X(:,1:floor(p_train*end))-lower_out)./(upper_out-lower_out))';%output
        U_train = ((U(:,1:floor(p_train*end))-lower_in)./(upper_in-lower_in))';%input
        % Val set
        X_val = ((X(:,floor(p_train*end)+1:floor((p_train+p_val)*end))-lower_out)./(upper_out-lower_out))';
        U_val = ((U(:,floor(p_train*end)+1:floor((p_train+p_val)*end))-lower_in)./(upper_in-lower_in))';
        % Test set
        X_test = ((X(:,floor((p_train+p_val)*end)+1:floor(end))-lower_out)./(upper_out-lower_out))';%(Xt'-mean_out)/std_out;
        U_test = ((U(:,floor((p_train+p_val)*end)+1:floor(end))-lower_in)./(upper_in-lower_in))'; %(Ut'-mean_in)/std_out;

        statistics.lower_out = lower_out;
        statistics.upper_out = upper_out;
        end
        
        if normalization_type =="z_score"
        % Z-score normalization
        % Mean and std calculation
        mean_in = mean(U(:,1:floor(p_train*end)),2);
        std_in = std(U(:,1:floor(p_train*end)),[],2);
        mean_out = mean(X(:,1:floor(p_train*end)),2);
        std_out = std(X(:,1:floor(p_train*end)),[],2);

        % Train set
        X_train = ((X(:,1:floor(p_train*end))-mean_out)./std_out)';%output
        U_train = ((U(:,1:floor(p_train*end))-mean_in)./std_in)';%input
        % Val set
        X_val = ((X(:,floor(p_train*end)+1:floor((p_train+p_val)*end))-mean_out)./std_out)';
        U_val = ((U(:,floor(p_train*end)+1:floor((p_train+p_val)*end))-mean_in)./std_in)';
        % Test set
        X_test = ((X(:,floor((p_train+p_val)*end)+1:floor(end))-mean_out)./std_out)';%(Xt'-mean_out)/std_out;
        U_test = ((U(:,floor((p_train+p_val)*end)+1:floor(end))-mean_in)./std_in)'; %(Ut'-mean_in)/std_out;
        
        statistics.mean_out = mean_out;
        statistics.std_out = std_out;
        end

        % Create batch of examples from training set/Windowing

        % Specify time horizon
        t = linspace(0, 15, window_length);

        numberofdatax = length(X_train(:,1))-window_length;
        idx = 1:1:numberofdatax;%randperm(size(X_tank,2)-window_length-1,numberofdata);
        numberofdata = length(idx);

        for i=1:numberofdata
            data_in(i,:,:) = dlarray(U_train(idx(i):idx(i)+window_length-1,:)');
            data_out(i,:,:) = dlarray(X_train(idx(i):idx(i)+window_length-1,:)');
        end

        idx_train=(1:floor(numberofdata));
        
        % Train set --windowed version
        train_in = (dlarray(data_in(idx_train,:,:),"BCT"));
        train_out = (dlarray(data_out(idx_train,:,:),"BCT"));

        train_in_all = dlarray(U_train,"TCB");
        train_out_all = dlarray(X_train,"TCB");

        val_in = dlarray(U_val,"TCB");
        val_out = dlarray(X_val,"TCB");

        test_in = dlarray(U_test,"TCB");
        test_out = dlarray(X_test,"TCB");

end
