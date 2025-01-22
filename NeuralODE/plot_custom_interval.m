function plot_custom_interval(plot_var)


train_out_all = plot_var.train_out_all;
val_out = plot_var.val_out;
test_out = plot_var.test_out;

pred_mean_train = plot_var.pred_mean_train;
pred_mean_val = plot_var.pred_mean_val;
pred_mean_test = plot_var.pred_mean_test;

pred_upper_train = plot_var.pred_upper_train;
pred_upper_val = plot_var.pred_upper_val;
pred_upper_test = plot_var.pred_upper_test;

pred_lower_train = plot_var.pred_lower_train;
pred_lower_val = plot_var.pred_lower_val;
pred_lower_test = plot_var.pred_lower_test;



% Plotting predictions vs true values for y1
figure;
hold on;

% Define colors
color_train = [0, 0.45, 0.75];   % Light blue for training data
color_val = [0.3, 0.6, 1];  % Darker blue for validation data
color_test = [0.2, 0.8, 1]; % Bright cyan/teal for test data


N=size(train_out_all,1);% Output size
for i=1:N

    % Plot true values for train, val, and test data
    subplot(N,1,i);
    hold on;

    % Plot true values for train, val, and test data
    subplot(N,1,i);
    hold on;


    % Define the x-coordinates for each section
    x_train = 1:size(train_out_all,3)-1;
    x_val = size(train_out_all,3)+1:(size(train_out_all,3) + size(val_out,3) - 1);
    x_test = (size(train_out_all,3)+1 + size(val_out,3)):(size(train_out_all,3) + size(val_out,3) + size(test_out,3) - 1);

    % Fill the area between the upper and lower bounds for the training section
    % Extract and prepare data for upper and lower bounds
    upper_bound = extractdata(squeeze(pred_upper_train(:,i)))';
    lower_bound = extractdata(squeeze(pred_lower_train(:,i)))';

    % Fill the area between the bounds
    fill([x_train, fliplr(x_train)], [upper_bound, fliplr(lower_bound)], ...
        [0.95, 0.95, 0.95], 'EdgeColor', 'none', 'FaceAlpha', 1);


    % Fill the area between the upper and lower bounds for the validation section
    % Extract and prepare data for upper and lower bounds
    upper_bound1 = extractdata(squeeze(pred_upper_val(:,i)))';
    lower_bound1 = extractdata(squeeze(pred_lower_val(:,i)))';

    % Fill the area between the bounds
    fill([x_val, fliplr(x_val)], [upper_bound1, fliplr(lower_bound1)], ...
        [0.95, 0.95, 0.95], 'EdgeColor', 'none', 'FaceAlpha', 1);

    % Fill the area between the upper and lower bounds for the test section
    % Extract and prepare data for upper and lower bounds
    upper_bound2 = extractdata(squeeze(pred_upper_test(:,i)))';
    lower_bound2 = extractdata(squeeze(pred_lower_test(:,i)))';
    % Fill the area between the bounds
    fill([x_test, fliplr(x_test)], [upper_bound2, fliplr(lower_bound2)], ...
        [0.95, 0.95, 0.95], 'EdgeColor', 'none', 'FaceAlpha', 1);

    plot(1:size(train_out_all,3)-1, squeeze(extractdata(train_out_all(i,:,2:end))), 'Color', color_train, 'DisplayName', 'True Train');
    plot(size(train_out_all,3)+1:(size(train_out_all,3) + size(val_out,3)-1), squeeze(extractdata(val_out(i,:,2:end))), 'Color', color_val, 'DisplayName', 'True Val');
    plot((size(train_out_all,3) + size(val_out,3) + 1):(size(train_out_all,3) + size(val_out,3) + size(test_out,3) - 1), squeeze(extractdata(test_out(i,:,2:end))), 'Color', color_test, 'DisplayName', 'True Test');

    % Plot mean predictions for train, val, and test data
    plot(1:size(train_out_all,3)-1, squeeze(pred_mean_train(:,i)), '--', 'Color', color_train, 'DisplayName', 'Pred Train');
    plot(size(train_out_all,3)+1:(size(train_out_all,3) + size(val_out,3) - 1), squeeze(pred_mean_val(:,i)), '--', 'Color', color_val, 'DisplayName', 'Pred Val');
    plot((size(train_out_all,3) + size(val_out,3) + 1):(size(train_out_all,3) + size(val_out,3) + size(test_out,3) - 1), squeeze(pred_mean_test(:,i)), '--', 'Color', color_test, 'DisplayName', 'Pred Test');

    % Add vertical lines to separate train, val, and test sets
    xline(size(train_out_all, 3) - 0.5, 'k--', 'LineWidth', 1.5); % Separation between Train and Val
    xline(size(train_out_all, 3) + size(val_out, 3) - 0.5, 'k--', 'LineWidth', 1.5); % Separation between Val and Test

    % Add PICP annotations for each part
    text_x_positions = [size(train_out_all,3) / 2, ...
        size(train_out_all,3) + size(val_out,3) / 2, ...
        size(train_out_all,3) + size(val_out,3) + size(test_out,3) / 2];

    max_y_value = max([squeeze(train_out_all(i,:,2:end)); squeeze(val_out(i,:,2:end)); squeeze(test_out(i,:,2:end))], [], 'all');
    text_y_position = extractdata(max_y_value) * 1.1;


    % Compute PICP for each set
    picp_train = PICP(squeeze(extractdata(train_out_all(i,:,2:end))), ...
        squeeze(pred_lower_train(:,:)), ...
        squeeze(pred_upper_train(:,:))); % Assuming you have upper/lower for train
    picp_val = PICP(squeeze(extractdata(val_out(i,:,2:end))), ...
        squeeze(pred_lower_val(:,:)), ...
        squeeze(pred_upper_val(:,:))); % Assuming you have upper/lower for val
    picp_test = PICP(squeeze(extractdata(test_out(:,:,2:end))), ...
        squeeze(pred_lower_test(:,:)), ...
        squeeze(pred_upper_test(:,:))); % Assuming you have upper/lower for test

    % Compute PINAW for each set
    pinaw_train = PINAW(squeeze(extractdata(train_out_all(:,:,2:end))), ...
        squeeze(pred_lower_train(:,:)), ...
        squeeze(pred_upper_train(:,:))); % Assuming you have upper/lower for train
    pinaw_val = PINAW(squeeze(extractdata(val_out(:,:,2:end))), ...
        squeeze(pred_lower_val(:,:)), ...
        squeeze(pred_upper_val(:,:))); % Assuming you have upper/lower for val
    pinaw_test = PINAW(squeeze(extractdata(test_out(:,:,2:end))), ...
        squeeze(pred_lower_test(:,:)), ...
        squeeze(pred_upper_test(:,:))); % Assuming you have upper/lower for test

    text(text_x_positions(1), text_y_position, sprintf('PICP Train: %.2f --- PINAW Train: %.2f', picp_train, pinaw_train), 'HorizontalAlignment', 'center');
    text(text_x_positions(2), text_y_position, sprintf('PICP val: %.2f --- PINAW val: %.2f', picp_val, pinaw_val), 'HorizontalAlignment', 'center');
    text(text_x_positions(3), text_y_position, sprintf('PICP test: %.2f --- PINAW test: %.2f', picp_test, pinaw_test), 'HorizontalAlignment', 'center');

    if N>1
        title("y" + i);
    end



end

sgtitle("Interval Predictions vs True Values with PICP");
hold off;

end
