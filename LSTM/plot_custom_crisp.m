function plot_custom_crisp(plot_var)

% Get variables
train_out_all = plot_var.train_out_all;
val_out = plot_var.val_out;
test_out = plot_var.test_out;

out_pred_train = plot_var.out_pred_train;
out_pred_val = plot_var.out_pred_val;
out_pred_test = plot_var.out_pred_test;

mse_train = plot_var.mse_train;
mse_val = plot_var.mse_val;
mse_test = plot_var.mse_test;


% Plotting predictions vs true values for y1
figure();
hold on;

% Define colors
color_train = [0, 0.45, 0.75];   % Light blue for training data
color_val = [0.3, 0.6, 1];  % Darker blue for validation data
color_test = [0.2, 0.8, 1]; % Bright cyan/teal for test data


N=size(train_out_all,1); % Output size
for i=1:N
    % Plot true values for train, val, and test data
    subplot(N,1,i);
    hold on;
    plot(1:size(train_out_all,3)-1, squeeze(extractdata(train_out_all(i,:,2:end))), 'Color', color_train, 'DisplayName', 'True Train');
    plot(size(train_out_all,3)+1:(size(train_out_all,3) + size(val_out,3)-1), squeeze(extractdata(val_out(i,:,2:end))), 'Color', color_val, 'DisplayName', 'True Val');
    plot((size(train_out_all,3) + size(val_out,3) + 1):(size(train_out_all,3) + size(val_out,3) + size(test_out,3) - 1), squeeze(extractdata(test_out(i,:,2:end))), 'Color', color_test, 'DisplayName', 'True Test');

    % Plot predictions for train, val, and test data
    plot(1:size(train_out_all,3)-1, squeeze(out_pred_train(i,:,:)), '--', 'Color', color_train, 'DisplayName', 'Pred Train');
    plot(size(train_out_all,3)+1:(size(train_out_all,3) + size(val_out,3) - 1), squeeze(out_pred_val(i,:,:)), '--', 'Color', color_val, 'DisplayName', 'Pred Val');
    plot((size(train_out_all,3) + size(val_out,3) + 1):(size(train_out_all,3) + size(val_out,3) + size(test_out,3) - 1), squeeze(out_pred_test(i,:,:)), '--', 'Color', color_test, 'DisplayName', 'Pred Test');


    % % Add vertical lines to separate train, val, and test sets
    xline(size(train_out_all, 3) - 0.5, 'k--', 'LineWidth', 1.5); % Separation between Train and Val
    xline(size(train_out_all, 3) + size(val_out, 3) - 0.5, 'k--', 'LineWidth', 1.5); % Separation between Val and Test


    % Add MSE annotations for each part
    text_x_positions = [size(train_out_all,3) / 2, ...
        size(train_out_all,3) + size(val_out,3) / 2, ...
        size(train_out_all,3) + size(val_out,3) + size(test_out,3) / 2];

    max_y_value = max([squeeze(train_out_all(i,:,2:end)); squeeze(val_out(i,:,2:end)); squeeze(test_out(i,:,2:end))], [], 'all');
    text_y_position = extractdata(max_y_value) * 1.1;

    text(text_x_positions(1), text_y_position, sprintf('MSE Train: %.4f', mse_train), 'HorizontalAlignment', 'center');
    text(text_x_positions(2), text_y_position, sprintf('MSE Val: %.4f', mse_val), 'HorizontalAlignment', 'center');
    text(text_x_positions(3), text_y_position, sprintf('MSE Test: %.4f', mse_test), 'HorizontalAlignment', 'center');

    if N>1
        title("y" + i);
    end

end

sgtitle("Predictions vs True Values");
hold off;

end