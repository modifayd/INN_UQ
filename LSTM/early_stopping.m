
function [counter,min_val_loss,early_indicator] = early_stopping(val_loss,patience,min_delta,counter,min_val_loss)

early_indicator=0;
    if val_loss < min_val_loss
        min_val_loss = val_loss;
        counter=0;

    elseif (val_loss > (min_val_loss+min_delta))
        counter = counter+1;
            if counter >= patience
                early_indicator=1;
            end
    end
end


% 
% function [counter, min_val_loss, early_indicator] = early_stopping(val_loss, patience, min_delta, counter, min_val_loss)
%     % Initialize the early stopping indicator
%     early_indicator = 0;
% 
%     % Check if the current validation loss is the new minimum
%     if val_loss < (min_val_loss -min_delta)
%         min_val_loss = val_loss; % Update minimum validation loss
%         counter = 0;            % Reset counter
%     else
%         % Increment counter if no improvement beyond min_delta
%         counter = counter + 1;
% 
%         % Trigger early stopping if patience is exceeded
%         if counter >= patience
%             early_indicator = 1;
%         end
%     end
% end