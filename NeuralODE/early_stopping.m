
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
