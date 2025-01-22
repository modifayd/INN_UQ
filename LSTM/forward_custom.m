
function out = forward_custom(net,input,target0,output_lag)

out = dlarray(zeros([size(target0,1) size(input,2) size(input,3)-1]));
target0=cat(3,zeros([size(out,1) size(out,2) output_lag-1]),target0);
target0=permute(stripdims(target0),[3 2 1]);

for i=1:size(input,3)-1

    % Given u(k)...u(k-nx) and y(k-1)...y(k-ny) predict y(k)
    model_in = cat(1,input(:,:,i),target0);
    [out(:,:,i),state] = predict(net,dlarray(model_in,"CBT"));
    net.State=state;

    % Get initial state for next step
    if output_lag>i
        target0=cat(3,zeros([size(out,1) size(out,2) output_lag-i]),out(:,:,1:i));
    else
        target0=out(:,:,i-output_lag+1:i);
    end
    target0 = permute((target0),[3 2 1]);
end


end

