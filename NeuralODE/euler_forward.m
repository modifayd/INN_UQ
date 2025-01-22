function out = euler_forward(odefun, tspan, y0, X, neuralOdeParameters,h, output_lag)
    
    %Discrete integration(Forward euler) for ode solver

    T = tspan(:);  % Ensure tspan is a column vector
    Y = dlarray((zeros([length(tspan) size(y0)])));
    Y(1,:,:) = y0;   % Initial condition
    h=1;
    
    for i = 1:length(T)-1

         in = X(:,:,i);
         % y = Y(i, :,:);

         %laggeed output
        if output_lag>i
            y=cat(1,zeros([output_lag-i  size(Y,2) size(Y,3)]),Y(1:i,:,:));
        else
            y=Y(i-output_lag+1:i,:,:);
        end
        y = squeeze(permute((y),[3 2 1]));


         % if length(size(y))==2
         %    y=squeeze(y);
         % else
         % 
         %    if size(y0,1)==1
         %        y=squeeze(y);
         %    else
         %        y=(permute(squeeze(y),[2 1]));
         %    end
         % 
         % end

         in=squeeze(stripdims(in))';
         % y=squeeze(stripdims(y))';

         if size(y0,2)==1
             y=permute((y),[2 1]);
         end
         model_in = cat(2,y,in);
         inc = h*odefun(i,model_in',neuralOdeParameters);
         Y(i+1,:,:)=inc;
    
    end

    out = Y(2:end,:,:);

   