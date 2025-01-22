function [upper, lower, mean] = euler_forward_interval(odefunInterval, odefun, tspan, y0, X, neuralOdeParameters_interval,pre_net, h, output_lag)

    %Discrete integration(Forward euler) for ode solver

    T = tspan(:);  % Ensure tspan is a column vector
    Y = dlarray((zeros([length(tspan) size(y0)])));
    Yu = dlarray((zeros([length(tspan) size(y0)])));
    Yl = dlarray((zeros([length(tspan) size(y0)])));

    Y(1,:,:) = y0;   % Initial condition
    Yu(1,:,:) = y0;
    Yl(1,:,:) = y0;

    h=1;
    for i = 1:length(T)-1

        % Mean prediction
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

         if size(y0,2)==1
             y=permute((y),[2 1]);
         end

         model_in = cat(2,y,in);

         % Mean calc
         [inc] = odefun(i,model_in',pre_net);
         % Y(i+1,:,:) = reshape(reshape(squeeze(Y(i,:,:)),size(inc))+inc,size(Y(1,:,:)));
         % Y(i+1,:,:) = reshape(squeeze(Y(i,:,:)),size(inc))+inc;
         % Y(i+1,:,:) = permute(Y(i,:,:),[2 3 1])+inc;
         Y(i+1,:,:)=inc;
        
        %Independent band expansion -- not w.r.t mean --most of te time
        %diverges
        % yu=Yu(i,:,:);
        % yl=Yl(i,:,:);
        % yu=permute(squeeze(yu),[1 2]);
        % yl=permute(squeeze(yl),[1 2]);

        model_in_upper=model_in;%cat(2,yu,in);
        model_in_lower=model_in;%cat(2,yl,in);

        % Upper Lower band wrt mean
         [y_upper,y_lower] = odefunInterval(model_in_upper',model_in_lower',neuralOdeParameters_interval);
         Yu(i+1,:,:) = y_upper;
         Yl(i+1,:,:) = y_lower;
        
         % DELTAu(i,:)=deltau;
         % DELTAl(i,:)=deltal;
         % DELTAm(i,:)=deltam;

    end

    mean = Y(2:end,:,:);
    upper = Yu(2:end,:,:);
    lower = Yl(2:end,:,:);