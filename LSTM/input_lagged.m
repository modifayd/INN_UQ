function U=input_lagged(V,delay)

    % Initialize the output matrix
    num = size(V,2);
    U = zeros(delay+1,num);
    
    % Fill the matrix iteratively
    for i = 1:delay+1
        if i == 1
            U(i,:) = V; % No delay for the first column
        else
            U(i,:) = [zeros(size(V,1),i-1) V(:,1:end-(i-1))]; % Delayed values
        end
    end

end