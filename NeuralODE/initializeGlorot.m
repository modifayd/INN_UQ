function weights = initializeGlorot(sz,numOut,numIn)

Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
% weights = (rand(sz)*0.5);
weights = (dlarray((weights)));

end