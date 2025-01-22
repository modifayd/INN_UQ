
function X = model(tspan,X0,input,neuralOdeParameters, output_lag)

X = euler_forward(@odeModel,tspan,X0,input,neuralOdeParameters, 1e-1, output_lag);

end

