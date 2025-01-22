function [out_upper,out_lower] = Interval_elementwiseProduct(upper_in1,lower_in1,upper_in2,lower_in2)

a1 = upper_in1.*upper_in2;
a2 = upper_in1.*lower_in2;
a3 = lower_in1.*upper_in2;
a4 = lower_in1.*lower_in2;
A=cat(3,a1,a2,a3,a4);
out_upper = max(A,[],3);
out_lower = min(A,[],3);

end

