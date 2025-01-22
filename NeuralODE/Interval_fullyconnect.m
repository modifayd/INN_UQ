function [pred_upper,pred_lower] = Interval_fullyconnect(upper_w,lower_w,upper_in,lower_in,upper_b,lower_b)

[upper,lower] = pagemtimesInterval(lower_w,upper_w,lower_in,upper_in);
% [upper1,lower1] = pagemtimesInterval_slowed(lower_w,upper_w,lower_in,upper_in);

pred_upper =upper+upper_b;
pred_lower =lower+lower_b;

end




