%{
ReturnBin function returns the bin number for which the incoming data lies.
Alternative to histc, instead returns the valid values
Input params : minFeat, maxFeat - minimum and maximum feature values;
               val, class - value and class to validate the condition
               nBins - number of bins
Output params : binNum - the bin number which fits the incoming data
                fVal, fClass - values and class that match; debug purposes
                i, endVal - starting and endval for loop continuity outside
                the function                
%}
    function[binNum, fVal, fClass, i, endVal] = ReturnBin(minFeat, maxFeat, val, class, nBins)    
    bInterval = (maxFeat - minFeat)/nBins;
    %fprintf('min %g\tmax %g\tinterval %g\n', minFeat, maxFeat, bInterval);
    i = minFeat;
    binNum = 1;
    binAssigned = false;
    % Divide into n Bins with equal intervals
    while i < maxFeat        
        endVal = i + bInterval;
        %fprintf('%g fits between %g and %g?\n', val, i, endVal);
        if(val ~= minFeat)
            if((val > i && val <= endVal) || (val == maxFeat))                
                %fprintf('yes\n');
                binAssigned = true;
                fVal = val;
                fClass = class;
                break;
            end
        else
            %add to first bin  
            %fprintf('fyes\n');
            binAssigned = true;
            fVal = val;
            fClass = class;
            break;
        end
        i = endVal;   
        binNum = binNum + 1;
    end
    if(binAssigned == false)
        binNum = 0;
    end
end