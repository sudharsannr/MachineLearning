%{
calcProbProd function returns the probability for the particular class
and feature. Stops instantaneously when the probability becomes zero.
Input params : classNum - class index
               binNum - bin number
               mainTbl - source frequency table
               measSize - total measurement dimension
Output params : probVal - probability value
%}
function[probVal] = calcProbProd(classNum, binNum, mainTbl, measSize)
   probVal = 1;
   for i = (1 : measSize)
       v1 = mainTbl(classNum, binNum, i);
       v2 = sum(mainTbl(classNum, :, i));       
       %fprintf('Feature %g Bin %g\t%g * (%g/%g)\n', i, binNum, probVal, v1, v2);
       probVal = probVal * (v1/v2);
       if(probVal == 0)
           break;
       end
   end
end