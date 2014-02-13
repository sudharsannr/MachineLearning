%{
Naive Bayes implementation for 10%, 30%, 50% of the random training data on Fisheriris
dataset
The script does the following:
1. Load the data set and split into training and testing parts.
2. Calculate the overall probability of the classes p(c).
3. Train the Fisheriris dataset dividing into bins.
4. Test the Fisheriris dataset and obtain the classifications.
5. Compute the miscalculation error.
%}

%Load Fisheriris dataset
load fisheriris;
p = [10, 30, 50];
totalSpecies = size(species, 1);
miscError = zeros(3, 10);
nBins = 8;
graphPlotted = false;
figCnt = 1;


%Looping
for probLV = (1 : 3)
    graphPlotted = false;
    fprintf('Misclassification error for %g percent data \n', p(probLV));
    for loopVar = (1 : 10)
        % Divide into training and testing sets
        [trainInd, testInd, valInd] = dividerand(150, p(probLV)/100, 1 - p(probLV)/100, 0.0);

        
        % Init
         trainIndSize = size(trainInd, 2);
         testIndSize = size(testInd, 2);         
         trainMeas = zeros(trainIndSize, 4);
         trainSpec = cell(trainIndSize, 1);
         testMeas = zeros(testIndSize, 4);
         testSpec = cell(testIndSize, 1);

         
        % Training meas and species assignment
          for i=(1 : trainIndSize)              
              trainMeas(i,:) = meas(trainInd(i), :);         
              trainSpec(i) = species(trainInd(i));
          end

          
        % Testing meas and species assignment
          for i=(1 : testIndSize)    
              testMeas(i,:) = meas(testInd(i), :);         
              testSpec(i) = species(testInd(i));
          end

        % Plot data
        if(graphPlotted == false)
            figure(figCnt);
            gscatter(testMeas(:,1), testMeas(:,2), testSpec,'rgb','od*');
            xlabel('Sepal length');
            ylabel('Sepal width');            
            figCnt = figCnt + 1;
        end
          
        % Calculate P(C)
        clStr = unique(species);
        countStr = zeros(3, 1);
        pc = zeros(3, 1);
        totSize = size(trainSpec, 1);
        for i=(1: 3)        
            countStr(i)=sum(ismember(trainSpec,clStr(i)));    
            pc(i) = countStr(i)/totSize;
        end

        
        % Bin allocation for training data        
        measSize1 = size(meas,1);
        measSize2 = size(meas, 2);
        featTable = zeros(4, nBins, ceil(measSize1));
        classTable = cell(4, nBins, ceil(measSize1));
        for f=(1:measSize2)
            maxFeat = max(meas(:, f));
            minFeat = min(meas(:, f));    
            bin = zeros(nBins, ceil(measSize1));
            cbin = cell(nBins, ceil(measSize1));
            binWidth = ones(nBins);    
            % Divide into n Bins with equal intervals
            for k=(1: size(trainMeas, 1))
                val = trainMeas(k, f);
                class = trainSpec(k);        
                [binNum, fVal, fClass, i, endVal] = ReturnBin(minFeat, maxFeat, val, class, nBins);
                if(binNum ~= 0)
                    %fprintf('Adding %g to bin %g : %g - %g\tbinWidth %g\n', val, binNum, i, endVal, binWidth(binNum)+1);
                    bin(binNum, binWidth(binNum)) = fVal;        
                    cbin(binNum, binWidth(binNum)) = fClass;
                    binWidth(binNum) = binWidth(binNum) + 1;
                end
            end  
            featTable(f,:,:) = bin;
            classTable(f,:,:) = cbin;
        end

        
        %Reorganize table structure
        mainTbl = zeros(size(clStr, 1), nBins, measSize2);
        for k=(1: measSize2)
            tempTbl = squeeze(classTable(k,:,:));
            cStr = zeros(3, 1);
            chTbl = zeros(size(clStr, 1), nBins);
            for i=(1: 3)        
                for j=(1: size(tempTbl, 1))        
                    tVar = squeeze(tempTbl(j,:));            
                    cStr=sum(ismember(tVar(~cellfun('isempty', tVar)),clStr(i)));    
                    chTbl(i, j) = cStr;
                end    
            end
            mainTbl(:, :, k) = chTbl(:, :);
        end


        %Read testing data and calculate P(Fi/C)
        %display('Testing');
        testClassification = cell(size(testSpec, 1), 1);
        pfcTbl = -1 * ones(size(clStr, 1), nBins, measSize2);        
        for f=(1:size(meas,2))
            maxFeat = max(meas(:, f));
            minFeat = min(meas(:, f));
            %fprintf('--Feature %g--\n',f);    
            % Divide into n Bins with equal intervals
            for k=(1: size(testMeas, 1))
                val = testMeas(k, f);
                class = testSpec(k);
                %fprintf('Searching %g\n', val);
                [binNum, fVal, fClass, i, endVal] = ReturnBin(minFeat, maxFeat, val, class, nBins);
                if(binNum ~= 0)
                    probVal = zeros(size(clStr, 1), 1);
                    for i = (1: size(clStr, 1))                  
                        %fprintf('Class %g\n', i);                        
                        probValProd = calcProbProd(i, binNum, mainTbl, measSize2);
                        probVal(i) = pc(i) * probValProd;                
                        %fprintf('ProbValabc %g\n', probVal(i));                                             
                        %fprintf('Value %g\tFeatuer %g\tClass %g\tBin %g\tProb %g\n', val, f, fIndx, binNum, pfc);                
                    end                    
                    [maxVal, mIdx] = max(probVal);            
                    %TODO : Remove
                    clear tClStr;
                    tClStr = clStr{mIdx};
                    testClassification(k) = clStr(mIdx);
                    %fprintf('Max %g\tIndx %g\tClassification %s\n', maxVal, mIdx, tClStr);
                end
            end      
        end

        %Histogram plot
        if(graphPlotted == false)            
%             figure(figCnt);
%             hist(trainMeas(:, 1), nBins);
%             figure(figCnt);
%             hist(testMeas(:, 1), nBins);
%             graphPlotted = true;
%             figCnt = figCnt + 1;           
            figure(figCnt);
            gscatter(testMeas(:,1), testMeas(:,2), testClassification,'rgb','od*');
            xlabel('Sepal length');
            ylabel('Sepal width');            
            graphPlotted = true;
            figCnt = figCnt + 1;
        end        
        
        %Misclassification error
        mcMat = ~cellfun(@strcmp, testSpec, testClassification);
        mError = sum(mcMat)/size(species, 1);
        miscError(probLV, loopVar) = mError;
        fprintf('%g\n', mError);
    end
end
for i=(1 : 3)
    fprintf('Average misclassification error for %g percent data %g\n', p(i), sum(miscError(i,:))/10);
end

% Direct approach
%# train model
%nb = NaiveBayes.fit(trainMeas, trainSpec);
%# prediction
%y = nb.predict(testMeas);
%# confusion matrix
%cMat = confusionmat(testSpec,y);
%display(cMat);