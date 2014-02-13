%{
Implementation of Least squares estimation for 10%, 30%, 50% of the random training data on Fisheriris
dataset
The script does the following:
1. Load the data set and split into training and testing parts.
2. Calculate lambda such that the matrix X'T*X' is positive definite.
3. Complete training by calculating W~.
4. Test the data by multiplying weight with the remaining test data.
5. Compute the misclassification error.
%}

%Load Fisheriris dataset
load fisheriris;
p = [10, 30, 50];
totalSpecies = size(species, 1);
miscError = zeros(3, 10);
clStr = unique(species);
nClasses = size(clStr, 1);
lambdaList = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
trial = zeros(10);

for lambdaV = (1 : size(lambdaList, 2))
    fprintf('Lambda %g\n', lambdaList(lambdaV));
    for probLV = (1 : nClasses)        
        fprintf('Misclassification error for %g percent data \n', p(probLV));
        for loopVar = (1 : 10)
            fprintf('Trial %g\n', loopVar);
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
          for i = (1 : trainIndSize)              
              trainMeas(i,:) = meas(trainInd(i), :);         
              trainSpec(i) = species(trainInd(i));
          end


        % Testing meas and species assignment
          for i = (1 : testIndSize)    
              testMeas(i,:) = meas(testInd(i), :);         
              testSpec(i) = species(testInd(i));
          end

          %Add one new column to training data      
          newCol = ones(trainIndSize, 1);
          t_trainMeas = [trainMeas newCol];
          x = t_trainMeas;
          %Calculate X'T*X'
          tx = x'*x;
          t_x = tx;


          lambda = lambdaList(lambdaV);
          I = eye(size(t_x, 1), size(t_x, 2));
          t_x = t_x + lambda * I;
          %Remove negative eigen values and make the matrix positive definite
          EPS = 10^-6;     
          ZERO = 10^-10;   
          [~, err] = cholcov(t_x, 0);          
          if (err ~= 0) 
            [v, d] = eig(t_x); 
            d=diag(d);         
            d(d<=ZERO)=EPS; 
            d=diag(d);      
            t_x = v*d*v'; 
          end 

          %Formulate T
          T = [];      
          for i = (1 : trainIndSize)
              row = [0 0 0];
              [f, fIdx] = ismember(trainSpec(i), clStr);
              if(f == 1)
                row(fIdx) = 1;
              end
              T = [T; row];
          end

          %Calculate Weight
          W = (t_x) \ (x'* T);

          newCol = ones(testIndSize, 1);
          t_testMeas = [testMeas newCol]';
          x2 = t_testMeas;
          R = W' * x2;
          R = R';

          %Test classification
          testClassification = cell(testIndSize, 1);
          for i = (1 : testIndSize)
              maxVal = max(R(i, :));
              idx = 0;
              for j = (1 : size(clStr, 1))
                if(maxVal == R(i, j))
                    %fprintf('setting %g\n', j);
                    idx = j;
                end
              end
              if(idx ~= 0)
                  testClassification(i) = clStr(idx);
              end
          end
          

          %Percentage of classifications for test data
          classTotal = zeros(size(clStr));
          trainTotal = zeros(size(clStr));
          for i = (1 : size(clStr))
              class = clStr(i);
              for j = (1 : testIndSize)
                  if(strcmp(class, testClassification(j)) == 1)                  
                      classTotal(i) = classTotal(i) + 1;
                  end
              end
              for k = (1 : trainIndSize)
                  if(strcmp(class, trainSpec(k)) == 1)                  
                      trainTotal(i) = trainTotal(i) + 1;
                  end
              end
          end

          %Print percentage of classified data
          %fprintf('Classification stats \n');
          percClass = zeros(size(clStr));
          trainpercClass = zeros(size(clStr));          
          fprintf('\t\t\t');
          for i = (1 : size(clStr))          
              fprintf('%s\t\t', clStr{i});
          end
          fprintf('\n');
          fprintf('Training');
          fprintf('\t');
          for i = (1 : size(clStr))          
              trainpercClass(i) = ((trainTotal(i)) * 100) / trainIndSize;
              fprintf('%.4f%%\t', trainpercClass(i));          
          end
%           fprintf('\n');          
%           for i = (1 : size(clStr))          
%               fprintf('%s\t', clStr{i});
%           end
          fprintf('\n');
          fprintf('Testing');
          fprintf('\t\t');
          for i = (1 : size(clStr))
              percClass(i) = ((classTotal(i)) * 100) / testIndSize;          
              fprintf('%.4f%%\t', percClass(i));
          end
          
          %Misclassification error calculations
          mcMat = ~cellfun(@strcmp, testSpec, testClassification);
          mError = sum(mcMat)/size(species, 1);  
          miscError(probLV, loopVar) = mError;
          fprintf('\n');
          fprintf('Misclassification error : %g\n', mError);  
          
        end
    end
    fprintf('Average misclassification errors \n');
    for i=(1 : nClasses)
        fprintf('%g%% : %g\n', p(i), sum(miscError(i,:))/10);
    end
    fprintf('\n');
end

