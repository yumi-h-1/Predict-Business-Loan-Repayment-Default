%% Training Logistic Regression Model %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear command, workspace, and figures
clc;
clear;
close all;

%% Load data
data = readtable('data.csv');
data = removevars(data, ['Var1']); % Remove the index in the datatable

%% Split the data into train and test using a random nonstratified partition
% The codes in this section are from MATLAB, https://uk.mathworks.com/help/stats/cvpartition.html

% Control random number 
rng('default') % 'default' provided by MATLAB, https://uk.mathworks.com/help/matlab/ref/rng.html?s_tid=doc_ta

% Split the data into 80% for training and 20% for testing
h = cvpartition(size(data, 1), 'Holdout', 0.2); 

% Set the index for training and testing sets and split the data into a train set and test set
trainid = training(h);
trainset = data(trainid, :);

testid = ~training(h);
testset = data(testid, :);

%% Plot the target variable to visualize its imbalance
% figure;
% histogram(trainset.Default, 'BinWidth', 0.1, 'Normalization', 'count');
% title('Target Distribution');
% xlabel('Target');
% ylabel('Count');

%% Add Gaussian distribution noise to balance the target data
all_std = std(trainset); % Check each standard deviation in the trainset

std = 0.5; % Adjust the standard deviation for target balancing
noisytrainset = trainset;
% This code below is from MATLAB Answers
% https://uk.mathworks.com/matlabcentral/answers/253208-add-gaussian-distributed-noise-with-mean-and-variance-to-matrix
noisytrainset = trainset + std * randn(size(trainset)); 
noisytrainset.Default = max(0, min(1, noisytrainset.Default)); % Make sure the values in the target are within the range [0, 1]

thresh_noise = 0.5; % Set the threshold for the noisy target to convert it to binary values
noisytrainset.Default = noisytrainset.Default > thresh_noise;

%% Plot the noisy target variable to visualize its balance
% figure;
% histogram(noisytrainset.Default, 'BinWidth', 0.1, 'Normalization', 'count');
% title('Noisy Target Distribution');
% xlabel('Noisy Target');
% ylabel('Count');

%% Split the noisy trainset into x and y
train_y = noisytrainset{:, end};
train_x = noisytrainset{:, 1:end-1};

%% 1st Model with Logistic Regression without regularization.
%lg_tt = tic; % testing the stop watch
lr_mdl = fitglm(train_x, train_y)
%toc(lg_tt);

%% Split the testset into x and y
test_x = testset{:, 1:end-1};
test_y = testset{:, end};

%% Evaluation the first model
lr_mdl_predictions = predict(lr_mdl, test_x);

thresh = 0.5; % Set the threshhold
lr_mdl_discrete_preds = lr_mdl_predictions > thresh;
lr_mdl_discrete_preds = double(lr_mdl_discrete_preds);

% Evaluation of the LR model using confusion matrix
lr_mdl_cm = confusionmat(test_y, lr_mdl_discrete_preds);
lr_mdl_TP = lr_mdl_cm(1, 1); % True positive
lr_mdl_TN = lr_mdl_cm(2, 2); % True negative
lr_mdl_FP = lr_mdl_cm(2, 1); % False positive
lr_mdl_FN = lr_mdl_cm(1, 2); % False negative

lr_mdl_accuracy = (lr_mdl_TP + lr_mdl_TN) / (lr_mdl_TP + lr_mdl_TN + lr_mdl_FP + lr_mdl_FN);
lr_mdl_precision = lr_mdl_TP / (lr_mdl_TP + lr_mdl_FP);
lr_mdl_recall = lr_mdl_TP / (lr_mdl_TP + lr_mdl_FN);
lr_mdl_f1score = 2 * (lr_mdl_precision * lr_mdl_recall) / (lr_mdl_precision + lr_mdl_recall);

disp(['Accuracy of the 1st model using LR: ', num2str(lr_mdl_accuracy)]);
disp(['Precision of the 1st model using LR: ', num2str(lr_mdl_precision)]);
disp(['Recall of the 1st model using LR: ', num2str(lr_mdl_recall)]);
disp(['F1-score of the 1st model using LR: ', num2str(lr_mdl_f1score)]);

%% Change its distribution from normal to binomial and set it as the second model
lr_mdl2 = fitglm(train_x, train_y, 'Distribution','binomial');

%% Evaluation the second model
lr_mdl_predictions2 = predict(lr_mdl2, test_x);

thresh = 0.5; % Set the threshhold
lr_mdl_discrete_preds2 = lr_mdl_predictions2 > thresh;
lr_mdl_discrete_preds2 = double(lr_mdl_discrete_preds2);

% Evaluation of the LR model using confusion matrix
lr_mdl_cm2 = confusionmat(test_y, lr_mdl_discrete_preds2);
lr_mdl_TP2 = lr_mdl_cm2(1, 1); % True positive
lr_mdl_TN2 = lr_mdl_cm2(2, 2); % True negative
lr_mdl_FP2 = lr_mdl_cm2(2, 1); % False positive
lr_mdl_FN2 = lr_mdl_cm2(1, 2); % False negative

lr_mdl_accuracy2 = (lr_mdl_TP2 + lr_mdl_TN2) / (lr_mdl_TP2 + lr_mdl_TN2 + lr_mdl_FP2 + lr_mdl_FN2);
lr_mdl_precision2 = lr_mdl_TP2 / (lr_mdl_TP2 + lr_mdl_FP2);
lr_mdl_recall2 = lr_mdl_TP2 / (lr_mdl_TP2 + lr_mdl_FN2);
lr_mdl_f1score2 = 2 * (lr_mdl_precision2 * lr_mdl_recall2) / (lr_mdl_precision2 + lr_mdl_recall2);

disp(['Accuracy of the 2nd model using LR: ', num2str(lr_mdl_accuracy2)]);
disp(['Precision of the 2nd model using LR: ', num2str(lr_mdl_precision2)]);
disp(['Recall of the 2nd model using LR: ', num2str(lr_mdl_recall2)]);
disp(['F1-score of the 2nd model using LR: ', num2str(lr_mdl_f1score2)]);

% Accuracy of the 2nd model using LR: 0.80432
% Precision of the 2nd model using LR: 0.80803
% Recall of the 2nd model using LR: 0.97619
% F1-score of the 2nd model using LR: 0.88419

%% Increase the maximum iteration of the second model
lr_mdl2 = fitglm(train_x, train_y, 'Distribution','binomial', 'Options',statset('MaxIter',1000));

%% Evaluation the second model
lr_mdl_predictions2 = predict(lr_mdl2, test_x);

thresh = 0.5; % Set the threshhold
lr_mdl_discrete_preds2 = lr_mdl_predictions2 > thresh;
lr_mdl_discrete_preds2 = double(lr_mdl_discrete_preds2);

% Evaluation of the LR model using confusion matrix
lr_mdl_cm2 = confusionmat(test_y, lr_mdl_discrete_preds2);
lr_mdl_TP2 = lr_mdl_cm2(1, 1); % True positive
lr_mdl_TN2 = lr_mdl_cm2(2, 2); % True negative
lr_mdl_FP2 = lr_mdl_cm2(2, 1); % False positive
lr_mdl_FN2 = lr_mdl_cm2(1, 2); % False negative

lr_mdl_accuracy2 = (lr_mdl_TP2 + lr_mdl_TN2) / (lr_mdl_TP2 + lr_mdl_TN2 + lr_mdl_FP2 + lr_mdl_FN2);
lr_mdl_precision2 = lr_mdl_TP2 / (lr_mdl_TP2 + lr_mdl_FP2);
lr_mdl_recall2 = lr_mdl_TP2 / (lr_mdl_TP2 + lr_mdl_FN2);
lr_mdl_f1score2 = 2 * (lr_mdl_precision2 * lr_mdl_recall2) / (lr_mdl_precision2 + lr_mdl_recall2);

disp(['Accuracy of the 2nd model using LR: ', num2str(lr_mdl_accuracy2)]);
disp(['Precision of the 2nd model using LR: ', num2str(lr_mdl_precision2)]);
disp(['Recall of the 2nd model using LR: ', num2str(lr_mdl_recall2)]);
disp(['F1-score of the 2nd model using LR: ', num2str(lr_mdl_f1score2)]);

% Accuracy of the 2nd model using LR: 0.80432
% Precision of the 2nd model using LR: 0.80803
% Recall of the 2nd model using LR: 0.97619
% F1-score of the 2nd model using LR: 0.88419

% No difference after this tuning

%% Find the optimal hyperparameters and set it as the third model
% As the code below is likely to suggest another parameters, this code is
% changed to the conmment.
%rng default
%lr_mdl3 = fitclinear(train_x, train_y, 'OptimizeHyperparameters','auto'); % Use fitclinear to adjust the hyperparameter

%% Store the optimal lambda in the third model and create the fourth model with lasso regularization 
lambda = 2.1929e-06;

rng default
lr_mdl3 = fitclinear(train_x, train_y, 'Lambda', lambda, 'Learner', 'logistic', 'Regularization', 'ridge');
% Change into lasso regularization to check the difference between the 3rd and 4th model's accuracy
lr_mdl4 = fitclinear(train_x, train_y, 'Lambda', lambda, 'Learner', 'logistic', 'Regularization', 'lasso');

%% Evaluation the third model
lr_mdl_predictions3 = predict(lr_mdl3, test_x);

thresh = 0.5; % Set the threshhold
lr_mdl_discrete_preds3 = lr_mdl_predictions3 > thresh;
lr_mdl_discrete_preds3 = double(lr_mdl_discrete_preds3);

% Evaluation of the LR model using confusion matrix
lr_mdl_cm3 = confusionmat(test_y, lr_mdl_discrete_preds3);
lr_mdl_TP3 = lr_mdl_cm3(1, 1); % True positive
lr_mdl_TN3 = lr_mdl_cm3(2, 2); % True negative
lr_mdl_FP3 = lr_mdl_cm3(2, 1); % False positive
lr_mdl_FN3 = lr_mdl_cm3(1, 2); % False negative

lr_mdl_accuracy3 = (lr_mdl_TP3 + lr_mdl_TN3) / (lr_mdl_TP3 + lr_mdl_TN3 + lr_mdl_FP3 + lr_mdl_FN3);
lr_mdl_precision3 = lr_mdl_TP3 / (lr_mdl_TP3 + lr_mdl_FP3);
lr_mdl_recall3 = lr_mdl_TP3 / (lr_mdl_TP3 + lr_mdl_FN3);
lr_mdl_f1score3 = 2 * (lr_mdl_precision3 * lr_mdl_recall3) / (lr_mdl_precision3 + lr_mdl_recall3);

disp(['Accuracy of the 3rd model using LR: ', num2str(lr_mdl_accuracy3)]);
disp(['Precision of the 3rd model using LR: ', num2str(lr_mdl_precision3)]);
disp(['Recall of the 3rd model using LR: ', num2str(lr_mdl_recall3)]);
disp(['F1-score of the 3rd model using LR: ', num2str(lr_mdl_f1score3)]);

% Accuracy of the 3rd model using LR: 0.76518
% Precision of the 3rd model using LR: 0.76518
% Recall of the 3rd model using LR: 1
% F1-score of the 3rd model using LR: 0.86697

%% Evaluation the fourth model
lr_mdl_predictions4 = predict(lr_mdl4, test_x);

thresh = 0.5; % Set the threshhold
lr_mdl_discrete_preds4 = lr_mdl_predictions4 > thresh;
lr_mdl_discrete_preds4 = double(lr_mdl_discrete_preds4);

% Evaluation of the LR model using confusion matrix
lr_mdl_cm4 = confusionmat(test_y, lr_mdl_discrete_preds4);
lr_mdl_TP4 = lr_mdl_cm4(1, 1); % True positive
lr_mdl_TN4 = lr_mdl_cm4(2, 2); % True negative
lr_mdl_FP4 = lr_mdl_cm4(2, 1); % False positive
lr_mdl_FN4 = lr_mdl_cm4(1, 2); % False negative

lr_mdl_accuracy4 = (lr_mdl_TP4 + lr_mdl_TN4) / (lr_mdl_TP4 + lr_mdl_TN4 + lr_mdl_FP4 + lr_mdl_FN4);
lr_mdl_precision4 = lr_mdl_TP4 / (lr_mdl_TP4 + lr_mdl_FP4);
lr_mdl_recall4 = lr_mdl_TP4 / (lr_mdl_TP4 + lr_mdl_FN4);
lr_mdl_f1score4 = 2 * (lr_mdl_precision4 * lr_mdl_recall4) / (lr_mdl_precision4 + lr_mdl_recall4);

disp(['Accuracy of the 4th model using LR: ', num2str(lr_mdl_accuracy4)]);
disp(['Precision of the 4th model using LR: ', num2str(lr_mdl_precision4)]);
disp(['Recall of the 4th model using LR: ', num2str(lr_mdl_recall4)]);
disp(['F1-score of the 4th model using LR: ', num2str(lr_mdl_f1score4)]);

% Accuracy of the 4th model using LR: 0.76518
% Precision of the 4th model using LR: 0.76518
% Recall of the 4th model using LR: 1
% F1-score of the 4th model using LR: 0.86697

% No difference between the 3rd model and the 4th model

%% Find the optimal hyperparameters with a grid search and set it as the fifth model
% As the code below is likely to suggest another parameters, this code is
% changed to the conmment.
%rng default
%lr_mdl5 = fitclinear(train_x, train_y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch'));

%% Store the optimal lambda in the fifth model
lambda2 = 0.00060604;

rng default
lr_mdl5 = fitclinear(train_x, train_y, 'Lambda', lambda2, 'Learner', 'logistic');

%% Evaluation the fifth model
lr_mdl_predictions5 = predict(lr_mdl5, test_x);

thresh = 0.5; % Set the threshhold
lr_mdl_discrete_preds5 = lr_mdl_predictions5 > thresh;
lr_mdl_discrete_preds5 = double(lr_mdl_discrete_preds5);

% Evaluation of the LR model using confusion matrix
lr_mdl_cm5 = confusionmat(test_y, lr_mdl_discrete_preds5);
lr_mdl_TP5 = lr_mdl_cm5(1, 1); % True positive
lr_mdl_TN5 = lr_mdl_cm5(2, 2); % True negative
lr_mdl_FP5 = lr_mdl_cm5(2, 1); % False positive
lr_mdl_FN5 = lr_mdl_cm5(1, 2); % False negative

lr_mdl_accuracy5 = (lr_mdl_TP5 + lr_mdl_TN5) / (lr_mdl_TP5 + lr_mdl_TN5 + lr_mdl_FP5 + lr_mdl_FN5);
lr_mdl_precision5 = lr_mdl_TP5 / (lr_mdl_TP5 + lr_mdl_FP5);
lr_mdl_recall5 = lr_mdl_TP5 / (lr_mdl_TP5 + lr_mdl_FN5);
lr_mdl_f1score5 = 2 * (lr_mdl_precision5 * lr_mdl_recall5) / (lr_mdl_precision5 + lr_mdl_recall5);

disp(['Accuracy of the 5th model using LR: ', num2str(lr_mdl_accuracy5)]);
disp(['Precision of the 5th model using LR: ', num2str(lr_mdl_precision5)]);
disp(['Recall of the 5th model using LR: ', num2str(lr_mdl_recall5)]);
disp(['F1-score of the 5th model using LR: ', num2str(lr_mdl_f1score5)]);

% Accuracy of the 5th model using LR: 0.76518
% Precision of the 5th model using LR: 0.76518
% Recall of the 5th model using LR: 1
% F1-score of the 5th model using LR: 0.86697

% Even though the models using the code 'fitclinear' with hyperparameter tuning
% have higher accuracy than the models using the code 'fitglm',
% the model 3, 4, and 5 show poor predictions.

%% Build the sixth model using lasso regularization
% The codes in this section are from MATLAB, https://uk.mathworks.com/help/stats/lassoglm.html
rng default

[B,FitInfo] = lassoglm(train_x, train_y, 'binomial', 'CV', 5); % Select binomial distribution and 5-fold cross-validation

%% Evaluation the sixth model
% The codes in this section are from MATLAB, https://uk.mathworks.com/help/stats/lassoglm.html
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)];
yhat = glmval(coef,test_x,'logit');
yhatBinom = (yhat>=0.5);

% Evaluation of the LR model using confusion matrix
lr_mdl_cm6 = confusionmat(test_y,double(yhatBinom));
lr_mdl_TP6 = lr_mdl_cm6(1, 1); % True positive
lr_mdl_TN6 = lr_mdl_cm6(2, 2); % True negative
lr_mdl_FP6 = lr_mdl_cm6(2, 1); % False positive
lr_mdl_FN6 = lr_mdl_cm6(1, 2); % False negative

lr_mdl_accuracy6 = (lr_mdl_TP6 + lr_mdl_TN6) / (lr_mdl_TP6 + lr_mdl_TN6 + lr_mdl_FP6 + lr_mdl_FN6);
lr_mdl_precision6 = lr_mdl_TP6 / (lr_mdl_TP6 + lr_mdl_FP6);
lr_mdl_recall6 = lr_mdl_TP6 / (lr_mdl_TP6 + lr_mdl_FN6);
lr_mdl_f1score6 = 2 * (lr_mdl_precision6 * lr_mdl_recall6) / (lr_mdl_precision6 + lr_mdl_recall6);

disp(['Accuracy of the 6th model using LR: ', num2str(lr_mdl_accuracy6)]);
disp(['Precision of the 6th model using LR: ', num2str(lr_mdl_precision6)]);
disp(['Recall of the 6th model using LR: ', num2str(lr_mdl_recall6)]);
disp(['F1-score of the 6th model using LR: ', num2str(lr_mdl_f1score6)]);

% Accuracy of the 6th model using LR: 0.80499
% Precision of the 6th model using LR: 0.80505
% Recall of the 6th model using LR: 0.98325
% F1-score of the 6th model using LR: 0.88527

%% Plot the cross-validation error with the lambda regularization parameter
% The codes in this section are from MATLAB, https://uk.mathworks.com/help/stats/lassoplot.html?s_tid=doc_ta
lassoPlot(B,FitInfo,'plottype','CV'); 
legend('show')

%% Build the seventh model changing maximum iterations and set the lambda with minimum cross-validation error
% The codes in this section are from MATLAB, https://uk.mathworks.com/help/stats/lassoglm.html
rng default
[B2,FitInfo2] = lassoglm(train_x, train_y, 'binomial', 'CV', 5, 'MaxIter', 1000, 'Lambda', 0.0015288);

%% Evaluation the seventh model
% The codes in this section are from MATLAB, https://uk.mathworks.com/help/stats/lassoglm.html
idxLambdaMinDeviance = FitInfo2.IndexMinDeviance;
B02 = FitInfo2.Intercept(idxLambdaMinDeviance);
coef2 = [B02; B2(:,idxLambdaMinDeviance)];
yhat2 = glmval(coef2,test_x,'logit');
yhatBinom2 = (yhat2>=0.5);

% Evaluation of the LR model using confusion matrix
lr_mdl_cm7 = confusionmat(test_y,double(yhatBinom2));
lr_mdl_TP7 = lr_mdl_cm7(1, 1); % True positive
lr_mdl_TN7 = lr_mdl_cm7(2, 2); % True negative
lr_mdl_FP7 = lr_mdl_cm7(2, 1); % False positive
lr_mdl_FN7 = lr_mdl_cm7(1, 2); % False negative

lr_mdl_accuracy7 = (lr_mdl_TP7 + lr_mdl_TN7) / (lr_mdl_TP7 + lr_mdl_TN7 + lr_mdl_FP7 + lr_mdl_FN7);
lr_mdl_precision7 = lr_mdl_TP7 / (lr_mdl_TP7 + lr_mdl_FP7);
lr_mdl_recall7 = lr_mdl_TP7 / (lr_mdl_TP7 + lr_mdl_FN7);
lr_mdl_f1score7 = 2 * (lr_mdl_precision7 * lr_mdl_recall7) / (lr_mdl_precision7 + lr_mdl_recall7);

disp(['Accuracy of the 7th model using LR: ', num2str(lr_mdl_accuracy7)]);
disp(['Precision of the 7th model using LR: ', num2str(lr_mdl_precision7)]);
disp(['Recall of the 7th model using LR: ', num2str(lr_mdl_recall7)]);
disp(['F1-score of the 7th model using LR: ', num2str(lr_mdl_f1score7)]);

% Accuracy of the 7th model using LR: 0.80499
% Precision of the 7th model using LR: 0.80505
% Recall of the 7th model using LR: 0.98325
% F1-score of the 7th model using LR: 0.88527

% No difference between the 6th model and the 7th model

%% Compare the cross-entropy loss between tuned models
lr_mdl2_loss = crossentropy(test_y, lr_mdl_predictions2);
lr_mdl5_loss = crossentropy(test_y, lr_mdl_predictions5);
lr_mdl7_loss = crossentropy(test_y, double(yhatBinom2));
disp(['Cross-Entropy Loss of the 2nd model: ', num2str(lr_mdl2_loss)]);
disp(['Cross-Entropy Loss of the 5th model: ', num2str(lr_mdl5_loss)]);
disp(['Cross-Entropy Loss of the 7th model: ', num2str(lr_mdl7_loss)]);

% Cross-Entropy Loss of the 2nd model: 0.19646
% Cross-Entropy Loss of the 5th model: 8.4637
% Cross-Entropy Loss of the 6th model: 6.5667

%% Save the final model
% The second model with the smallest cross-entropy loss)
%save('lr_final_mdl.mat', 'lr_mdl2')
