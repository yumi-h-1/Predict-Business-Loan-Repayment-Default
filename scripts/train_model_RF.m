%% Training Random Forest Model %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%figure;
%histogram(trainset.Default, 'BinWidth', 0.1, 'Normalization', 'count');
%title('Target Distribution');
%xlabel('Target');
%ylabel('Count');

%% Add Gaussian distribution noise to balance the data in the target
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
%figure;
%histogram(noisytrainset.Default, 'BinWidth', 0.1, 'Normalization', 'count');
%title('Noisy Target Distribution');
%xlabel('Noisy Target');
%ylabel('Count');

%% Train the first model with bagging 
% The code in this section is from MATLAB, https://kr.mathworks.com/help/stats/fitcensemble.html
%rf_tt = tic; % testing the stop watch
rng default
t = templateTree('Reproducible',true); % Set the base model as decision tree
rf_mdl = fitcensemble(noisytrainset, 'Default', 'Method', 'Bag', 'Learners', t); % Use bagging as a method
%rf_mdl_time = toc(rf_tt);

%disp(['Elasped time of the 1st model using RF: ', num2str(rf_mdl_time)]);

%% Split the testset into x and y
test_x = testset{:, 1:end-1};
test_y = testset{:, end};

%% Evaluate the first model
% Predictions from the model
rf_mdl_predictions = predict(rf_mdl, test_x);
rf_mdl_predictions = double(rf_mdl_predictions);

% Evaluation of the RF model using confusion matrix
rf_mdl_cm = confusionmat(test_y, rf_mdl_predictions);
rf_mdl_TP = rf_mdl_cm(1, 1); % True positive
rf_mdl_TN = rf_mdl_cm(2, 2); % True negative
rf_mdl_FP = rf_mdl_cm(2, 1); % False positive
rf_mdl_FN = rf_mdl_cm(1, 2); % False negative
     
rf_mdl_accuracy = (rf_mdl_TP + rf_mdl_TN) / (rf_mdl_TP + rf_mdl_TN + rf_mdl_FP + rf_mdl_FN);
rf_mdl_precision = rf_mdl_TP / (rf_mdl_TP + rf_mdl_FP);
rf_mdl_recall = rf_mdl_TP / (rf_mdl_TP + rf_mdl_FN);
rf_mdl_f1score = 2 * (rf_mdl_precision * rf_mdl_recall) / (rf_mdl_precision + rf_mdl_recall);

disp(['Accuracy of the 1st model using RF: ', num2str(rf_mdl_accuracy)]);
disp(['Precision of the 1st model using RF: ', num2str(rf_mdl_precision)]);
disp(['Recall of the 1st model using RF: ', num2str(rf_mdl_recall)]);
disp(['F1-score of the 1st model using RF: ', num2str(rf_mdl_f1score)]);

%% Measure the loss with the number of trained trees
% The codes in this section are from MATLAB, https://uk.mathworks.com/help/stats/methods-to-evaluate-ensemble-quality.html
figure
plot(loss(rf_mdl, test_x, test_y,'mode','cumulative'))
xlabel('Number of trees')
ylabel('Test classification error')

%% Set the number of trees to 20 and build the second model
rng default
rf_mdl2 = fitcensemble(noisytrainset, 'Default', 'Method', 'Bag', 'NumLearningCycles', 20, 'Learners', t);

%% Evaluate the second model
rf_mdl_predictions2 = predict(rf_mdl2, test_x);
rf_mdl_predictions2 = double(rf_mdl_predictions2);

% Evaluation of the RF model using confusion matrix
rf_mdl_cm2 = confusionmat(test_y, rf_mdl_predictions2);
rf_mdl_TP2 = rf_mdl_cm2(1, 1); % True positive
rf_mdl_TN2 = rf_mdl_cm2(2, 2); % True negative
rf_mdl_FP2 = rf_mdl_cm2(2, 1); % False positive
rf_mdl_FN2 = rf_mdl_cm2(1, 2); % False negative
    
rf_mdl_accuracy2 = (rf_mdl_TP2 + rf_mdl_TN2) / (rf_mdl_TP2 + rf_mdl_TN2 + rf_mdl_FP2 + rf_mdl_FN2);
rf_mdl_precision2 = rf_mdl_TP2 / (rf_mdl_TP2 + rf_mdl_FP2);
rf_mdl_recall2 = rf_mdl_TP2 / (rf_mdl_TP2 + rf_mdl_FN2);
rf_mdl_f1score2 = 2 * (rf_mdl_precision2 * rf_mdl_recall2) / (rf_mdl_precision2 + rf_mdl_recall2);

disp(['Accuracy of the 2nd model using RF: ', num2str(rf_mdl_accuracy2)]);
disp(['Precision of the 2nd model using RF: ', num2str(rf_mdl_precision2)]);
disp(['Recall of the 2nd model using RF: ', num2str(rf_mdl_recall2)]);
disp(['F1-score of the 2nd model using RF: ', num2str(rf_mdl_f1score2)]);

%% Find the optimal hyperparameters and set it as the third model
% As the code below is likely to suggest another parameters, this code is
% changed to the conmment.
%rng default
%rf_mdl3 = fitcensemble(noisytrainset, 'Default', 'Learners', t, ...
%    'OptimizeHyperparameters', 'auto'); % Automatically find the optimal hyperparameters

%% Store the optimal hyperparameters in the third model
rng default
tTree = templateTree('Reproducible', true, 'MinLeafSize', 1);
rf_mdl3 = fitcensemble(noisytrainset, 'Default', 'Method', 'LogitBoost', ...
    'NumLearningCycles', 440, 'LearnRate',  0.05745, 'Learners', tTree);
%% Evaluate the third model
rf_mdl_predictions3 = predict(rf_mdl3, test_x);
rf_mdl_predictions3 = double(rf_mdl_predictions3);

%Evaluation of the RF model using confusion matrix
rf_mdl_cm3 = confusionmat(test_y, rf_mdl_predictions3);
rf_mdl_TP3 = rf_mdl_cm3(1, 1); % True positive
rf_mdl_TN3 = rf_mdl_cm3(2, 2); % True negative
rf_mdl_FP3 = rf_mdl_cm3(2, 1); % False positive
rf_mdl_FN3 = rf_mdl_cm3(1, 2); % False negative

rf_mdl_accuracy3 = (rf_mdl_TP3 + rf_mdl_TN3) / (rf_mdl_TP3 + rf_mdl_TN3 + rf_mdl_FP3 + rf_mdl_FN3);
rf_mdl_precision3 = rf_mdl_TP3 / (rf_mdl_TP3 + rf_mdl_FP3);
rf_mdl_recall3 = rf_mdl_TP3 / (rf_mdl_TP3 + rf_mdl_FN3);
rf_mdl_f1score3 = 2 * (rf_mdl_precision3 * rf_mdl_recall3) / (rf_mdl_precision3 + rf_mdl_recall3);

disp(['Accuracy of the 3rd model using RF: ', num2str(rf_mdl_accuracy3)]);
disp(['Precision of the 3rd model using RF: ', num2str(rf_mdl_precision3)]);
disp(['Recall of the 3rd model using RF: ', num2str(rf_mdl_recall3)]);
disp(['F1-score of the 3rd model using RF: ', num2str(rf_mdl_f1score3)]);

% Accuracy of the 3rd model using RF: 0.9386
% Precision of the 3rd model using RF: 0.95152
% Recall of the 3rd model using RF: 0.96914
% F1-score of the 3rd model using RF: 0.96024

%% Store the optimal hyperparameters with cross-validation and copy it to the fourth model
rng default
tTree = templateTree('Reproducible', true, 'MinLeafSize', 1);
rf_mdl4 = fitcensemble(noisytrainset, 'Default', 'Method', 'LogitBoost', ...
    'NumLearningCycles', 440, 'LearnRate',  0.05745, ...
    'Kfold', 5, 'Learners', tTree); % 5-fold cross-validation

%% Extract submodels from the fourth model
submodels = rf_mdl4.Trained;

%% Evaluate each submodel
% The code in this section is generated by ChatGPT
% https://chat.openai.com/
numsubmodels = numel(submodels);

for i = 1:numsubmodels

    % Predictions from each submodel
    submodel_predictions = predict(submodels{i}, test_x);
    submodel_predictions = double(submodel_predictions);

    % Evaluate submodel
    cm = confusionmat(test_y, submodel_predictions);

    TP = cm(1,1); 
    TN = cm(2,2);
    FP = cm(2,1);
    FN = cm(1,2);

    accuracy(i) = (TP + TN) / (TP + TN + FP + FN);
    precision(i) = TP / (TP + FP); 
    recall(i) = TP / (TP + FN);
    f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));

    disp(['Accuracy of Submodel ', num2str(i), ': ', num2str(accuracy(i))]);
    disp(['Precision of Submodel ', num2str(i), ': ', num2str(precision(i))]); 
    disp(['Recall of Submodel ', num2str(i), ': ', num2str(recall(i))]); 
    disp(['F1-score of Submodel ', num2str(i), ': ', num2str(f1(i))]); 

end

% Accuracy of Submodel 1: 0.9386
% Precision of Submodel 1: 0.95308
% Recall of Submodel 1: 0.96737
% F1-score of Submodel 1: 0.96018
% Accuracy of Submodel 2: 0.92713
% Precision of Submodel 2: 0.96892
% Recall of Submodel 2: 0.93474
% F1-score of Submodel 2: 0.95153
% Accuracy of Submodel 3: 0.93995
% Precision of Submodel 3: 0.95395
% Recall of Submodel 3: 0.96825
% F1-score of Submodel 3: 0.96105
% Accuracy of Submodel 4: 0.93725
% Precision of Submodel 4: 0.94909
% Recall of Submodel 4: 0.97002
% F1-score of Submodel 4: 0.95944
% Accuracy of Submodel 5: 0.9278
% Precision of Submodel 5: 0.93926
% Recall of Submodel 5: 0.96825
% F1-score of Submodel 5: 0.95354

%% Find AUC
[X1, Y1, T1, AUC1] = perfcurve(test_y, rf_mdl_predictions, 1);
[X2, Y2, T2, AUC2] = perfcurve(test_y, rf_mdl_predictions2, 1);
[X3, Y3, T3, AUC3] = perfcurve(test_y, rf_mdl_predictions3, 1);
[X4, Y4, T4, AUC4] = perfcurve(test_y, submodel_predictions, 1);
disp(['AUC of the 1st model: ', num2str(AUC1)]);
disp(['AUC of the 2nd model: ', num2str(AUC2)]);
disp(['AUC of the 3rd model: ', num2str(AUC3)]);
disp(['AUC of the 4th model(CV): ', num2str(AUC3)]);

% AUC of the 1st model: 0.90081
% AUC of the 2nd model: 0.90688
% AUC of the 3rd model: 0.90411
% AUC of the 4th model(CV): 0.90411

%% ROC curve of each model
figure
plot(X1, Y1, LineWidth=1)
hold on
plot(X2, Y2, "--", LineWidth=1.2)
plot(X3, Y3, ":", LineWidth=2)
plot(X4, Y4, "-.", LineWidth=1)
hold off
legend('1st model', '2nd model', '3rd model', '4th model(CV)')
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curves of Random Forest models')
hold off

%% Select and save the final model
% The second model with the highest AUC
%save('rf_final_mdl.mat', 'rf_mdl2')
