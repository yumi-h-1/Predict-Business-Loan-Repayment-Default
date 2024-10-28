%% Testing Random Forest Model and Logistic Regression Model %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear command, workspace, and figures
clc;
clear;
close all;

%% Load testset, test_x, and test_y
% Please import testset.csv, test_x.csv, and test_y.csv by clicking those
% .csv files to upload to the workspace in this script

% If that method above doesn't work, the code below will upload each file
% to the workspace

%Load the testset and split it to x and y
testset = readtable('testset.csv');
test_x = testset{:, 1:end-1};
test_y = testset{:, end};

%% Load the final model with RF and LR
% Please import rf_final_mdl.mat, and lr_final_mdl.mat by clicking those
% .mat files to upload to the workspace in this script

% If that method above doesn't work, the code below will upload each model
% to the workspace
rf_final_mdl = load('rf_final_mdl.mat');
rf_final_mdl = rf_final_mdl.rf_mdl2;
lr_final_mdl = load('lr_final_mdl.mat');
lr_final_mdl = lr_final_mdl.lr_mdl2;

%% Evaluation the final Random Forest model
rng default

rf_tt = tic; % Check the elapsed time for running RF model
rf_mdl_predictions = predict(rf_final_mdl, test_x);
rf_mdl_time = toc(rf_tt);

disp(['Elapsed time of the RF model: ', num2str(rf_mdl_time)]);

% Evaluation of the final RF model using confusion matrix
rf_mdl_cm = confusionmat(test_y, double(rf_mdl_predictions));
rf_mdl_TP = rf_mdl_cm(1, 1); % True positive
rf_mdl_TN = rf_mdl_cm(2, 2); % True negative
rf_mdl_FP = rf_mdl_cm(2, 1); % False positive
rf_mdl_FN = rf_mdl_cm(1, 2); % False negative
     
rf_mdl_accuracy = (rf_mdl_TP + rf_mdl_TN) / (rf_mdl_TP + rf_mdl_TN + rf_mdl_FP + rf_mdl_FN);
rf_mdl_precision = rf_mdl_TP / (rf_mdl_TP + rf_mdl_FP);
rf_mdl_recall = rf_mdl_TP / (rf_mdl_TP + rf_mdl_FN);
rf_mdl_f1score = 2 * (rf_mdl_precision * rf_mdl_recall) / (rf_mdl_precision + rf_mdl_recall);

disp(['Accuracy of the RF model: ', num2str(rf_mdl_accuracy)]);
disp(['Precision of the RF model: ', num2str(rf_mdl_precision)]);
disp(['Recall of the RF model: ', num2str(rf_mdl_recall)]);
disp(['F1-score of the RF model: ', num2str(rf_mdl_f1score)]);

%% Evaluation the final Logistic Regression model
rng default

lr_tt = tic; % Check the elapsed time for running LR model
lr_mdl_predictions = predict(lr_final_mdl, test_x);
lr_mdl_time = toc(lr_tt);

disp(['Elapsed time of the LR model: ', num2str(lr_mdl_time)]);

thresh = 0.5; % Set the threshhold
lr_mdl_discrete_preds = lr_mdl_predictions > thresh;
lr_mdl_discrete_preds = double(lr_mdl_discrete_preds);

% Evaluation of the final LR models using confusion matrix
lr_mdl_cm = confusionmat(test_y, lr_mdl_discrete_preds);
lr_mdl_TP = lr_mdl_cm(1, 1); % True positive
lr_mdl_TN = lr_mdl_cm(2, 2); % True negative
lr_mdl_FP = lr_mdl_cm(2, 1); % False positive
lr_mdl_FN = lr_mdl_cm(1, 2); % False negative

lr_mdl_accuracy = (lr_mdl_TP + lr_mdl_TN) / (lr_mdl_TP + lr_mdl_TN + lr_mdl_FP + lr_mdl_FN);
lr_mdl_precision = lr_mdl_TP / (lr_mdl_TP + lr_mdl_FP);
lr_mdl_recall = lr_mdl_TP / (lr_mdl_TP + lr_mdl_FN);
lr_mdl_f1score = 2 * (lr_mdl_precision * lr_mdl_recall) / (lr_mdl_precision + lr_mdl_recall);

disp(['Accuracy of the LR model: ', num2str(lr_mdl_accuracy)]);
disp(['Precision of the LR model: ', num2str(lr_mdl_precision)]);
disp(['Recall of the LR model: ', num2str(lr_mdl_recall)]);
disp(['F1-score of the LR model: ', num2str(lr_mdl_f1score)]);

%% Confusion chart of the Random Forest model
c_rf = confusionchart(test_y, double(rf_mdl_predictions), 'RowSummary','row-normalized','ColumnSummary','column-normalized');

%% Confusion chart of the Logistic Regression model
c_lr = confusionchart(test_y,lr_mdl_discrete_preds,  'RowSummary','row-normalized','ColumnSummary','column-normalized');

%% Find each AUC
[X1, Y1, T1, AUC1] = perfcurve(test_y, double(rf_mdl_predictions), 1);
[X2, Y2, T2, AUC2] = perfcurve(test_y, lr_mdl_discrete_preds, 1);

disp(['AUC of the RF model: ', num2str(AUC1)]);
disp(['AUC of the LR model: ', num2str(AUC2)]);

%% Plot ROC curve of each model
figure
plot(X1, Y1)
hold on
plot(X2, Y2)
hold off
legend('Random Forest', 'Linear Regression')
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curves of RF & LR')
hold off
