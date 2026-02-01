%% EV Fault Detection using SOC + LSTM 
clc; clear; close all; 
%% STEP 1: Load SOC data from CSV (Fresh vs Aged Cells) 
aged_path  = 'C:\Users\Suyash Naik\OneDrive\文档\MATLAB\Experimental_data_aged_cell.csv'; 
fresh_path = 'C:\Users\Suyash Naik\OneDrive\文档\MATLAB\Experimental_data_fresh_cell.csv'; 
aged  = readtable(aged_path); 
fresh = readtable(fresh_path); 
% Extract columns (Time[s], Current[A], Voltage[V], Temperature[C]) 
t_aged  = aged.Time; 
I_aged  = aged.Current; 
V_aged  = aged.Voltage; 
t_fresh = fresh.Time; 
I_fresh = fresh.Current; 
V_fresh = fresh.Voltage; 
%% STEP 2: Estimate SOC using Coulomb Counting 
C_nom = 2.3 * 3600;   % nominal capacity (2.3Ah assumed, in Coulombs) 
dt_aged  = mean(diff(t_aged)); 
dt_fresh = mean(diff(t_fresh)); 
SOC_aged  = 1 - cumsum(I_aged .* dt_aged)  / C_nom; 
SOC_fresh = 1 - cumsum(I_fresh .* dt_fresh)/ C_nom; 
% Normalize [0,1] 
SOC_aged  = max(min(SOC_aged,1),0); 
SOC_fresh = max(min(SOC_fresh,1),0); 
%% STEP 3: Combine datasets (Fresh = Normal class 1, Aged = Fault class 2) 
soc_raw = [SOC_fresh(:); SOC_aged(:)]; 
labels_raw = [ones(length(SOC_fresh),1); 2*ones(length(SOC_aged),1)]; 
% Simple preprocessing (smoothing = ANC-like) 
soc_anc = movmean(soc_raw,5)'; 
%% STEP 4: Sequence parameters 
fs = 1;                                    
t = 0:1/fs:(length(soc_raw)-1);            
sequence_length = 50; 
step = 10; 
X = {}; Y_labels = []; 
idx = 1; 
% sampling frequency 
% time axis 
for start_idx = 1:step:(length(soc_anc)-sequence_length+1) 
end_idx = start_idx + sequence_length - 1; 
seq = soc_anc(start_idx:end_idx); 
% Force row vector [1 × T] 
X{idx} = reshape(seq(:)', [1, sequence_length]); 
% Use dataset labels (based on average of segment) 
if mean(labels_raw(start_idx:end_idx)) < 1.5 
Y_labels(idx) = 1;   % Normal (Fresh) 
    else 
        Y_labels(idx) = 2;   % Fault (Aged) 
    end 
    idx = idx + 1; 
end 
 
% If no sequences, pad 
if isempty(X) 
    warning('Data too short, padding with zeros'); 
    soc_pad = [soc_anc(:)' zeros(1, sequence_length - length(soc_anc))]; 
    X{1} = reshape(soc_pad(1:sequence_length), [1, sequence_length]); 
    Y_labels(1) = 1; 
end 
 
%% STEP 5: Prepare data for training/testing 
Y = categorical(Y_labels(:)); 
N = numel(X); 
idx = randperm(N); 
train_idx = idx(1:round(0.7*N)); 
test_idx  = idx(round(0.7*N)+1:end); 
 
XTrain = X(train_idx); 
YTrain = Y(train_idx); 
XTest  = X(test_idx); 
YTest  = Y(test_idx); 
 
disp("=== Debugging sequence shapes ==="); 
disp(size(XTrain)); 
disp(size(XTrain{1})); 
disp(size(YTrain)); 
 
%% STEP 6: Define LSTM network 
numFeatures = 1; 
numHiddenUnits = 50; 
numClasses = numel(categories(Y)); 
 
layers = [ ... 
    sequenceInputLayer(numFeatures) 
    lstmLayer(numHiddenUnits,'OutputMode','last') 
    fullyConnectedLayer(numClasses) 
    softmaxLayer 
    classificationLayer]; 
 
options = trainingOptions('sgdm', ... 
    'MaxEpochs', 20, ... 
    'MiniBatchSize', min(16,numel(XTrain)), ... 
    'InitialLearnRate', 0.01, ... 
    'Verbose', true, ... 
    'Plots', 'training-progress'); 
 
%% STEP 7: Train the network 
[net, info] = trainNetwork(XTrain, YTrain, layers, options); 
 
%% STEP 8: Test and accuracy 
YPred = classify(net, XTest); 
acc = mean(YPred == YTest); 
disp(['Test Accuracy: ', num2str(acc*100), '%']); 
 
%% STEP 9: Plot example 
figure; 
subplot(2,1,1); 
plot(t, soc_raw); title('Raw SOC Signal (Fresh+Fault Combined)'); 
xlabel('Time (s)'); ylabel('SOC'); 
subplot(2,1,2); 
plot(t, soc_anc); title('Preprocessed SOC (ANC-smoothed)'); 
xlabel('Time (s)'); ylabel('SOC'); 
%% STEP 10: Error Metrics 
Ytrue = double(YTest); Yhat = double(YPred); 
MAE  = mean(abs(Ytrue - Yhat)); 
RMSE = sqrt(mean((Ytrue - Yhat).^2)); 
MAPE = mean(abs((Ytrue - Yhat)./max(Ytrue,1)))*100; 
disp(['MAE  = ', num2str(MAE)]); 
disp(['RMSE = ', num2str(RMSE)]); 
disp(['MAPE = ', num2str(MAPE), '%']); 