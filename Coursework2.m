%% Coursework 2 - Ahaan Tagare, Student ID - 33865799
% Coursework is divided into 2 parts first the simple network and second
% the sunspots dataset
% 

%% Part 1 
% Network Configuration
weights = [-0.25, 0.33, 0.14, -0.17, 0.16, 0.43, 0.21, -0.25, 0, 0, 0];  % w1 to w11
initial_weights = weights;
lambda = 1e-5;              % Regularization parameter
epochs = 100;               % Epocs number
X = [1; 0];                 % Input [X1; X2]
target = 1;                 % Target output
hidden_neurons = 3;         % y1, y2, y3
num_weights = 11;           % Total weights

sigmoid = @(x) 1./(1 + exp(-x));  % Sigmoid activation function
sigmoid_prime = @(x) sigmoid(x) .* (1 - sigmoid(x));  % Sigmoid derivative

% Training Logging
error_history = zeros(epochs, 1);

% Main Training Loop
fprintf('Starting Newton Optimization for %d epochs...\n', epochs);
for epoch = 1:epochs
    % Forward Pass
    z = zeros(hidden_neurons, 1);
    y = zeros(hidden_neurons, 1);
    
    % Hidden layer activations with specified weights
    for n = 1:hidden_neurons
        z(n) = weights(n) * X(1) + weights(8 + n) * X(2);  % y1, y2, y3
    end
    y = sigmoid(z);  % Apply sigmoid to hidden layer outputs
    
    % Output layer (Direct input-to-output and hidden-to-output contributions)
    output = weights(2) * X(1) + weights(11) * X(2);  % Direct input-to-output weights
    for n = 1:hidden_neurons
        output = output + weights(5 + n) * y(n);  % Hidden-to-output weights (w6, w7, w8)
    end
    
    error = output - target;
    error_history(epoch) = abs(error);
    
    % (Jacobian Calculation) Backward Pass 
    dy_dz = sigmoid_prime(z);  % Derivatives of hidden outputs
    
    % Jacobian initialization
    J = zeros(1, num_weights);
    
    % Input to hidden weights
    input_to_hidden = [1, 9; 10, 4; 3, 5];  % [w1, w9; w10, w4; w3, w5]
    for n = 1:hidden_neurons
        J(input_to_hidden(n, 1)) = weights(5 + n) * dy_dz(n) * X(1);  
        J(input_to_hidden(n, 2)) = weights(5 + n) * dy_dz(n) * X(2);  
    end
    
    % Hidden to output weights (w6, w7, w8)
    for n = 1:hidden_neurons
        J(5 + n) = y(n);
    end
    
    % Direct input-to-output weights
    J(2) = X(1);  % w2
    J(11) = X(2);  % w11
    
    % Hessian and Regularization (Approximate Hessian)
    H = J' * J + lambda * eye(num_weights);  % Hessian approximation (regularization)
    
    % Weight Update using Newton's Method
    gradient = J' * error;  % Gradient (First-order derivative)
    delta_w = H \ gradient;  %  Solving H * delta_w = gradient with Newton step 
    weights = weights - delta_w(:)';  % Update weights
end

% Final Results
fprintf('\nTraining Complete\n');
fprintf('Final Error: %.6f\n', abs(error));
fprintf('\nFinal Jacobian Matrix:\n'); disp(J);
fprintf('Total Weight Changes (Initial to Final):\n');
for w_idx = 1:num_weights
    fprintf('w%d: Change = %.6f\n', w_idx, weights(w_idx) - initial_weights(w_idx));
end
fprintf('\nFinal Weights:\n'); disp(weights);

% Plotting the training error progression
figure;
semilogy(1:epochs, error_history);
title('Training Error Progression');
xlabel('Epoch'); ylabel('Absolute Error');
grid on;



%% Part 2
% Load and preprocess sunspot data
load sunspot.dat
sunspots = sunspot(:,2);

% Normalize using Z-score
mu = mean(sunspots);
sigma = std(sunspots);
sunspots = (sunspots - mu)/sigma;

% Create time-lagged input (10 previous values)
inputSize = 10;
[X, Y] = deal([]);
for i = 1:length(sunspots)-inputSize
    X = [X; sunspots(i:i+inputSize-1)'];
    Y = [Y; sunspots(i+inputSize)];
end

% Train-test split (80-20)
split = floor(0.8*size(X,1));
X_train = X(1:split,:);
Y_train = Y(1:split);
X_test = X(split+1:end,:);
Y_test = Y(split+1:end);

% Network parameters
hiddenNodes = 5;
outputSize = 1;
epochs = 100;
lambda = 1e-3;  
learningRateBP = 0.1;

% Activation functions
sigmoid = @(x) 1./(1 + exp(-x));
sigmoid_prime = @(x) sigmoid(x).*(1 - sigmoid(x));

% Initialize weights using Xavier initialization
rng(42);
W1 = randn(hiddenNodes, inputSize) * sqrt(2/(inputSize + hiddenNodes));
b1 = zeros(hiddenNodes, 1);
W2 = randn(outputSize, hiddenNodes) * sqrt(2/(hiddenNodes + outputSize));
b2 = zeros(outputSize, 1);

% Clone weights for both methods
[W1_BP, b1_BP, W2_BP, b2_BP] = deal(W1, b1, W2, b2);
[W1_NM, b1_NM, W2_NM, b2_NM] = deal(W1, b1, W2, b2);

% Loss tracking arrays
lossBP = zeros(epochs,1);
lossNM = zeros(epochs,1);

%  ---Backpropagation Training ---
for epoch = 1:epochs
    epochLoss = 0;
    for i = 1:size(X_train,1)
        % Forward pass
        Z1 = W1_BP * X_train(i,:)' + b1_BP;
        A1 = sigmoid(Z1);
        Z2 = W2_BP * A1 + b2_BP;
        
        % Backward pass
        error = Y_train(i) - Z2;
        epochLoss = epochLoss + error^2;
        
        dW2 = error * A1';
        db2 = error;
        dA1 = W2_BP' * error;
        dZ1 = dA1 .* sigmoid_prime(Z1);
        dW1 = dZ1 * X_train(i,:);
        db1 = dZ1;
        
        % Update weights
        W1_BP = W1_BP + learningRateBP * dW1;
        b1_BP = b1_BP + learningRateBP * db1;
        W2_BP = W2_BP + learningRateBP * dW2;
        b2_BP = b2_BP + learningRateBP * db2;
    end
    lossBP(epoch) = epochLoss/size(X_train,1);
    fprintf('Epoch %d (BP): Loss = %.4f\n', epoch, lossBP(epoch));
end

% --- Newton's Method Training 
numWeights = numel(W1_NM) + numel(b1_NM) + numel(W2_NM) + numel(b2_NM);
N = size(X_train,1);  % training samples

for epoch = 1:epochs
    H = zeros(numWeights);
    grad = zeros(numWeights, 1);
    epochLoss = 0;
    
    % Implementation of Equation 4.108 
    for i = 1:N
        % Forward pass
        Z1 = W1_NM * X_train(i,:)' + b1_NM;
        A1 = sigmoid(Z1);
        Z2 = W2_NM * A1 + b2_NM;
        
        % Error calculation
        error = Y_train(i) - Z2;
        epochLoss = epochLoss + error^2;
        
        % Jacobian calculation
        dZ2 = -1; 
        J_W2 = dZ2 * A1';
        J_b2 = dZ2;
        
        dA1 = W2_NM' * dZ2;
        dZ1 = dA1 .* sigmoid_prime(Z1);
        J_W1 = dZ1 * X_train(i,:);
        J_b1 = dZ1;
        
        % Construct full Jacobian
        J = [J_W1(:); J_b1(:); J_W2(:); J_b2(:)];
        
        % Accumulate (Equation 4.108 from the book)
        H = H + (J * J');  
        grad = grad + J * error;  % Gradient term
    end
    
    % Finalize Hessian 
    H = H/N + lambda*eye(numWeights);
    grad = grad/N;
    
    % Newton update (Equation 4.121)
    if rcond(H) < 1e-15
        delta = -pinv(H) * grad; 
    else
        delta = -H \ grad;  % Δw = -H⁻¹g
    end
    
    % Update weights
    weights = [W1_NM(:); b1_NM(:); W2_NM(:); b2_NM(:)] + delta;
    
    % Reshape weights
    W1_NM = reshape(weights(1:numel(W1_NM)), size(W1_NM));
    b1_NM = weights(numel(W1_NM)+1:numel(W1_NM)+numel(b1_NM));
    W2_NM = reshape(weights(numel(W1_NM)+numel(b1_NM)+1:end-numel(b2_NM)), size(W2_NM));
    b2_NM = weights(end);
    
    lossNM(epoch) = epochLoss/N;
    fprintf('Epoch %d (NM): Loss = %.4f\n', epoch, lossNM(epoch));
end

% Prediction function
predict = @(W1, b1, W2, b2, X) arrayfun(@(i) ...
    W2 * sigmoid(W1 * X(i,:)' + b1) + b2, 1:size(X,1))';

% Generate predictions
Y_pred_BP = predict(W1_BP, b1_BP, W2_BP, b2_BP, X_test);
Y_pred_NM = predict(W1_NM, b1_NM, W2_NM, b2_NM, X_test);

% NMSE
NMSE_BP = mean((Y_test - Y_pred_BP).^2)/var(Y_test);
NMSE_NM = mean((Y_test - Y_pred_NM).^2)/var(Y_test);

fprintf('Final Backpropagation NMSE: %.4f\n', NMSE_BP);
fprintf('Final Newton''s Method NMSE: %.4f\n', NMSE_NM);


% Plot Training Loss Comparison (Newton's Method vs Backpropagation)
figure;
hold on;
plot(lossBP, 'b', 'LineWidth', 1.5); % Backpropagation loss in blue
plot(lossNM, 'r', 'LineWidth', 1.5); % Newton's Method loss in red
title('Training Loss Comparison');
xlabel('Epoch');
ylabel('Mean Squared Error');
legend('Backpropagation', 'Newton''s Method');
grid on;

% Plot Sunspot Prediction Comparison (Actual vs Predicted)
figure;
hold on;
plot(Y_test, 'k', 'LineWidth', 2); % Actual values ( black)
plot(Y_pred_BP, 'b', 'LineWidth', 1.5); % Backpropagation (blue)
plot(Y_pred_NM, 'r', 'LineWidth', 1.5); % Newton's Method (red)
title('Sunspot Prediction Comparison');
xlabel('Time Step');
ylabel('Normalized Sunspot Number');
legend('Actual', 'Backpropagation', 'Newton''s Method');
grid on;