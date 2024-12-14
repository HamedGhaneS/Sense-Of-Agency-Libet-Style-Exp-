function [finalSvmModel, mean_cv_accuracy, X_channels] = SVM_Leave1Out_1(X_channels, X_train, y_train)

    % This function performs Leave-One-Out cross-validation on a Support Vector Machine (SVM) model with a Gaussian kernel.
    % As in the previous implementation with the LDA model, it first removes a set of indices from the data and then proceeds with cross-validation.
    % The function iterates over a range of box constraint (C) and kernel scale values to find the best performing combination.
    % The best parameters are then used to train an SVM model.
    % However, in contrast to models like LDA, it is not straightforward to determine the importance of features in an SVM.
    % Therefore, this function does not calculate the weights of the channels.
    % The function finally returns the trained SVM model, cross-validation accuracy, and the data structures related to channels.
    % The function is specifically designed for EEG data where each channel corresponds to a specific location on the scalp.

    % Define the names of the channels
    channel_names = {'F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'Cz'};

    addpath('E:\My Educational Documents\2nd MSc\Academic\Master Thesis\June 2023\Data Analysis Phase');


    % Exclude every 10th index
    indices_to_exclude = 1:10:151;
    all_indices = 1:size(X_train, 2);
    indices_to_keep = setdiff(all_indices, indices_to_exclude);

    % Retain only columns with indices to keep
    X_train = X_train(:, indices_to_keep);
    X_channels = X_channels(:, indices_to_keep);

    % Initialize variables for samples and cross-validation accuracy
    n_samples = size(X_train, 1);
    cv_accuracies = zeros(1, n_samples);

    % Initialize possible box constraint and kernel scale values and array to store best values
    box_constraint_values = logspace(-2,2,5); % Box constraints (C)
    kernel_scale_values = logspace(-2,2,5);   % Kernel scales (sigma)
    best_params = zeros(n_samples, 2);

    % Begin cross-validation
    for i = 1:n_samples
        % Segment training and validation data
        X_train_fold = X_train([1:i-1, i+1:end], :);
        y_train_fold = y_train([1:i-1, i+1:end]);
        X_val_fold = X_train(i, :);
        y_val_fold = y_train(i);
        
        X_channels_fold = X_channels([1:i-1, i+1:end], :);

        % Initialize accuracy array
        param_accuracies = zeros(length(box_constraint_values), length(kernel_scale_values));

        % Train an SVM classifier for each combination of C and sigma
        for j = 1:length(box_constraint_values)
            for k = 1:length(kernel_scale_values)
                svmModel = fitcsvm(X_train_fold, y_train_fold, 'KernelFunction', 'gaussian', 'BoxConstraint', box_constraint_values(j), 'KernelScale', kernel_scale_values(k));
                y_pred_fold = predict(svmModel, X_val_fold);
                confusionMatrix = confusionmat(y_val_fold, y_pred_fold);
                param_accuracies(j, k) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
            end
        end

        % Get best parameters and retrain SVM model
        [best_param_values, best_param_index] = max(param_accuracies(:));
        [best_C_index, best_scale_index] = ind2sub(size(param_accuracies), best_param_index);
        best_C = box_constraint_values(best_C_index);
        best_scale = kernel_scale_values(best_scale_index);
        best_params(i,:) = [best_C, best_scale];
        svmModel = fitcsvm(X_train_fold, y_train_fold, 'KernelFunction', 'gaussian', 'BoxConstraint', best_C, 'KernelScale', best_scale);
        y_pred_fold = predict(svmModel, X_val_fold);
        confusionMatrix = confusionmat(y_val_fold, y_pred_fold);
        cv_accuracies(i) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
    end

    % Compute final SVM model and cross-validation accuracy
    mean_cv_accuracy = mean(cv_accuracies);
    mean_params = mean(best_params,1);
    finalSvmModel = fitcsvm(X_train, y_train, 'KernelFunction', 'gaussian', 'BoxConstraint', mean_params(1), 'KernelScale', mean_params(2));

end
