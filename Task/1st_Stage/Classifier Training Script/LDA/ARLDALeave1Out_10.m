function [finalLdaModel, mean_cv_accuracy, X_channels, channel_weights, relative_channel_weights, channel_weights_table] = ARLDALeave1Out_10(X_channels, X_train, y_train)

    % This function performs Leave-One-Out cross-validation on a Linear Discriminant Analysis (LDA) model.
    % It first removes a set of indices from the data and then proceeds with cross-validation.
    % For each validation, it iterates over a range of gamma values to find the best performing one.
    % The best gamma value is then used to train an LDA model, and the model's weights are extracted.
    % These weights are assigned to channels based on the X_channels matrix.
    % The function calculates both absolute and relative weights of each channel. 
    % The relative weights are computed as percentages of the total weights.
    % The function finally returns the trained LDA model, cross-validation accuracy, 
    % channel weights, relative weights, and a table summarizing the channel weights.
    
    % Define the names of the channels
    channel_names = {'F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'Cz'};

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

    % Initialize possible gamma values and array to store best gammas
    gamma_values = 0:0.1:1;
    best_gammas = zeros(1, n_samples);

    % Begin cross-validation
    for i = 1:n_samples
        % Segment training and validation data
        X_train_fold = X_train([1:i-1, i+1:end], :);
        y_train_fold = y_train([1:i-1, i+1:end]);
        X_val_fold = X_train(i, :);
        y_val_fold = y_train(i);
        
        X_channels_fold = X_channels([1:i-1, i+1:end], :);

        % Initialize gamma accuracies
        gamma_accuracies = zeros(1, length(gamma_values));

        % Train an LDA classifier for each gamma value
        for j = 1:length(gamma_values)
            ldaModel = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'diagLinear', 'Gamma', gamma_values(j));
            y_pred_fold = predict(ldaModel, X_val_fold);
            confusionMatrix = confusionmat(y_val_fold, y_pred_fold);
            gamma_accuracies(j) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
        end

        % Get best gamma and retrain LDA model
        [~, best_gamma_index] = max(gamma_accuracies);
        best_gamma = gamma_values(best_gamma_index);
        best_gammas(i) = best_gamma;
        ldaModel = fitcdiscr(X_train_fold, y_train_fold, 'DiscrimType', 'diagLinear', 'Gamma', best_gamma);
        y_pred_fold = predict(ldaModel, X_val_fold);
        confusionMatrix = confusionmat(y_val_fold, y_pred_fold);
        cv_accuracies(i) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
    end

    % Compute final LDA model and absolute weights
    mean_cv_accuracy = mean(cv_accuracies);
    final_gamma = mean(best_gammas);
    finalLdaModel = fitcdiscr(X_train, y_train, 'DiscrimType', 'diagLinear', 'Gamma', final_gamma);
    channel_weights = zeros(16, 1);
    relative_channel_weights = zeros(16, 1);

    % If finalLdaModel is defined and has the necessary field, compute relative weights
    if exist('finalLdaModel', 'var') == 1 && isfield(finalLdaModel.Coeffs(1,2), 'Linear')
        feature_weights = finalLdaModel.Coeffs(1,2).Linear;
        X_channels_row = X_channels(1, :);
        for channel = 1:16
            channel_feature_weights = feature_weights(X_channels_row == channel);
            channel_weights(channel) = mean(abs(channel_feature_weights));
        end
        total_weight = sum(channel_weights);
        for channel = 1:16
            relative_channel_weights(channel) = (channel_weights(channel) / total_weight) * 100;  % in percentage
        end
        channel_weights_table = table(channel_names', relative_channel_weights, 'VariableNames', {'Channel', 'Relative_Weight_in_Percentage'});
        
        % Sort the rows of the table based on the 'Relative_Weight_in_Percentage' column in descending order
        channel_weights_table = sortrows(channel_weights_table, 'Relative_Weight_in_Percentage', 'descend');
        
    else
        disp('Could not extract feature weights. The finalLdaModel might not be defined or does not contain the Coeffs.Linear field.');
    end
        
end
