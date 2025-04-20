import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from .preprocess import DataPreprocessor # Import from the new preprocess module

class SOHDataset(Dataset):
    """Standard Dataset for sequence models (Transformer, TCN, CNN)."""
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SOHDatasetLSTM(Dataset):
    """Dataset potentially specific to LSTM structure if needed."""
    def __init__(self, X, y):
        # If LSTM requires specific formatting different from SOHDataset, implement here
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X, y, seq_len):
    """Creates sequences and corresponding targets."""
    sequences = []
    targets = []

    if X.shape[0] <= seq_len:
        print(f"Warning: Data length ({X.shape[0]}) is less than or equal to sequence length ({seq_len}). Cannot create sequences.")
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    # Ensure target creation doesn't go out of bounds
    max_start_index = len(X) - seq_len - 1 # Predict the value right after the sequence

    for i in range(max_start_index + 1):
        sequences.append(X[i : i + seq_len])
        targets.append(y[i + seq_len]) # Predict the single next value

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def create_multi_step_sequences(X, y, seq_len, target_len):
    """Creates sequences and multi-step targets."""
    sequences = []
    targets = []

    if X.shape[0] <= seq_len + target_len -1:
        print(f"Warning: Data length ({X.shape[0]}) is insufficient for seq_len ({seq_len}) and target_len ({target_len}). Cannot create sequences.")
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    max_start_index = len(X) - seq_len - target_len

    for i in range(max_start_index + 1):
        sequences.append(X[i : i + seq_len])
        targets.append(y[i + seq_len : i + seq_len + target_len])

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def get_data_loaders(file_list, features, targets, seq_len, batch_size, model_type, train_split_ratio=0.7, val_split_ratio=0.15, preprocessing_config=None):
    """Loads data, preprocesses, creates sequences, and returns DataLoaders."""
    all_X_seq = []
    all_y_seq = []
    scaler_data = StandardScaler() # Default scaler
    target_scaler = None # Separate scaler for target if needed
    target_col = targets[0] # Assume single target for now
    multi_step_target_len = 1 # Default to single step prediction

    # Determine target length based on model type (example)
    if model_type in ['transformer', 'transformer_hdmr', 'tcn', 'lstm', 'cnn']:
        # These models predicted multiple steps in the original utils.py
        # Get output_dim from model config later, assume seq_len for now if not available
        multi_step_target_len = seq_len # Example: predict next 'seq_len' steps

    if preprocessing_config and preprocessing_config.get('type') == 'battery_transformer_style':
        print("Using battery_transformer_style preprocessing...")
        # Use the DataPreprocessor logic
        preprocessor = DataPreprocessor(
            voltage_col=preprocessing_config['voltage_col'],
            current_col=preprocessing_config['current_col'],
            timestamp_col=preprocessing_config['timestamp_col'],
            target_col=preprocessing_config['target_col'],
            window_size_hours=preprocessing_config.get('window_size_hours', 5),
            sampling_rate_hz=preprocessing_config.get('sampling_rate_hz', 1/60),
            downsample_freq=preprocessing_config.get('downsample_freq', '15T'),
            filter_type=preprocessing_config.get('filter_type', 'butterworth'),
            remove_negatives=preprocessing_config.get('remove_negatives', True),
            normalize=preprocessing_config.get('normalize', True)
        )
        all_processed_data = []
        final_scaler = None
        for file_path in file_list:
            try:
                processed_df, scaler = preprocessor.process_file(file_path)
                if not processed_df.empty:
                    all_processed_data.append(processed_df)
                    final_scaler = scaler # Use the last scaler, assumes consistent scaling
            except Exception as e:
                print(f"Skipping file {file_path} due to preprocessing error: {e}")

        if not all_processed_data:
            raise ValueError("No data could be processed using battery_transformer_style.")

        combined_df = pd.concat(all_processed_data, ignore_index=True)
        # Features are now derived from the processed data
        proc_features = [col for col in combined_df.columns if col not in [preprocessing_config['timestamp_col'], preprocessing_config['target_col']]]
        if not all(f in combined_df.columns for f in proc_features):
             raise ValueError(f"Feature columns mismatch after preprocessing. Expected based on processed data: {proc_features}, Available: {combined_df.columns.tolist()}")
        if preprocessing_config['target_col'] not in combined_df.columns:
             raise ValueError(f"Target column '{preprocessing_config['target_col']}' not found after preprocessing.")

        X_all = combined_df[proc_features].values
        y_all = combined_df[preprocessing_config['target_col']].values
        scaler_data = final_scaler # Store the scaler used

        # Create sequences based on the model's prediction type
        if multi_step_target_len > 1:
            X_seq_temp, y_seq_temp = create_multi_step_sequences(X_all, y_all, seq_len, multi_step_target_len)
        else:
            X_seq_temp, y_seq_temp = create_sequences(X_all, y_all, seq_len)

        all_X_seq.extend(X_seq_temp)
        all_y_seq.extend(y_seq_temp)

    else:
        print("Using standard preprocessing...")
        # Use the standard preprocessing (like in original utils.py)
        all_X = []
        all_y = []
        for file in file_list: # Consider limiting files if too large
            try:
                df = pd.read_csv(file)
                # Check if all features and targets exist
                if not all(f in df.columns for f in features):
                    print(f"Warning: Skipping file {file}. Missing features. Required: {features}, Available: {df.columns.tolist()}")
                    continue
                if not all(t in df.columns for t in targets):
                    print(f"Warning: Skipping file {file}. Missing targets. Required: {targets}, Available: {df.columns.tolist()}")
                    continue

                all_X.append(df[features].values)
                all_y.append(df[target_col].values)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue

        if not all_X:
            raise ValueError("No data loaded. Check file paths and contents.")

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)

        # Scaling
        X_all = scaler_data.fit_transform(X_all)

        # Optional: Scale target (example: normalize by max)
        max_y = y_all.max()
        if max_y > 0 : # Avoid division by zero
             y_all = y_all / max_y
             target_scaler = {'type': 'max', 'value': max_y} # Store scaling info
        else:
            print("Warning: Max target value is 0 or less. Skipping target scaling.")

        # Create sequences based on the model's prediction type
        if multi_step_target_len > 1:
            X_seq_temp, y_seq_temp = create_multi_step_sequences(X_all, y_all, seq_len, multi_step_target_len)
        else:
            X_seq_temp, y_seq_temp = create_sequences(X_all, y_all, seq_len)

        all_X_seq.extend(X_seq_temp)
        all_y_seq.extend(y_seq_temp)

    if not all_X_seq:
        raise ValueError("No sequences created. Check data length and sequence length.")

    # Split data
    X_seq_np = np.array(all_X_seq)
    y_seq_np = np.array(all_y_seq)

    X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(
        X_seq_np, y_seq_np, test_size=(1 - train_split_ratio), random_state=42
    )
    # Calculate split size for validation relative to the temp set
    relative_val_size = val_split_ratio / (1 - train_split_ratio)
    if relative_val_size >= 1.0 or relative_val_size <= 0.0:
         print(f"Warning: Invalid split ratios. Train: {train_split_ratio}, Val: {val_split_ratio}. Adjusting validation split.")
         # Default to 50/50 split of the temp data if ratios are problematic
         relative_val_size = 0.5 if (1 - train_split_ratio) > 0 else 0

    if relative_val_size > 0 and len(X_temp_seq) > 1:
        X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(
            X_temp_seq, y_temp_seq, test_size=(1 - relative_val_size), random_state=42
        )
    else: # Handle cases with very little data or only train/test split needed
        X_val_seq, y_val_seq = np.array([]), np.array([])
        X_test_seq, y_test_seq = X_temp_seq, y_temp_seq

    print(f"Data split sizes: Train={len(X_train_seq)}, Val={len(X_val_seq)}, Test={len(X_test_seq)}")

    # Create Datasets
    dataset_class = SOHDatasetLSTM if model_type == 'lstm' else SOHDataset
    train_dataset = dataset_class(X_train_seq, y_train_seq)
    val_dataset = dataset_class(X_val_seq, y_val_seq) if len(X_val_seq) > 0 else None
    test_dataset = dataset_class(X_test_seq, y_test_seq)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, {'feature_scaler': scaler_data, 'target_scaler': target_scaler}

def get_data_loaders_xgb(file_list, features, targets, test_size=0.2, val_size=0.15):
    """Loads and preprocesses data specifically for XGBoost (no sequences)."""
    all_X = []
    all_y = []
    target_col = targets[0]

    for file in file_list: # Consider limiting files
        try:
            df = pd.read_csv(file)
            if not all(f in df.columns for f in features):
                print(f"Warning: Skipping file {file}. Missing features.")
                continue
            if target_col not in df.columns:
                print(f"Warning: Skipping file {file}. Missing target column '{target_col}'.")
                continue

            # XGBoost typically predicts next step, align X and y
            all_X.append(df[features].values[:-1])
            all_y.append(df[target_col].values[1:])
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

    if not all_X:
        raise ValueError("No data loaded for XGBoost.")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    scaler_data = StandardScaler()
    X = scaler_data.fit_transform(X)

    # Optional: Scale target
    max_y = y.max()
    if max_y > 0:
        y = y / max_y
        target_scaler = {'type': 'max', 'value': max_y}
    else:
        target_scaler = None

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    # Calculate validation size relative to the remaining temp data
    relative_val_size = val_size / test_size if test_size > 0 else 0
    if relative_val_size >= 1.0 or relative_val_size <= 0.0:
        print(f"Warning: Invalid split ratios for XGB. Test: {test_size}, Val: {val_size}. Adjusting validation split.")
        relative_val_size = 0.5 if test_size > 0 else 0

    if relative_val_size > 0 and len(X_temp) > 1:
         X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - relative_val_size), random_state=42)
    else:
        X_val, y_val = np.array([]), np.array([])
        X_test, y_test = X_temp, y_temp

    print(f"XGB Data split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, {'feature_scaler': scaler_data, 'target_scaler': target_scaler}
