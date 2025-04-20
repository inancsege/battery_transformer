import xgboost as xgb

class SOHXGBoost:
    """Wrapper class for XGBoost Regressor to fit the project structure."""
    def __init__(self, **params):
        # Filter out non-XGBoost params if necessary, though XGBoost ignores them
        xgb_params = {
            'objective': params.get('objective', 'reg:squarederror'),
            'n_estimators': params.get('n_estimators', 100),
            'learning_rate': params.get('learning_rate', 0.1),
            'max_depth': params.get('max_depth', 3),
            'subsample': params.get('subsample', 1.0),
            'colsample_bytree': params.get('colsample_bytree', 1.0),
            'tree_method': params.get('tree_method', 'hist'), # Consider 'gpu_hist'
            'eval_metric': params.get('eval_metric', 'rmse'),
            'early_stopping_rounds': params.get('early_stopping_rounds', None),
            'random_state': params.get('random_state', 42)
        }
        # Handle GPU device if specified and available
        if params.get('tree_method') == 'gpu_hist':
            try:
                # Check CUDA availability for XGBoost (might need separate check than torch)
                # For simplicity, assume if 'gpu_hist' is set, user intends GPU
                print("Using GPU for XGBoost")
            except Exception as e:
                print(f"Warning: Could not confirm GPU for XGBoost, falling back. Error: {e}")
                xgb_params['tree_method'] = 'hist'

        self.model = xgb.XGBRegressor(**xgb_params)
        self._is_fitted = False

    def fit(self, X_train, y_train, eval_set=None, verbose=True):
        """Trains the XGBoost model."""
        print(f"Fitting XGBoost with params: {self.model.get_params()}")
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
        self._is_fitted = True

    def predict(self, X_test):
        """Makes predictions with the trained model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X_test)

    def save_model(self, path):
        """Saves the trained XGBoost model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving.")
        self.model.save_model(path)
        print(f"XGBoost model saved to {path}")

    def load_model(self, path):
        """Loads a trained XGBoost model."""
        self.model.load_model(path)
        self._is_fitted = True # Assume loaded model is fitted
        print(f"XGBoost model loaded from {path}")

    def eval(self): # For compatibility with torch .eval()
        """Sets model to evaluation mode (no effect for XGBoost predict)."""
        pass # XGBoost doesn't have train/eval modes like PyTorch

    def train(self): # For compatibility with torch .train()
        """Sets model to training mode (no effect for XGBoost predict)."""
        pass

    def to(self, device): # For compatibility with torch .to()
        """Handles device compatibility (mainly for consistency)."""
        print(f"XGBoost: Device parameter '{device}' ignored for CPU/GPU, set via 'tree_method'.")
        return self # Return self to allow chaining

    def state_dict(self): # For compatibility
        """Returns None, as XGBoost models are saved/loaded differently."""
        return None

    def load_state_dict(self, state_dict, strict=True): # For compatibility
        """Does nothing, models loaded via load_model."""
        print("Warning: load_state_dict called on XGBoost wrapper. Use load_model(path) instead.")
        pass
