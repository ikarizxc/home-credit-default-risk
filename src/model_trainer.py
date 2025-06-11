import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

class ModelTrainer():
    def __init__(self, model, params_grid=None):
        super().__init__()
        self._grid_search = None
        self._model = None
        self._set_model(model, params_grid)
        self._use_grid_search = params_grid != None
        self._test_ids = []
        
    def _set_model(self, model, model_params):
        if model_params == None:
            self._model = model
            return
            
        self._grid_search = GridSearchCV(model, model_params, scoring='roc_auc', verbose=1)
        self._model = None
    
    def _train_model(self, X_train, X_val, y_train, y_val):
        if self._use_grid_search:
            self._grid_search.fit(X_train, y_train)
            self._model = self._grid_search.best_estimator_
        else:
            self._model.fit(X_train, y_train)
            
        if hasattr(self._model, "predict_proba"):
            val_probas = self._model.predict_proba(X_val)[:, 1]
            self._roc_auc_score = roc_auc_score(y_val, val_probas)
        elif hasattr(self._model, "decision_function"):
            val_scores = self._model.decision_function(X_val)
            self._roc_auc_score = roc_auc_score(y_val, val_scores)
        else:
            raise AttributeError("Model has no method to produce scores for ROC-AUC")
            
        print(f"Model fitted")
        print(f"ROC-AUC on validation data = {self._roc_auc_score}")
        if self._use_grid_search:
            print(f"Best model params: {self._grid_search.best_params_}")
    
    def _train_val_split(self, X, y, test_size):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=True)
        return X_train, X_val, y_train, y_val
    
    def _save_submission(self, test_ids, predicted_probas, file_name):
        submission = pd.DataFrame()

        submission['SK_ID_CURR'] = test_ids
        submission['TARGET'] = predicted_probas
        
        os.makedirs('submissions', exist_ok=True)
        submission.to_csv(f'submissions/{file_name}.csv', index=False)
        print(f"Submission file saved to 'submissions/{file_name}.csv'")
        
    def train_model(self, X, y, test_size=0.3, drop_id=True):
        if drop_id:
            X = X.drop(columns='SK_ID_CURR')
        X_train, X_val, y_train, y_val = self._train_val_split(X, y, test_size)
        self._train_model(X_train, X_val, y_train, y_val)
        
    def get_submission(self, X_test, file_name, drop_id=True):
        test_ids = X_test['SK_ID_CURR']
        if drop_id:
            X_test = X_test.drop(columns='SK_ID_CURR')
        self._save_submission(test_ids, self._model.predict_proba(X_test)[:, 1], file_name)
        
    def get_feature_importance(self):
        if hasattr(self._model, 'feature_importances_'):
            return pd.DataFrame({
                'importances': self._model.feature_importances_,
                'feature': self._model.feature_names_in_
            }).sort_values(by='importances', ascending=False)
        elif hasattr(self._model, "coef_"):
            return pd.DataFrame({
                'importances': np.abs(self._model.coef_).ravel(),
                'feature': self._model.feature_names_in_
            }).sort_values(by='importances', ascending=False)
        else:
            raise AttributeError("Model you fitted doesn't has attrubite 'feature_importances_'")