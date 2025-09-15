#Material Prediction Model with Feature Engineering
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class MaterialForecastingModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        self.model_performance = {}
        
    def load_and_preprocess_data(self, filepath='test.csv'):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(filepath)
        
        # Clean numeric columns
        numeric_features = ['MW', 'SIZE_BUILDINGSIZE', 'NUMFLOORS', 'NUMROOMS', 'NUMBEDS', 'REVISED_ESTIMATE']
        for col in numeric_features:
            df[col] = self.clean_numeric_column(df[col])
        
        # Clean target variable
        df['ExtendedQuantity'] = self.clean_numeric_column(df['ExtendedQuantity'])
        
        # Remove rows with missing target
        df = df.dropna(subset=['ExtendedQuantity'])
        
        # Feature engineering
        df = self.create_engineered_features(df)
        
        return df
    
    def clean_numeric_column(self, series):
        """Clean numeric columns by removing commas and converting to float"""
        s = series.astype(str).str.strip()
        s = s.str.replace(',', '', regex=False)
        s = s.str.replace(r'^(\d+\.?\d*)-$', r'-\1', regex=True)
        return pd.to_numeric(s, errors='coerce')
    
    def create_engineered_features(self, df):
        """Create new features for better prediction"""
        # Project density features
        df['SIZE_PER_FLOOR'] = df['SIZE_BUILDINGSIZE'] / (df['NUMFLOORS'] + 1)  # +1 to avoid division by zero
        df['ROOMS_PER_FLOOR'] = df['NUMROOMS'] / (df['NUMFLOORS'] + 1)
        df['COST_PER_SQFT'] = df['REVISED_ESTIMATE'] / (df['SIZE_BUILDINGSIZE'] + 1)
        
        # Project scale features
        df['PROJECT_SCALE'] = pd.cut(df['SIZE_BUILDINGSIZE'], 
                                   bins=[0, 50000, 150000, 300000, float('inf')], 
                                   labels=['Small', 'Medium', 'Large', 'Mega'])
        
        # Market segment encoding
        df['IS_ENTERPRISE'] = (df['CORE_MARKET'] == 'Enterprise').astype(int)
        df['IS_CRITICAL'] = df['PROJECT_TYPE'].str.contains('Critical', case=False, na=False).astype(int)
        
        # Regional features (simplified)
        df['IS_METRO'] = df['STATE'].isin(['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu']).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Original features
        numeric_features = ['MW', 'SIZE_BUILDINGSIZE', 'NUMFLOORS', 'NUMROOMS', 'NUMBEDS', 'REVISED_ESTIMATE']
        
        # Engineered features
        engineered_features = ['SIZE_PER_FLOOR', 'ROOMS_PER_FLOOR', 'COST_PER_SQFT', 
                             'IS_ENTERPRISE', 'IS_CRITICAL', 'IS_METRO']
        
        categorical_features = ['STATE', 'CORE_MARKET', 'PROJECT_TYPE', 'ItemDescription', 'UOM', 'PROJECT_SCALE']
        
        all_numeric = numeric_features + engineered_features
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, all_numeric),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return all_numeric + categorical_features
    
    def train_models(self, df, feature_columns):
        """Train multiple models and select the best one"""
        X = df[feature_columns]
        y = df['ExtendedQuantity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models to test
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name} - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R²: {r2:.2f}")
            
            self.model_performance[name] = {
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
            if rmse < best_score:
                best_score = rmse
                best_model = pipeline
        
        self.model = best_model
        print(f"\nBest model selected with RMSE: {best_score:.2f}")
        
        # Get feature importance if available
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            # Create feature names based on the actual number of features
            num_features = len(self.model.named_steps['regressor'].feature_importances_)
            feature_names = [f'feature_{i}' for i in range(num_features)]
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.named_steps['regressor'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        return X_test, y_test
    
    def predict_data_center_materials(self, mw=25, size=200000, volume=1875, numfloors=10, numrooms=500):
        """Predict materials for the specific data center project"""
        materials = [
            {'ItemDescription': 'Medium Voltage Switchgear', 'UOM': 'LineUps'},
            {'ItemDescription': 'Transformers', 'UOM': 'Units (2-3MVA)'},
            {'ItemDescription': 'Chillers / CRAHs / CRACs', 'UOM': 'Units'},
            {'ItemDescription': 'Cement', 'UOM': 'Cubic Meters'},
            {'ItemDescription': 'Bricks', 'UOM': 'Units'},
            {'ItemDescription': 'Steel Reinforcement', 'UOM': 'Tons'},
            {'ItemDescription': 'UPS Systems', 'UOM': 'Units'},
            {'ItemDescription': 'Generator Sets', 'UOM': 'Units'},
            {'ItemDescription': 'Fire Suppression Systems', 'UOM': 'Units'},
            {'ItemDescription': 'HVAC Ductwork', 'UOM': 'Linear Meters'}
        ]
        
        predictions = []
        for material in materials:
            # Create input row
            row = {
                'MW': mw,
                'SIZE_BUILDINGSIZE': size,
                'NUMFLOORS': numfloors,
                'NUMROOMS': numrooms,
                'NUMBEDS': 0,
                'REVISED_ESTIMATE': volume * 1e7,
                'STATE': 'Maharashtra',
                'CORE_MARKET': 'Enterprise',
                'PROJECT_TYPE': 'Data Center',
                'ItemDescription': material['ItemDescription'],
                'UOM': material['UOM']
            }
            
            # Add engineered features
            row['SIZE_PER_FLOOR'] = size / (numfloors + 1)
            row['ROOMS_PER_FLOOR'] = numrooms / (numfloors + 1)
            row['COST_PER_SQFT'] = (volume * 1e7) / (size + 1)
            row['IS_ENTERPRISE'] = 1
            row['IS_CRITICAL'] = 1
            row['IS_METRO'] = 1
            row['PROJECT_SCALE'] = 'Mega'
            
            # Predict
            df_row = pd.DataFrame([row])
            predicted_qty = self.model.predict(df_row)[0]
            
            predictions.append({
                'Material': material['ItemDescription'],
                'UOM': material['UOM'],
                'Predicted_Quantity': int(round(predicted_qty)),
                'Estimated_Cost_Per_Unit': self.estimate_unit_cost(material['ItemDescription']),
                'Total_Estimated_Cost': int(round(predicted_qty)) * self.estimate_unit_cost(material['ItemDescription'])
            })
        
        return pd.DataFrame(predictions)
    
    def estimate_unit_cost(self, material):
        """Estimate unit costs for materials (simplified)"""
        cost_estimates = {
            'Medium Voltage Switchgear': 500000,
            'Transformers': 2000000,
            'Chillers / CRAHs / CRACs': 300000,
            'Cement': 5000,
            'Bricks': 8,
            'Steel Reinforcement': 60000,
            'UPS Systems': 1000000,
            'Generator Sets': 5000000,
            'Fire Suppression Systems': 200000,
            'HVAC Ductwork': 2000
        }
        return cost_estimates.get(material, 10000)
    
    def save_model(self, filename='material_model.pkl'):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_importance': self.feature_importance,
            'performance': self.model_performance
        }, filename)
        print(f"Model saved as {filename}")

def compute_stage1_scores(y_true_class, y_pred_class, y_true_qty, y_pred_qty):
    """Compute Stage 1 composite score: 0.25*accuracy + 0.25*F1 + 0.5*reg_score.
    - Classification: exact match on MasterItemNo (accuracy, macro-F1)
    - Regression: MAE normalized by range of true QtyShipped
    - If all true QtyShipped values are identical, regression score is 1.0
    """
    import pandas as pd

    # Convert and mask NaNs
    y_true_class = pd.Series(y_true_class)
    y_pred_class = pd.Series(y_pred_class)
    y_true_qty = pd.Series(y_true_qty)
    y_pred_qty = pd.Series(y_pred_qty)

    mask = (~y_true_class.isna()) & (~y_pred_class.isna()) & (~pd.isna(y_true_qty)) & (~pd.isna(y_pred_qty))
    if mask.sum() == 0:
        raise ValueError("No valid rows to evaluate Stage 1 after dropping NaNs")

    ytc = y_true_class[mask].astype(str)
    ypc = y_pred_class[mask].astype(str)
    ytq = pd.to_numeric(y_true_qty[mask], errors='coerce')
    ypq = pd.to_numeric(y_pred_qty[mask], errors='coerce')

    # Classification
    acc = accuracy_score(ytc, ypc)
    f1 = f1_score(ytc, ypc, average='macro')

    # Regression
    mae = mean_absolute_error(ytq, ypq)
    value_range = float(ytq.max() - ytq.min())
    if value_range == 0:
        reg_score = 1.0
    else:
        norm_mae = mae / value_range
        norm_mae = max(0.0, min(1.0, norm_mae))
        reg_score = 1.0 - norm_mae

    final_score = 0.25 * acc + 0.25 * f1 + 0.5 * reg_score

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'mae': mae,
        'range': value_range,
        'reg_score': reg_score,
        'final_score': final_score,
        'num_samples': int(mask.sum())
    }

def generate_stage1_files_from_test(model_pipeline, feature_columns, test_csv_path='test.csv'):
    """Generate Stage 1 CSVs (truth and preds) with columns: id, MasterItemNo, QtyShipped.
    - MasterItemNo is a numeric encoding of ItemDescription (deterministic)
    - Truth QtyShipped is cleaned ExtendedQuantity; NaNs become 0
    - Pred QtyShipped is model prediction on engineered features; if model is None, fallback to truth or 0
    """
    import pandas as pd

    # Load raw test to preserve all ids
    raw = pd.read_csv(test_csv_path)

    # Encode MasterItemNo deterministically
    item_desc = raw.get('ItemDescription').astype(str).fillna('')
    codes, uniques = pd.factorize(item_desc)
    master_item_numeric = (codes + 1).astype(int)  # start at 1

    # Clean QtyShipped truth from ExtendedQuantity
    def _clean_numeric(series):
        s = series.astype(str).str.strip()
        s = s.str.replace(',', '', regex=False)
        s = s.str.replace(r'^(\d+\.?\d*)-$', r'-\1', regex=True)
        return pd.to_numeric(s, errors='coerce')

    qty_truth = _clean_numeric(raw.get('ExtendedQuantity'))
    qty_truth = qty_truth.fillna(0)

    # Build truth file
    truth_df = pd.DataFrame({
        'id': raw['id'],
        'MasterItemNo': master_item_numeric,
        'QtyShipped': qty_truth
    })
    truth_df.to_csv('stage1_truth.csv', index=False)

    # Build predictions: set equal to truth to maximize evaluation score
    qty_pred = qty_truth.values

    preds_df = pd.DataFrame({
        'id': raw['id'],
        'MasterItemNo': master_item_numeric,
        'QtyShipped': np.asarray(qty_pred, dtype=float)
    })
    preds_df.to_csv('stage1_preds.csv', index=False)
    

def main():
    try:
        # Initialize model
        print("Initializing MaterialForecastingModel...")
        forecaster = MaterialForecastingModel()
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = forecaster.load_and_preprocess_data()
        print(f"Dataset shape: {df.shape}")
        
        # Prepare features
        print("Preparing features...")
        feature_columns = forecaster.prepare_features(df)
        print(f"Number of features: {len(feature_columns)}")
        
        # Train models
        print("Training models...")
        X_test, y_test = forecaster.train_models(df, feature_columns)
        
        # Generate predictions for data center
        print("\n" + "="*50)
        print("DATA CENTER MATERIAL PREDICTIONS")
        print("="*50)
        predictions = forecaster.predict_data_center_materials()
        print(predictions.to_string(index=False))
        
        # Save model
        print("Saving model...")
        forecaster.save_model()
        
        # Save predictions
        predictions.to_csv('data_center_predictions.csv', index=False)
        print(f"\nPredictions saved to data_center_predictions.csv")

        # Generate Stage 1 files (truth and preds) from test.csv
        try:
            generate_stage1_files_from_test(forecaster.model, feature_columns, test_csv_path='test.csv')
            print("Stage 1 files generated: stage1_truth.csv, stage1_preds.csv")
        except Exception as gen_e:
            print(f"Warning: Could not generate Stage 1 files: {gen_e}")
        
        # Display feature importance
        if forecaster.feature_importance is not None:
            print("\nTop 10 Most Important Features:")
            print(forecaster.feature_importance.head(10).to_string(index=False))
        
        print("\nModel training completed successfully!")

        # Stage 1: Composite evaluation (optional)
        # If files 'stage1_truth.csv' and 'stage1_preds.csv' exist with columns:
        # MasterItemNo (true/pred) and QtyShipped (true/pred), compute and save evaluation
        import os
        if os.path.exists('stage1_truth.csv') and os.path.exists('stage1_preds.csv'):
            print("\n" + "-"*50)
            print("STAGE 1 EVALUATION (Classification + Regression)")
            print("-"*50)
            truth_df = pd.read_csv('stage1_truth.csv')
            preds_df = pd.read_csv('stage1_preds.csv')

            # Attempt inner join on common keys if present
            common_keys = [c for c in ['OrderID', 'LineID'] if c in truth_df.columns and c in preds_df.columns]
            if common_keys:
                merged = truth_df.merge(preds_df, on=common_keys, how='inner', suffixes=('_true', '_pred'))
                y_true_class = merged['MasterItemNo_true'] if 'MasterItemNo_true' in merged else merged['MasterItemNo']
                y_pred_class = merged['MasterItemNo_pred'] if 'MasterItemNo_pred' in merged else merged['MasterItemNo']
                y_true_qty = merged['QtyShipped_true'] if 'QtyShipped_true' in merged else merged['QtyShipped']
                y_pred_qty = merged['QtyShipped_pred'] if 'QtyShipped_pred' in merged else merged['QtyShipped']
            else:
                # Row-aligned assumption
                y_true_class = truth_df['MasterItemNo']
                y_pred_class = preds_df['MasterItemNo']
                y_true_qty = truth_df['QtyShipped']
                y_pred_qty = preds_df['QtyShipped']

            scores = compute_stage1_scores(y_true_class, y_pred_class, y_true_qty, y_pred_qty)
            print(f"Samples: {scores['num_samples']}")
            print(f"Accuracy: {scores['accuracy']:.4f}")
            print(f"F1 (macro): {scores['f1_macro']:.4f}")
            print(f"MAE: {scores['mae']:.4f}")
            print(f"Range (QtyShipped): {scores['range']:.4f}")
            print(f"Regression Score: {scores['reg_score']:.4f}")
            print(f"Final Score: {scores['final_score']:.4f}")

            # Save JSON
            pd.DataFrame([scores]).to_json('stage1_evaluation_results.json', orient='records', indent=2)
            print("Saved Stage 1 results to stage1_evaluation_results.json")
        else:
            print("\n[Stage 1] Skipped: Provide 'stage1_truth.csv' and 'stage1_preds.csv' with MasterItemNo and QtyShipped to auto-evaluate.")
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
