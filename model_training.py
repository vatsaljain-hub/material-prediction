#Material Prediction Model with Feature Engineering
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
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
        
        # Display feature importance
        if forecaster.feature_importance is not None:
            print("\nTop 10 Most Important Features:")
            print(forecaster.feature_importance.head(10).to_string(index=False))
        
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
