"""
Fraud Detection - Prediction on New Data
==========================================
Use the trained model to predict fraud on new, unseen datasets

Usage:
    python predict_new_data.py --transactions new_transactions.csv --customers new_customers.csv
    
Or run interactively for guided prediction
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

class FraudPredictor:
    """
    Fraud Detection Prediction System
    Loads trained model and makes predictions on new data
    """
    
    def __init__(self, model_path='models/best_fraud_detection_model_random_forest.pkl'):
        """Initialize predictor with trained model"""
        print("="*80)
        print("🔮 FRAUD DETECTION PREDICTION SYSTEM")
        print("="*80)
        
        try:
            self.model = joblib.load(model_path)
            print(f"\n✅ Model loaded successfully from: {model_path}")
            print(f"   Model Type: {type(self.model).__name__}")
        except FileNotFoundError:
            print(f"\n❌ Error: Model file not found at {model_path}")
            print("   Please run fraud_detection_pipeline.py first to train the model.")
            raise
    
    def load_and_merge_data(self, transactions_path, customers_path):
        """Load and merge transaction and customer data"""
        print(f"\n{'='*80}")
        print("📂 LOADING NEW DATA")
        print(f"{'='*80}")
        
        # Load datasets
        print(f"\n📊 Loading transactions from: {transactions_path}")
        transactions_df = pd.read_csv(transactions_path, index_col=0)
        print(f"   Shape: {transactions_df.shape}")
        
        print(f"\n📊 Loading customers from: {customers_path}")
        customers_df = pd.read_csv(customers_path, index_col=0)
        print(f"   Shape: {customers_df.shape}")
        
        # Merge datasets
        print(f"\n🔗 Merging datasets on 'customerEmail'...")
        df = transactions_df.merge(customers_df, on='customerEmail', how='left')
        print(f"   Merged shape: {df.shape}")
        
        # Store original data for results
        self.original_data = df.copy()
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess data same as training pipeline"""
        print(f"\n{'='*80}")
        print("🔧 PREPROCESSING DATA")
        print(f"{'='*80}")
        
        # Drop identifier columns (same as training)
        identifiers = ['transactionId', 'orderId', 'customerEmail', 'customerPhone', 
                       'customerDevice', 'customerIPAddress', 'customerBillingAddress']
        
        # Also drop Fraud column if it exists (for unseen data it might not exist)
        if 'Fraud' in df.columns:
            print("\n⚠️  Note: 'Fraud' column found in data (will be used for validation)")
            self.has_labels = True
            self.true_labels = df['Fraud'].astype(int)
            identifiers.append('Fraud')
        else:
            print("\n📝 Note: No 'Fraud' column found (prediction only mode)")
            self.has_labels = False
            self.true_labels = None
        
        df_processed = df.drop(columns=identifiers, errors='ignore')
        print(f"\n🗑️  Dropped identifier columns")
        print(f"   Remaining features: {df_processed.shape[1]}")
        
        # Identify numerical and categorical columns
        numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df_processed.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        print(f"\n📈 Numerical features: {len(numerical_features)}")
        print(f"📝 Categorical features: {len(categorical_features)}")
        
        # Handle missing values
        print(f"\n🔍 Checking missing values...")
        missing_count = df_processed.isnull().sum().sum()
        print(f"   Total missing values: {missing_count}")
        
        if missing_count > 0:
            print(f"   Handling missing values...")
            # Numerical: median imputation
            num_imputer = SimpleImputer(strategy='median')
            df_processed[numerical_features] = num_imputer.fit_transform(df_processed[numerical_features])
            
            # Categorical: mode imputation
            if len(categorical_features) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_features] = cat_imputer.fit_transform(df_processed[categorical_features])
            
            print(f"   ✅ Missing values handled")
        
        # One-Hot Encoding
        print(f"\n🔢 Encoding categorical variables...")
        df_encoded = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)
        print(f"   Features after encoding: {df_encoded.shape[1]}")
        
        # Feature Scaling
        print(f"\n⚖️  Scaling numerical features...")
        scaler = StandardScaler()
        df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
        print(f"   ✅ Features scaled")
        
        # Add transaction count feature (if possible)
        if 'customerEmail' in df.columns:
            df_encoded['transaction_count'] = df.groupby('customerEmail')['transactionId'].transform('count')
            print(f"   ✅ Added 'transaction_count' feature")
        
        print(f"\n📊 Final preprocessed shape: {df_encoded.shape}")
        
        return df_encoded
    
    def make_predictions(self, X):
        """Make fraud predictions"""
        print(f"\n{'='*80}")
        print("🔮 MAKING PREDICTIONS")
        print(f"{'='*80}")
        
        print(f"\n🤖 Running model inference...")
        print(f"   Input shape: {X.shape}")
        
        # Get predictions and probabilities
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of fraud
        
        fraud_count = predictions.sum()
        fraud_percentage = (fraud_count / len(predictions)) * 100
        
        print(f"\n✅ Predictions complete!")
        print(f"   Total transactions: {len(predictions)}")
        print(f"   Predicted fraud: {fraud_count} ({fraud_percentage:.2f}%)")
        print(f"   Predicted non-fraud: {len(predictions) - fraud_count} ({100-fraud_percentage:.2f}%)")
        
        return predictions, probabilities
    
    def generate_results(self, predictions, probabilities):
        """Generate detailed results DataFrame"""
        print(f"\n{'='*80}")
        print("📊 GENERATING RESULTS")
        print(f"{'='*80}")
        
        # Create results DataFrame
        results = self.original_data.copy()
        results['Predicted_Fraud'] = predictions
        results['Fraud_Probability'] = probabilities
        results['Risk_Level'] = pd.cut(probabilities, 
                                       bins=[0, 0.3, 0.7, 1.0],
                                       labels=['Low', 'Medium', 'High'])
        
        # If we have true labels, calculate accuracy
        if self.has_labels:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            accuracy = accuracy_score(self.true_labels, predictions)
            precision = precision_score(self.true_labels, predictions)
            recall = recall_score(self.true_labels, predictions)
            f1 = f1_score(self.true_labels, predictions)
            cm = confusion_matrix(self.true_labels, predictions)
            
            print(f"\n✅ VALIDATION METRICS (True labels available):")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f} ⭐")
            print(f"   F1-Score:  {f1:.4f}")
            
            print(f"\n📊 Confusion Matrix:")
            print(f"   True Negatives:  {cm[0][0]}")
            print(f"   False Positives: {cm[0][1]}")
            print(f"   False Negatives: {cm[1][0]}")
            print(f"   True Positives:  {cm[1][1]}")
            
            results['Actual_Fraud'] = self.true_labels
            results['Correct_Prediction'] = (predictions == self.true_labels)
        
        return results
    
    def save_results(self, results, output_path='predictions_output.csv'):
        """Save predictions to CSV"""
        results.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        return output_path
    
    def display_summary(self, results):
        """Display prediction summary"""
        print(f"\n{'='*80}")
        print("📋 PREDICTION SUMMARY")
        print(f"{'='*80}")
        
        # Risk level distribution
        print(f"\n🎯 Risk Level Distribution:")
        risk_dist = results['Risk_Level'].value_counts()
        for level in ['High', 'Medium', 'Low']:
            if level in risk_dist.index:
                count = risk_dist[level]
                pct = (count / len(results)) * 100
                print(f"   {level:8s}: {count:4d} ({pct:5.2f}%)")
        
        # High-risk transactions
        high_risk = results[results['Risk_Level'] == 'High']
        print(f"\n⚠️  HIGH-RISK TRANSACTIONS: {len(high_risk)}")
        if len(high_risk) > 0:
            print(f"\n   Top 5 highest risk transactions:")
            top_risk = high_risk.nlargest(5, 'Fraud_Probability')[
                ['transactionId', 'customerEmail', 'transactionAmount', 
                 'paymentMethodType', 'Fraud_Probability']
            ]
            print(top_risk.to_string(index=False))
        
        # Fraud by payment method
        if 'paymentMethodType' in results.columns:
            print(f"\n💳 Predicted Fraud by Payment Method:")
            fraud_by_payment = results.groupby('paymentMethodType')['Predicted_Fraud'].agg(['sum', 'count'])
            fraud_by_payment['fraud_rate'] = (fraud_by_payment['sum'] / fraud_by_payment['count'] * 100)
            fraud_by_payment = fraud_by_payment.sort_values('fraud_rate', ascending=False)
            print(fraud_by_payment.to_string())
    
    def predict(self, transactions_path, customers_path, output_path='predictions_output.csv'):
        """
        Complete prediction pipeline
        
        Args:
            transactions_path: Path to new transactions CSV
            customers_path: Path to new customers CSV
            output_path: Path to save predictions
        
        Returns:
            DataFrame with predictions
        """
        # Load and merge data
        df = self.load_and_merge_data(transactions_path, customers_path)
        
        # Preprocess
        X = self.preprocess_data(df)
        
        # Make predictions
        predictions, probabilities = self.make_predictions(X)
        
        # Generate results
        results = self.generate_results(predictions, probabilities)
        
        # Display summary
        self.display_summary(results)
        
        # Save results
        self.save_results(results, output_path)
        
        print(f"\n{'='*80}")
        print("✅ PREDICTION COMPLETE!")
        print(f"{'='*80}")
        
        return results


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Fraud Detection - Predict on New Data')
    parser.add_argument('--transactions', type=str, 
                       default='cust_transaction_details (1).csv',
                       help='Path to transactions CSV file')
    parser.add_argument('--customers', type=str,
                       default='Customer_DF (1).csv', 
                       help='Path to customers CSV file')
    parser.add_argument('--output', type=str,
                       default='predictions_output.csv',
                       help='Path to save predictions')
    parser.add_argument('--model', type=str,
                       default='models/best_fraud_detection_model_random_forest.pkl',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = FraudPredictor(model_path=args.model)
    
    # Make predictions
    results = predictor.predict(
        transactions_path=args.transactions,
        customers_path=args.customers,
        output_path=args.output
    )
    
    print(f"\n🎉 Done! Check '{args.output}' for detailed predictions.")
    
    return results


if __name__ == "__main__":
    # If run without arguments, use default files (demo mode)
    import sys
    
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("🔮 DEMO MODE - Using existing data files")
        print("="*80)
        print("\nTo use with your own data, run:")
        print("  python predict_new_data.py --transactions your_transactions.csv --customers your_customers.csv")
        print("\n" + "="*80 + "\n")
    
    main()
