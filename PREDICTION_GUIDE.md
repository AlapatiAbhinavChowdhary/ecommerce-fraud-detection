# How to Use the Fraud Detection System on New Data

## 🎯 Quick Start

### Option 1: Use the Prediction Script (Recommended)

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Predict on new data
python predict_new_data.py --transactions new_transactions.csv --customers new_customers.csv --output results.csv
```

### Option 2: Test with Existing Data (Demo)

```bash
# This will use the original data to validate the model
python predict_new_data.py
```

## 📋 Input Data Format

Your new data files must have the **same structure** as the training data:

### Transactions CSV (`new_transactions.csv`)
Required columns:
- `customerEmail` (join key)
- `transactionId`
- `orderId`
- `paymentMethodId`
- `paymentMethodRegistrationFailure`
- `paymentMethodType`
- `paymentMethodProvider`
- `transactionAmount`
- `transactionFailed`
- `orderState`

### Customers CSV (`new_customers.csv`)
Required columns:
- `customerEmail` (join key)
- `customerPhone`
- `customerDevice`
- `customerIPAddress`
- `customerBillingAddress`
- `No_Transactions`
- `No_Orders`
- `No_Payments`
- `Fraud` (optional - only if you want to validate predictions)

## 📊 Output

The script generates `predictions_output.csv` with:
- All original columns
- `Predicted_Fraud` (0 = Non-Fraud, 1 = Fraud)
- `Fraud_Probability` (0.0 to 1.0)
- `Risk_Level` (Low/Medium/High)
- `Actual_Fraud` (if labels provided)
- `Correct_Prediction` (if labels provided)

## 🔍 Understanding Results

### Risk Levels
- **High Risk** (>70% probability): Immediate review required
- **Medium Risk** (30-70% probability): Enhanced monitoring
- **Low Risk** (<30% probability): Normal processing

### Key Metrics (if labels available)
- **Recall**: % of actual fraud cases caught (most important)
- **Precision**: % of fraud predictions that are correct
- **F1-Score**: Balance between precision and recall

## 💡 Example Usage

### Example 1: Predict on completely new data
```bash
python predict_new_data.py \
    --transactions monday_transactions.csv \
    --customers monday_customers.csv \
    --output monday_predictions.csv
```

### Example 2: Use different model
```bash
python predict_new_data.py \
    --transactions new_data.csv \
    --customers new_customers.csv \
    --model models/custom_model.pkl \
    --output custom_predictions.csv
```

### Example 3: Programmatic usage (in Python)
```python
from predict_new_data import FraudPredictor

# Initialize predictor
predictor = FraudPredictor(model_path='models/best_fraud_detection_model_random_forest.pkl')

# Make predictions
results = predictor.predict(
    transactions_path='new_transactions.csv',
    customers_path='new_customers.csv',
    output_path='predictions.csv'
)

# Access results
high_risk = results[results['Risk_Level'] == 'High']
print(f"Found {len(high_risk)} high-risk transactions")
```

## ⚠️ Important Notes

1. **Data Format**: New data must have the same columns as training data
2. **Missing Values**: The script handles missing values automatically
3. **New Categories**: If new payment methods appear, they'll be handled gracefully
4. **Model Path**: Ensure the trained model file exists before running

## 🔧 Troubleshooting

### Error: "Model file not found"
**Solution**: Run `python fraud_detection_pipeline.py` first to train the model

### Error: "Column not found"
**Solution**: Check that your CSV has all required columns

### Error: "Shape mismatch"
**Solution**: Ensure new data has same structure as training data

## 📈 Next Steps

After getting predictions:

1. **Review High-Risk Transactions**: Manually investigate fraud probability > 70%
2. **Monitor Patterns**: Track fraud rates by payment method, customer, etc.
3. **Update Model**: Retrain periodically with new labeled data
4. **Integrate**: Connect to your transaction processing system

## 🎓 Advanced Usage

### Batch Processing Multiple Files
```python
import glob
from predict_new_data import FraudPredictor

predictor = FraudPredictor()

for trans_file in glob.glob('data/transactions_*.csv'):
    cust_file = trans_file.replace('transactions', 'customers')
    output_file = f'predictions_{trans_file.split("_")[1]}'
    
    predictor.predict(trans_file, cust_file, output_file)
```

### Custom Risk Thresholds
```python
results = predictor.predict(...)

# Custom risk levels
results['Custom_Risk'] = pd.cut(
    results['Fraud_Probability'],
    bins=[0, 0.2, 0.5, 0.8, 1.0],
    labels=['Very Low', 'Low', 'Medium', 'High']
)
```

---

**Need Help?** Check the README.md or review the walkthrough.md for more details!
