# 🎯 How to Check New/Unseen Data for Fraud

## Quick Answer

To predict fraud on **new, unseen data**, use the prediction script I created:

```bash
# 1. Activate virtual environment
.\venv\Scripts\activate

# 2. Run prediction on your new data
python predict_new_data.py --transactions YOUR_NEW_TRANSACTIONS.csv --customers YOUR_NEW_CUSTOMERS.csv
```

---

## 📋 Step-by-Step Guide

### Step 1: Prepare Your New Data

Your new CSV files must have the **same columns** as the training data:

**Transactions File:**
- customerEmail, transactionId, orderId
- paymentMethodId, paymentMethodType, paymentMethodProvider
- transactionAmount, transactionFailed, orderState
- etc.

**Customers File:**
- customerEmail, customerPhone, customerDevice
- customerIPAddress, No_Transactions, No_Orders
- etc.

### Step 2: Run Prediction

```bash
cd d:\ecommerce-fraud-detection
.\venv\Scripts\activate
python predict_new_data.py --transactions new_data.csv --customers new_customers.csv --output results.csv
```

### Step 3: Review Results

The script creates `results.csv` with:
- ✅ **Predicted_Fraud**: 0 (safe) or 1 (fraud)
- ✅ **Fraud_Probability**: 0.0 to 1.0 (confidence score)
- ✅ **Risk_Level**: Low/Medium/High

---

## 🔍 What the Script Does

```
┌─────────────────────────────────────────────────────────────┐
│  1. Load Your New Data                                      │
│     ├─ Transactions CSV                                     │
│     └─ Customers CSV                                        │
│                                                              │
│  2. Merge & Preprocess (same as training)                   │
│     ├─ Merge on customerEmail                               │
│     ├─ Handle missing values                                │
│     ├─ Encode categories                                    │
│     └─ Scale features                                       │
│                                                              │
│  3. Load Trained Model                                      │
│     └─ Random Forest (94.5% recall)                         │
│                                                              │
│  4. Make Predictions                                        │
│     ├─ Predict fraud (0 or 1)                               │
│     ├─ Calculate probability (0.0 to 1.0)                   │
│     └─ Assign risk level (Low/Medium/High)                  │
│                                                              │
│  5. Generate Results                                        │
│     ├─ Save predictions to CSV                              │
│     ├─ Show summary statistics                              │
│     └─ Highlight high-risk transactions                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 Example Output

```
================================================================================
🔮 FRAUD DETECTION PREDICTION SYSTEM
================================================================================

✅ Model loaded successfully
📊 Loading new data...
   Transactions: 500 records
   Customers: 150 records
   Merged: 500 records

🔧 Preprocessing...
   Features after encoding: 45
   ✅ Ready for prediction

🤖 Making predictions...
   Total transactions: 500
   Predicted fraud: 125 (25.0%)
   Predicted non-fraud: 375 (75.0%)

📋 PREDICTION SUMMARY
   High Risk:   45 (9.0%)  ⚠️
   Medium Risk: 80 (16.0%)
   Low Risk:    375 (75.0%) ✅

⚠️ HIGH-RISK TRANSACTIONS: 45
   Top 5 highest risk:
   transactionId  customerEmail           Amount  Probability
   tx_12345      fraud@example.com       $150    0.98
   tx_67890      suspicious@test.com     $200    0.95
   ...

💾 Results saved to: predictions_output.csv
✅ PREDICTION COMPLETE!
```

---

## 📊 Understanding Your Results

### Risk Levels Explained

| Risk Level | Probability | Action Required |
|------------|-------------|-----------------|
| **High** | > 70% | 🚨 **Block transaction** - Manual review required |
| **Medium** | 30-70% | ⚠️ **Flag for review** - Enhanced monitoring |
| **Low** | < 30% | ✅ **Allow** - Normal processing |

### Output Columns

```csv
transactionId,customerEmail,transactionAmount,Predicted_Fraud,Fraud_Probability,Risk_Level
tx_001,john@email.com,50,0,0.15,Low
tx_002,fraud@email.com,200,1,0.95,High
tx_003,jane@email.com,75,0,0.45,Medium
```

---

## 🎓 Advanced Usage

### Option 1: Python Script (Programmatic)

```python
from predict_new_data import FraudPredictor

# Initialize
predictor = FraudPredictor()

# Predict
results = predictor.predict(
    transactions_path='new_transactions.csv',
    customers_path='new_customers.csv',
    output_path='my_predictions.csv'
)

# Filter high-risk
high_risk = results[results['Risk_Level'] == 'High']
print(f"Found {len(high_risk)} high-risk transactions")

# Get specific transaction
fraud_prob = results.loc[results['transactionId'] == 'tx_12345', 'Fraud_Probability'].values[0]
print(f"Transaction tx_12345 has {fraud_prob:.1%} fraud probability")
```

### Option 2: Batch Processing

```python
import glob
from predict_new_data import FraudPredictor

predictor = FraudPredictor()

# Process all files in a folder
for trans_file in glob.glob('daily_data/transactions_*.csv'):
    date = trans_file.split('_')[1].split('.')[0]
    cust_file = f'daily_data/customers_{date}.csv'
    output = f'predictions/fraud_predictions_{date}.csv'
    
    results = predictor.predict(trans_file, cust_file, output)
    print(f"Processed {date}: {results['Predicted_Fraud'].sum()} fraud cases")
```

---

## ✅ Validation (If You Have Labels)

If your new data includes the `Fraud` column (true labels), the script automatically validates:

```
✅ VALIDATION METRICS:
   Accuracy:  0.9450
   Precision: 0.9500
   Recall:    0.9450 ⭐ (catches 94.5% of fraud)
   F1-Score:  0.9475

📊 Confusion Matrix:
   True Negatives:  350 (correctly identified as safe)
   False Positives: 15  (false alarms)
   False Negatives: 10  (missed fraud)
   True Positives:  125 (correctly caught fraud)
```

---

## 🚀 Real-World Integration

### Scenario 1: Daily Batch Processing
```bash
# Run every night on today's transactions
python predict_new_data.py \
    --transactions daily/transactions_2026-01-28.csv \
    --customers daily/customers_2026-01-28.csv \
    --output reports/fraud_report_2026-01-28.csv
```

### Scenario 2: Real-Time API (Future Enhancement)
```python
# Flask API endpoint (example)
@app.route('/predict', methods=['POST'])
def predict_fraud():
    transaction_data = request.json
    # Preprocess and predict
    fraud_prob = model.predict_proba([transaction_data])[0][1]
    return {'fraud_probability': fraud_prob, 'risk': 'High' if fraud_prob > 0.7 else 'Low'}
```

---

## 📁 Files You Need

```
d:\ecommerce-fraud-detection\
├── predict_new_data.py          ← Prediction script
├── PREDICTION_GUIDE.md          ← This guide
├── models/
│   └── best_fraud_detection_model_random_forest.pkl  ← Trained model
└── venv/                        ← Virtual environment
```

---

## ❓ Common Questions

**Q: What if my new data has different columns?**
A: The script will fail. Ensure your data has the same structure as training data.

**Q: Can I use this without the Fraud column?**
A: Yes! The script works in "prediction-only" mode if no labels are provided.

**Q: How accurate is the model?**
A: 94.5% recall (catches 94.5% of fraud), 98.5% ROC-AUC (excellent discrimination).

**Q: Can I retrain the model with new data?**
A: Yes, add your new labeled data to the original dataset and re-run `fraud_detection_pipeline.py`.

---

## 🎯 Summary

**To check new data for fraud:**

1. ✅ Prepare CSV files (same format as training data)
2. ✅ Run: `python predict_new_data.py --transactions NEW.csv --customers CUST.csv`
3. ✅ Review `predictions_output.csv` for fraud predictions
4. ✅ Focus on **High Risk** transactions (>70% probability)

**That's it!** The trained model will automatically detect fraud patterns in your new data.

---

**Need more help?** Check:
- `PREDICTION_GUIDE.md` - Detailed usage guide
- `README.md` - Project overview
- `walkthrough.md` - Complete implementation details
