import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"

def load_model(model_path=MODEL_PATH):
    
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model    

def predict_invoice_flag(input_data):
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df)
    return input_df

if __name__ == "__main__":
    sample_input = {
        'invoice_quantity': [10, 20, 30],
        'invoice_dollars': [1000, 2000, 3000],
        'Freight': [50, 100, 150],
        'total_item_quantity': [100, 200, 300],
        'total_item_dollars': [10000, 20000, 30000]
    }
    predictions = predict_invoice_flag(sample_input)
    print(predictions)

