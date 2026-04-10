import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

def load_model(model_path=MODEL_PATH):

    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df


if __name__ == "__main__":
    
    sample_input = {
        'Dollars': [1000, 2000, 3000]
    }
    predictions = predict_freight_cost(sample_input)
    print(predictions)