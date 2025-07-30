import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from dataset import Mammogramddataset, transform_val_test
from model_2 import Agepredictionmodel

def calculate_test_error():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    excel_file = '/home/anuvind.pramod/thesis/MINI-DDSM-Complete-JPEG-8/DataWMask.xlsx'
    root_dir = '/home/anuvind.pramod/thesis/MINI-DDSM-Complete-JPEG-8'
    model_path = '/home/anuvind.pramod/thesis/age_prediction_model_final.pth'

    df = pd.read_excel(excel_file, engine='openpyxl')
    df['fullPath'] = df['fullPath'].str.replace('\\', '/', regex=False)
    df = df[df["Status"] == "Normal"].copy()
    df = df.dropna(subset=['Age', 'fullPath', 'Density'])
    df['PatientID'] = df['fileName'].apply(lambda x: x.split("_")[1])
    all_patient_ids = df['PatientID'].unique()
    _, test_ids = train_test_split(all_patient_ids, test_size=0.15, random_state=42)
    test_df = df[df['PatientID'].isin(test_ids)]

    test_dataset = Mammogramddataset(df=test_df, root_dir=root_dir, transform=transform_val_test)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

   
    model =Agepredictionmodel().to(device)
    num_features=model.base_model.features.norm5.num_features
    model.age_predictor=nn.Sequential(
        nn.Linear(num_features+1, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

   
    all_predictions = []
    all_true_ages = []
    with torch.no_grad():
        for batch in test_loader:
            images, ages, densities = batch['images'].to(device), batch['age'].numpy(), batch['density'].to(device)
            
            features = model.base_model.features(images.view(-1, 3, 299, 299))
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
            pooled_features = torch.flatten(pooled_features, 1)
            features_vec = pooled_features.view(images.shape[0], images.shape[1], -1)
            averaged_features = torch.mean(features_vec, dim=1)
            
            # --- FIX: Add a dimension here as well ---
            combined_features = torch.cat([averaged_features, densities.unsqueeze(1)], dim=1)
            
            predicted_ages = model.age_predictor(combined_features).cpu().numpy().flatten()
            
            all_predictions.extend(predicted_ages)
            all_true_ages.extend(ages)

    mae = mean_absolute_error(all_true_ages, all_predictions)
    rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_true_ages))**2))

    mse= np.mean((np.array(all_predictions) - np.array(all_true_ages))**2)
    
    print("\n--- Test Set Performance ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.4f}")
    print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}")
    

if __name__ == '__main__':
    calculate_test_error()
