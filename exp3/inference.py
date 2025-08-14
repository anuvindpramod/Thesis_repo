import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import tqdm

from dataset import Mammogramddataset, transform_val_test
from model_3 import Agepredictionmodel

def calculate_test_error():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    excel_file = '/home/anuvind.pramod/thesis/MINI-DDSM-Complete-JPEG-8/DataWMask.xlsx'
    root_dir = '/home/anuvind.pramod/thesis/MINI-DDSM-Complete-JPEG-8'
    model_path = '/home/anuvind.pramod/thesis/exp3/age_prediction_model_final_b16.pth'

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
    num_combined_features = num_features + 1
    model.mean_predictor=nn.Sequential(
        nn.Linear(num_combined_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    ).to(device)
    model.variance_predictor=nn.Sequential(
        nn.Linear(num_combined_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

   
    all_predictions = []
    all_true_ages = []
    all_uncertainties = []
    test_loop=tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch in test_loop:
            images, ages, densities = batch['images'].to(device), batch['age'].numpy(), batch['density'].to(device)
            
            features = model.base_model.features(images.view(-1, 3, 299, 299))
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
            pooled_features = torch.flatten(pooled_features, 1)
            features_vec = pooled_features.view(images.shape[0], images.shape[1], -1)
            averaged_features = torch.mean(features_vec, dim=1)
            

            combined_features = torch.cat([averaged_features, densities.unsqueeze(1)], dim=1)
            
            predicted_mean, predicted_variance = model(combined_features)
            predicted_ages = predicted_mean.cpu().numpy().flatten()
            
            all_predictions.extend(predicted_ages)
            all_true_ages.extend(ages)
            uncertainity=torch.sqrt(F.softplus(predicted_variance)+1e+6).cpu().numpy().flatten()
            all_uncertainties.extend(uncertainity)

    mae = mean_absolute_error(all_true_ages, all_predictions)
    mse = mean_squared_error(all_true_ages, all_predictions)
    rmse = np.sqrt(mse)
    avg_uncertainity= np.mean(all_uncertainties)
    print("\n--- Test Set Performance ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.4f}")
    print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}")
    print(f"\nAverage Predicted Uncertainty (Age in years): {avg_uncertainity:.4f}")
    

if __name__ == '__main__':
    calculate_test_error()
