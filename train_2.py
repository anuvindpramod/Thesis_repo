import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model_2 import Agepredictionmodel
from dataset import Mammogramddataset, transform_train, transform_val_test

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
excel_file = '/home/anuvind.pramod/thesis/MINI-DDSM-Complete-JPEG-8/DataWMask.xlsx'
root_dir='/home/anuvind.pramod/thesis/MINI-DDSM-Complete-JPEG-8'
base_model_path = '/home/anuvind.pramod/thesis/age_prediction_model.pth'
final_model_path = '/home/anuvind.pramod/thesis/age_prediction_model_final.pth'
batch_size = 8
learning_rate = 1e-4
num_epochs = 100
writer=SummaryWriter('runs/final_model_with_density')

df=pd.read_excel(excel_file,engine='openpyxl')
df['fullPath'] = df['fullPath'].str.replace('\\', '/', regex=False)
df=df[df["Status"]=="Normal"].copy()
df=df.dropna(subset=['Age','fullPath','Density'])
df['PatientID'] = df['fileName'].apply(lambda x: x.split("_")[1])
all_patient_ids = df['PatientID'].unique()
train_val_ids, test_ids = train_test_split(all_patient_ids, test_size=0.15, random_state=42)
train_ids,val_ids=train_test_split(train_val_ids,test_size=0.1765, random_state=42)  # 0.1765 * 0
train_df=df[df['PatientID'].isin(train_ids)]
val_df=df[df['PatientID'].isin(val_ids)]
test_df=df[df['PatientID'].isin(test_ids)]
train_dataset=Mammogramddataset(df=train_df, root_dir=root_dir, transform=transform_train)
val_dataset=Mammogramddataset(df=val_df, root_dir=root_dir, transform=transform_val_test)
test_dataset=Mammogramddataset(df=test_df, root_dir=root_dir, transform=transform_val_test)
    
train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model=Agepredictionmodel().to(device)
model.load_state_dict(torch.load(base_model_path, map_location=device))

for params in model.base_model.parameters():
    params.requires_grad = False

num_features = model.base_model.features.norm5.num_features
model.age_predictor = nn.Sequential(
    nn.Linear(num_features+1, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 1)
).to(device)

optimizer=optim.AdamW(model.age_predictor.parameters(), lr=learning_rate, weight_decay=1e-2)
criterion=nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
best_val_loss = float('inf')
epochs_no_improvement = 0
early_stopping_patience = 7

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    train_loop=tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
    for batch in train_loop:
        images,ages,densities=batch['images'].to(device), batch['age'].to(device), batch['density'].to(device)
        with torch.no_grad():
            features = model.base_model.features(images.view(-1, 3, 299, 299))
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
            pooled_features = torch.flatten(pooled_features, 1)
            features_vec = pooled_features.view(images.shape[0], images.shape[1], -1)
            averaged_features = torch.mean(features_vec, dim=1)
        combined_features = torch.cat([averaged_features, densities.unsqueeze(1)], dim=1)
        predicted_ages = model.age_predictor(combined_features)
        loss = criterion(predicted_ages, ages.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())
        

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0.0
    loop=tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]')
    with torch.no_grad():
        for batch in loop:
            images, ages,densities = batch['images'].to(device), batch['age'].to(device),batch['density'].to(device)
            features = model.base_model.features(images.view(-1, 3, 299, 299))
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
            pooled_features = torch.flatten(pooled_features, 1)
            features_vec = pooled_features.view(images.shape[0], images.shape[1], -1)
            averaged_features = torch.mean(features_vec, dim=1)
            combined_features = torch.cat([averaged_features, densities.unsqueeze(1)], dim=1)
            predicted_ages = model.age_predictor(combined_features)
            
            loss = criterion(predicted_ages, ages.unsqueeze(1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    val_rmse=torch.sqrt(torch.tensor(avg_val_loss))
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation RMSE: {val_rmse:.4f}')

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('RMSE/Validation', val_rmse, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), final_model_path)
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        epochs_no_improvement = 0
    else:
        epochs_no_improvement += 1
    scheduler.step(avg_val_loss)
    if epochs_no_improvement >= early_stopping_patience:
        print(f"Early stopping triggered after {epochs_no_improvement} epochs without improvement.")
        break

writer.close()
print("Training complete. Best model saved at:", final_model_path)