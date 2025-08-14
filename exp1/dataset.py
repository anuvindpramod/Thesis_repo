import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

transform_train=transforms.Compose([transforms.Resize((299,299)),transforms.RandomHorizontalFlip(p=0.5),transforms.ColorJitter(brightness=0.3,contrast=0.2),transforms.RandomRotation(degrees=15),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform_val_test=transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class Mammogramddataset(Dataset):
    def __init__(self,df,root_dir, transform=None):
        self.df=df
        self.root_dir = root_dir
        self.transform = transform
        self.patient_groups = self.df.groupby('PatientID')
        self.patient_ids =list(self.patient_groups.groups.keys())
        
        self.samples=[]
        for patient_id in self.patient_ids:
            group = self.patient_groups.get_group(patient_id)
            views = {}
            for _, row in group.iterrows():
                view_type = f"{row['Side']}_{row['View']}"
                views[view_type] = os.path.join(self.root_dir, row['fullPath'])

            if all(v in views for v in ['LEFT_CC', 'LEFT_MLO', 'RIGHT_CC', 'RIGHT_MLO']):
                self.samples.append({
                    'patient_id': patient_id,
                    'views': views,
                    'age': float(group['Age'].iloc[0])
                })
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample=self.samples[index]
        image_views=[]

        for view_name in ['LEFT_CC', 'LEFT_MLO', 'RIGHT_CC', 'RIGHT_MLO']:
            image_path = sample['views'][view_name]
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            image_views.append(image)

        image_tensor = torch.stack(image_views)
        return {
            'images': image_tensor,
            'age': torch.tensor(sample['age'], dtype=torch.float32)
        }
    

        
'''if __name__ == "__main__":
    excel_file = '/home/anuvind.pramod/thesis/MINI-DDSM-Complete-JPEG-8/DataWMask.xlsx'
    root_dir='/home/anuvind.pramod/thesis/MINI-DDSM-Complete-JPEG-8'
    df=pd.read_excel(excel_file,engine='openpyxl')
    df=df[df["Status"]=="Normal"].copy()
    df=df.dropna(subset=['Age','fullPath','Density'])
    df['PatientID'] = df['fileName'].apply(lambda x: x.split("_")[1])

    all_patient_ids = df['PatientID'].unique()
    train_val_ids, test_ids = train_test_split(all_patient_ids, test_size=0.15, random_state=42)
    train_ids,val_ids=train_test_split(train_val_ids,test_size=0.1765, random_state=42)# 0.1765 * 0.85 = 0.15

    train_df=df[df['PatientID'].isin(train_ids)]
    val_df=df[df['PatientID'].isin(val_ids)]
    test_df=df[df['PatientID'].isin(test_ids)]
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    transform_train=transforms.Compose([transforms.Resize((299,299)),transforms.RandomHorizontalFlip(p=0.5),transforms.ColorJitter(brightness=0.3,contrast=0.2),transforms.RandomRotation(degrees=15),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val_test=transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset=Mammogramddataset(df=train_df, root_dir=root_dir, transform=transform_train)
    val_dataset=Mammogramddataset(df=val_df, root_dir=root_dir, transform=transform_val_test)
    test_dataset=Mammogramddataset(df=test_df, root_dir=root_dir, transform=transform_val_test)
    
    train_loader=DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader=DataLoader(test_dataset, batch_size=8, shuffle=False)
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")

'''

