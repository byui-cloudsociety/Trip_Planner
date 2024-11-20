# pip install scipy scikit-learn torch torchvision pandas numpy

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class POIDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    
        self.data['categories'] = self.data['categories'].apply(lambda x: x.split(';') if pd.notnull(x) else [])
        
    
        self.category_encoder = MultiLabelBinarizer(sparse_output=True)
        category_features = self.category_encoder.fit_transform(self.data['categories'])
        
    
        lat_lon_features = csr_matrix(self.data[['latitude_radian', 'longitude_radian']].values)
        
    
        self.feature_matrix = hstack([lat_lon_features, category_features], format='csr')

    def __len__(self):
        return self.feature_matrix.shape[0]
    
    def __getitem__(self, idx):
        features = torch.tensor(self.feature_matrix[idx].toarray().flatten(), dtype=torch.float32)
        label = self.data.iloc[idx]['name']
        return features, label


class RecommendationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
        
            outputs = model(inputs) 
            
        
            target_indices = torch.arange(len(labels)) 
            
        
            loss = criterion(outputs, target_indices)
            
        
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")



def recommend(model, user_interests, category_encoder, poi_data, top_n=5):

    interest_vector = category_encoder.transform([user_interests]).toarray()[0]
    

    avg_lat_lon = np.mean(poi_data[['latitude_radian', 'longitude_radian']].values, axis=0)
    user_input = np.hstack([avg_lat_lon, interest_vector])
    user_input_tensor = torch.tensor(user_input, dtype=torch.float32)
    

    with torch.no_grad():
        recommendations = model(user_input_tensor.unsqueeze(0))
    

    top_indices = torch.argsort(recommendations[0], descending=True)[:top_n]
    top_pois = poi_data.iloc[top_indices.numpy()]['name'].values
    return top_pois


def main():

    dataset = POIDataset('poiTrainingData.csv')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    

    input_size = dataset.feature_matrix.shape[1]    
    hidden_size = 128
    output_size = len(dataset)
    model = RecommendationModel(input_size, hidden_size, output_size)
    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    train_model(model, dataloader, optimizer, criterion)
    

    user_interests = ['ISLANDS OF TANZANIA', 'ZANZIBAR ARCHIPELAGO']
    recommendations = recommend(model, user_interests, dataset.category_encoder, dataset.data)
    print("Top recommendations:", recommendations)


if __name__ == "__main__":
    main()
