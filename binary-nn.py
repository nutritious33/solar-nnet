import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pytictoc import TicToc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

X = np.loadtxt('fault.txt') # data 
labels = np.loadtxt('faultlabel.txt')
Y = np.argmax(labels, axis=1) # labels

# input data features
dframe = pd.DataFrame(X, columns=['Vmp', 'Imp', 'Temperature', 'Irradiance', 'Isc', 'Gamma', 'Power', 'Voc', 'Fill Fector'])
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(dframe)
X_scaled.shape
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64) 
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

tensor_x = torch.tensor(X_train, dtype=torch.float32) 
tensor_y = torch.tensor(Y_train, dtype=torch.long) 
train_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# training loop
t = TicToc()
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100
criterion = nn.CrossEntropyLoss()
t.tic()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}')

time1 = t.tocvalue()
print(f"Training Time: {t.tocvalue():.4f}") # total training time

tensor_xTest = torch.tensor(X_test, dtype=torch.float32)
tensor_yTest = torch.tensor(Y_test, dtype=torch.long)
dataset_test = TensorDataset(tensor_xTest, tensor_yTest)
test_loader = DataLoader(dataset_test, batch_size=32) 

# model evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1) # predicted class indices 
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f'Test set accuracy: {accuracy * 100:.2f}%')

# confusion matrix and heatmap
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix")
print(cm)
labels = ['Fault', 'No Fault']
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
df_cm_normalized = df_cm.div(df_cm.sum(axis=1), axis=0) # normalize by true class counts

# plotting
sb.heatmap(df_cm_normalized, annot=True, fmt='.2%', cmap="Blues")
plt.title(f"Binary Fault Detection - Accuracy: {accuracy*100:.2f}%")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
