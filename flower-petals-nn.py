import torch
import torch.nn as nn  
import torch.optim as optim 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


iris = load_iris()
X = iris.data
y =iris.target

print(X)
print(y)

#split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.2, random_state=42)

#scalar
scalar = StandardScaler()
X_train = scalar.fit(X_train)
X_test = scalar.fit(X_test)
y_train = scalar.fit(y_train)
y_test = scalar.fit(y_test)


X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Create DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)