import os

import ROOT
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    # Load data.
    train = unpickle('data/data_batch_1')
    X_train = train[b'data']
    y_train = np.array(train[b'labels'])
    test = unpickle('data/test_batch')
    X_test = test[b'data']
    y_test = np.array(test[b'labels'])

    # Normalize.
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.uint8))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.uint8))
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Save test samples.
    np.savetxt('data/X.csv', X_test, delimiter=',')
    np.savetxt('data/y.csv', y_test, delimiter=',')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_model = Model().to(device)
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        torch_model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            out = torch_model(data[0])  # A single forward pass.
            loss = criterion(out, data[1])  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    @torch.no_grad()
    def test(loader, write=False):
        torch_model.eval()

        if write:
            with open('result.csv', 'w') as f:
                f.write('')

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            out = torch_model(data[0])

            if write:
                with open('result.csv', 'a') as f:
                    for i in out.cpu().numpy():
                        for j in i:
                            f.write(str(j) + ",")
                        f.write('\n')

            pred = out.argmax(dim=1)  # Use the class with the highest probability.
            correct += int((pred == data[1]).sum())  # Check against ground-truth labels.

        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    for epoch in range(1, 50):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    _ = test(test_loader, True)

    # Save state dictionary.
    torch.save(torch_model.state_dict(), 'model_dict.pt')

    # Save model.
    model_script = torch.jit.script(torch_model)
    torch.jit.save(model_script, 'model_script.pt')

    # Generate ROOT model.
    model = ROOT.TMVA.Experimental.SOFIE.RModel_TorchGNN(['X'], [[-1, 3072]])
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Linear('X', 3072, 200), 'linear_1')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_ReLU('linear_1'), 'relu_1')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Linear('relu_1', 200, 200), 'linear_2')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_ReLU('linear_2'), 'relu_2')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Linear('relu_2', 200, 10), 'linear_3')
    model.addModule(ROOT.TMVA.Experimental.SOFIE.RModule_Softmax('linear_3'), 'softmax')
    model.extractParameters(torch_model)
    model.save(os.getcwd(), 'Model', True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.activation = torch.nn.ReLU()

        self.linear_1 = torch.nn.Linear(3072, 200)
        self.linear_2 = torch.nn.Linear(200, 200)
        self.linear_3 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.activation(x)
        x = self.linear_3(x)
        x = self.softmax(x)
        return x


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    main()
