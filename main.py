import numpy
import torch
from titanic_dataset import TitanicPrepare
from titanic_nn import TitanicNN

device = "mps" if torch.backends.mps.is_available() else "cpu"

train, test = TitanicPrepare("cleared_titanic.csv", "Survived").get_data()

model = TitanicNN(23, 2).to(device)

# Используем CrossEntropyLoss для многоклассовой классификации (даже для 2 классов)
loss_fn = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for index, (x, y) in enumerate(train):

        model.train()

        x, y = x.to(device), y.to(device)

        predictions = model(x)

        loss = loss_fn(predictions, y)

        if index % 50 == 0:
            print(f"Epoch {epoch}, Batch {index}, Loss: {loss.item():.4f}")

        loss.backward()
        optim.step()
        optim.zero_grad()

    print("\n" + str(epoch))

model.eval()

correct = 0
total = 0
lost = 0

for index, (x, y) in enumerate(test):

    x, y = x.to(device), y.to(device)

    predictions = model(x)
    
    # Получаем предсказанные классы (индекс максимального значения)
    pred_classes = predictions.argmax(dim=1)
    # y теперь уже является классом (0 или 1), а не one-hot вектором

    print(f"Predicted class {predictions}")
    
    correct += (pred_classes == y.argmax(dim=1)).sum().item()
    total += y.size(0)

accuracy = correct / total
print(f"\nТочность на тестовой выборке: {accuracy:.4f} ({correct}/{total})")

torch.save(model, f"model.pth")
torch.save(model.state_dict(), f"model_weights.pth")






