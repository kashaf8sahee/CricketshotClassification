import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.dataset import CustomImageDataset, prepare_data
from src.transforms import get_transforms
from src.model import SimpleCNN
from src.train import train_one_epoch
from src.evaluate import evaluate_model
from src.visualize import show_predictions

if __name__ == "__main__":
    data_dir = r"E:\CricketshotClassification\data"
    class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']

    image_paths, labels, _ = prepare_data(data_dir, class_names)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    transform = get_transforms()
    train_dataset = CustomImageDataset(train_paths, train_labels, transform)
    test_dataset = CustomImageDataset(test_paths, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    for epoch in range(10):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

    evaluate_model(model, test_loader, device, class_names)
    show_predictions(model, test_dataset, class_names, device)