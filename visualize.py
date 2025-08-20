import matplotlib.pyplot as plt
import torch

def show_predictions(model, dataset, class_names, device, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 10))
    for i in range(num_images):
        image, true_label = dataset[i]
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
        image = image.numpy().transpose((1, 2, 0))
        axes[i].imshow(image)
        axes[i].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred.item()]}")
        axes[i].axis("off")
    plt.show()