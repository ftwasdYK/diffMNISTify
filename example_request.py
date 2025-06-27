import numpy as np
import requests
import os
import json
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

PORT = 8000
PREDICT_ENDPOINT = f"http://localhost:8000/predict"
TIMEOUT_SECONDS = 600

# Load data in order to test the API #
transform = transforms.Compose([
        transforms.ToTensor(),
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Send requests to the API #
for img, label in test_loader:
    
    mnist_image = img.squeeze().numpy()  # Convert to numpy array and remove batch dimension
    REQUEST_BODY = {
        "img_raw": mnist_image.tolist()
    }

    response = requests.post(PREDICT_ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS)
    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
    )
    
    if response.ok:
        with open(output_filename, "w") as outfile:
            json.dump(response.json(), outfile)
    
    print(label.item(), response.json())