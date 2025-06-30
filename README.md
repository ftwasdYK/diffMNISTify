# Description of the Project

This project is the final assignment for the AI Hands-On course of the MSc in Data Science program at [NTUA](https://mathtechfin.math.ntua.gr/?page_id=3661&lang=en).

The objective of the project is to utilize a benchmark dataset, experiment with machine learning algorithms, and develop an API capable of receiving sample input and returning model predictions.

For this purpose, the MNIST dataset was selected due to its popularity and low computational requirements for training models.

The following classical machine learning and deep learning models were used for training on the MNIST dataset:

* RidgeClassifier

* Random Forest

* Support Vector Machine (SVM)

* A modified ResNet-50

* A modified ViT (Vision Transformer)

Additionally, a Denoising Diffusion Probabilistic Model [DDPM](https://arxiv.org/abs/2006.11239) was trained to generate synthetic data. These synthetic samples were used to compare the performance of deep learning models trained on original data versus synthetic data.

Finally, an API endpoint was implemented to allow the generation of a specified number of synthetic samples on demand.

# Get started

To build the project (this may take a while), run:

```bash
bash build_project.sh
```

To send a request to the model and get a prediction for a digit, run:

```python
python example_request.py
```

To request a specific number of synthetic digit images from the server, use:
```python
python example_gen_request.py --num 7
```

The outputs will be saved in the `./diffMNISTify` directory:

- `sample_response.json` – contains the prediction response from the model.
- `img_gen_response.png` – contains the generated synthetic digit image.

# Training 

# Results
