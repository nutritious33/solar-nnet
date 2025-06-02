# solar-nnet
**Neural Network: **
Feed-forward neural network for binary classification using synthetic data from NREL. Trained using nine input features from solar data, like temperature, gamma, power, etc. and one-hot encoded labels. The model architecture starts with an input layer, then two hidden layers that use ReLU activations, outputting two logits.

**Data Preparation: **
The data is standardized using the StandardScaler from sklearn. Training and evaluation is done with an 80/20 split. One-hot encoded labels are converted to class indices.
Loss and Optimization: The model uses CrossEntropyLoss. Optimization is handled by the Adam optimizer with a learning rate of 0.01, balancing speed and stability during training.

**Evaluation:**
After training, model performance is evaluated on the test set using overall accuracy as well as a confusion matrix to visualize true vs. predicted labels. A seaborn heatmap provides a normalized view of this matrix.

**Training Performance: **
The model trains over 100 epochs with batches of size 16. On the Apple M3 Max (CPU execution), total training time is approximately 13 seconds, achieving fairly high test accuracy (~95%).
