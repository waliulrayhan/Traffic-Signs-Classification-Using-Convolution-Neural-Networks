# Traffic Signs Classification Using Convolutional Neural Networks

## Overview
This project involves the development of a Traffic Sign Classification system using a Convolutional Neural Network (CNN). The aim is to accurately classify traffic signs, which is essential for applications like autonomous driving and driver assistance systems.

## Project Structure
The project consists of two main scripts and a trained model:
- **TrafficSigns_main.py**: Script for training and evaluating the CNN model.
- **TrafficSign_Test.py**: Script for real-time testing of the model using a connected camera.
- **model_trained.p**: Serialized trained model used for real-time classification.

## Features
- Data preprocessing, including grayscale conversion, histogram equalization, and normalization.
- Image augmentation for improved model generalization.
- CNN architecture with convolutional, pooling, and dropout layers to mitigate overfitting.
- Training and validation plots to visualize loss and accuracy.
- Real-time classification capability using a webcam or video feed.

## Installation and Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/Traffic-Signs-Classification.git
   cd Traffic-Signs-Classification
2. **Install dependencies:** Ensure that Python and pip are installed. Run:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include numpy, opencv-python, keras, tensorflow, matplotlib, scikit-learn, and pandas.

3. **Set up the dataset:** Place the dataset of traffic signs in the myData directory and ensure a corresponding labels.csv file exists for mapping class IDs to labels.

## Usage
### Model Evaluation
After training the model, you can view the training and validation performance metrics in the form of plots for both loss and accuracy. These plots help visualize the model's progress over epochs and identify potential overfitting or underfitting issues.

To evaluate the trained model on the test dataset, the script automatically calculates the test score and accuracy at the end of the training process:
- **Test Score**: Represents the categorical cross-entropy loss on the test data.
- **Test Accuracy**: The proportion of correctly classified test images.

### Running the Model
Ensure that the `model_trained.p` file is present in the project directory. This file contains the serialized trained model, which is loaded during the real-time classification.

1. **Training Mode**:
   Run the training script with:
   ```bash
   python TrafficSigns_main.py
   ```
   This will train the model, display evaluation results, and save the trained model.
2. **Testing Mode**: For real-time testing with a webcam, execute:
   ```bash
   python TrafficSign_Test.py
   ```
   This script will:
   - Open a video window displaying the feed.
   - Predict and overlay the classified traffic sign and its probability.
   - Press q to exit the test window.

## Modifying the Model
To adapt the model to your specific needs, you can customize the following components in `TrafficSigns_main.py`:

1. **CNN Architecture**:
   Modify the `myModel()` function to experiment with different configurations such as:
   - **Number of convolutional layers**: Add or remove layers to explore deeper or shallower architectures.
   - **Filter sizes and counts**: Change the size and number of filters in each convolutional layer to enhance feature extraction.
   - **Pooling and Dropout**: Adjust the pool size and dropout rates to balance regularization and prevent overfitting.

2. **Training Hyperparameters**:
   - **Learning rate**: Modify the `learning_rate` parameter in the `Adam` optimizer to fine-tune the learning process.
   - **Batch size and epochs**: Adjust `batch_size_val` and `epochs_val` for different dataset sizes or to control training time.
   - **Augmentation techniques**: Edit the `ImageDataGenerator` parameters to try different augmentation strategies like rotation, zoom, or flipping for better generalization.

3. **Data Handling**:
   Ensure that the `path` and `labelFile` variables point to the correct directories and label CSV file. If using a different dataset, update the `myData` directory structure accordingly.

## Example Configurations
For those looking to experiment with new configurations, consider:
- **Adding Batch Normalization**: Enhance model stability and training speed by including batch normalization layers.
- **Advanced Architectures**: Integrate popular architectures like MobileNet, VGGNet, or ResNet for more robust results.
- **Hyperparameter Tuning**: Use tools like `Keras Tuner` or `Hyperopt` for automated tuning of hyperparameters.

## Results Interpretation
### Training and Validation Metrics
After training, review the generated plots for:
- **Training Loss**: Shows how well the model is learning during training.
- **Validation Loss**: Helps monitor overfitting if it starts diverging from the training loss.
- **Training and Validation Accuracy**: Use this to gauge how accurately the model predicts both seen (training) and unseen (validation) data.

### Real-Time Test Analysis
The `TrafficSign_Test.py` script displays the following in real-time:
- **Class Name**: The predicted class of the traffic sign.
- **Confidence Level**: The prediction probability, shown as a percentage, indicating the model's confidence.

Ensure that the webcam feed displays signs correctly for better prediction accuracy. The frame resolution, normalization, and preprocessing steps are crucial for optimal results.

## Future Improvements
To further enhance this project:
- **Transfer Learning**: Apply pretrained models on traffic sign data for potentially better accuracy with less training time.
- **Deployment**: Implement a web interface using Flask, FastAPI, or deploy as a mobile app using TensorFlow Lite for edge devices.
- **Dataset Expansion**: Integrate more diverse traffic sign images, including varied weather and lighting conditions, for improved real-world application.

## Contributing
Contributions are highly encouraged! To contribute:
1. Fork this repository.
2. Create a new branch (`feature/your-feature-name`).
3. Commit your changes and push them to your fork.
4. Open a pull request with a detailed explanation of your modifications.

## License
This project is licensed under the MIT License. Please see the `LICENSE` file for more information.

## Contact
For any inquiries, feedback, or support:
- Email: [mdwaliulislamrayhan@gmail.com]
- Open an issue on the GitHub repository for discussions or troubleshooting.
