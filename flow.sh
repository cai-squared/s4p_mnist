echo "Running the S4P MNIST pipeline..."
echo "Step 1: Making the dataset"
make data
echo "Step 2: Training the model"
make train
echo "Step 3: Predicting with the model"
make predict
echo "Pipeline execution completed successfully!"
echo "You can find the trained model in the 'models' directory and the predictions in the 'predictions.csv' in /app directory."
