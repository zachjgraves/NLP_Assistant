# NLP_Assistant

This Vending Machine Assistant was built with a custom sequential model using Keras's Sequenstial API.

The model is trained using the **intents.json** file. Customize this file to create your own NLP!

### Dependencies

Flask, json, nltk, numpy, tensorflow.keras

### Model Descrition:

Sequential Model: 
This model is created using Keras's Sequential API, which allows you to stack layers sequentially.

Dense Layers:
The first layer is a Dense layer with 128 units. It uses the ReLU (Rectified Linear Activation) activation function, which introduces non-linearity to the model. input_shape is the shape of the input data expected by the model.
The second layer is a Dropout layer with a dropout rate of 0.5. Dropout is a regularization technique used to prevent overfitting by randomly setting a fraction of input units to zero during training.
The third layer is another Dense layer with 64 units and ReLU activation.
The fourth layer is another Dropout layer with a dropout rate of 0.3.
The final layer is a Dense layer with output_shape units and softmax activation. The softmax activation is commonly used for multi-class classification problems as it normalizes the output into a probability distribution over the classes.
Optimizer: The model is compiled using the Adam optimizer (tf.keras.optimizers.Adam). Adam is an adaptive learning rate optimization algorithm that is well-suited for training deep neural networks.

Loss Function: 
The categorical cross-entropy loss function ('categorical_crossentropy') is used, which is commonly used for multi-class classification problems.

Metrics: 
The model will monitor accuracy during training.

Summary: 
print(model.summary()) prints out a summary of the model architecture, including the number of parameters in each layer and the total number of parameters in the model.

Training: 
model.fit() is used to train the model. It takes the training data (train_X and train_y), specifies the number of epochs (200 in this case), and other parameters such as batch size and verbosity.

Overall, this model is designed for multi-class classification tasks, likely with input data of shape input_shape and output data of shape output_shape. The model architecture includes two hidden layers with dropout regularization to prevent overfitting, and it's optimized using the Adam optimizer with a high initial learning rate (0.01).
