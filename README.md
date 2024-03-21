# Natural Language Processing Vending Machine Assistant

## Overview
The Vending Machine Assistant is a custom sequential model built using Keras's Sequential API. It facilitates natural language processing (NLP) tasks specifically tailored for interactions with vending machines. Users can customize the model by modifying the intents.json file to create their own NLP interactions.

## Model Description
Sequential Model
- The model is constructed using Keras's Sequential API, allowing for the stacking of layers sequentially.
Dense Layers
- The model consists of Dense layers with varying numbers of units and ReLU activation functions.
- Dropout layers are incorporated to prevent overfitting by randomly setting a fraction of input units to zero during training.
Output Layer
- The final layer is a Dense layer with softmax activation, suitable for multi-class classification problems.

## Dependencies
- Flask
- json
- nltk
- numpy
- tensorflow.keras

## Training
- The model is compiled using the Adam optimizer with a categorical cross-entropy loss function.
- During training, the model monitors accuracy and adjusts weights accordingly.
- Training data (train_X and train_y) is fed into the model via model.fit(), specifying the number of epochs and other parameters.

## Summary
- print(model.summary()) provides a comprehensive overview of the model architecture, including the number of parameters in each layer and the total number of parameters in the model.


## Customization
- Users can customize intents in the intents.json file to tailor NLP interactions according to specific vending machine contexts.
- Responses to common patterns can be modified to enhance user engagement and satisfaction.
- {"tag": "greeting",
     "patterns": ["Hello", "How are you?", "Hi there", "Hi", "Whats up", "help", "assistance"],
     "responses": ["G'day mate! How can I help today?", "Hi! Is there something I can help you with?"]
    },

## Conclusion
The Vending Machine Assistant leverages deep learning techniques to enhance user interactions with vending machines. By providing a customizable NLP framework, users can create tailored experiences that cater to diverse vending machine scenarios, ultimately improving user satisfaction and efficiency.
This Vending Machine Assistant was built with a custom sequential model using Keras's Sequenstial API.
