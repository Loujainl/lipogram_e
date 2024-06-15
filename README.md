## Lipogram Generator
Lipogram Generator Using Fine-Tuned LLM

## Overview
This project generates a French lipogram—a text that excludes a specific letter, in this case, the letter 'e'—using a fine-tuned language model. Inspired by Georges Perec’s novel "La Disparition", this project demonstrates how to use pre-trained models and fine-tuning techniques to create lipograms.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Fine-Tuning Process](#fine-tuning-process)
   - [Data Preparation](#data-preparation)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
4. [Pipeline Overview](#pipeline-overview)
5. [Parameters and Configurations](#parameters-and-configurations)
6. [Results](#results)
7. [Challenges and Future Work](#challenges-and-future-work)
8. [License](#license)


## Introduction
Generating a coherent lipogram is challenging due to the restrictions imposed on the vocabulary and syntax. 
This project utilizes a pre-trained GPT-2 model, fine-tuned on a dataset derived from Georges Perec’s "La Disparition", to achieve this goal.

## Dataset
The dataset for fine-tuning was created using the text from Georges Perec's "La Disparition". This text is inherently a lipogram, making it an ideal training source for our model.

## Fine-Tuning Process
### Data Preparation
1. **Tokenization**: Convert the text into tokens that the model can process. This involves mapping words to their respective indices in the model's vocabulary.
2. **Preprocessing**: Clean the text data to ensure there are no extraneous characters or formatting issues that might slow the model’s learning process.

### Model Training
The fine-tuning process involves the following steps:
1. **Model Selection**: Start with a pre-trained GPT-2 model.
2. **Fine-Tuning**: Adjust the model’s weights using the lipogram text data.
   - **Optimizer**: Adam optimizer with a learning rate of 5e-5.
   - **Loss Function**: Cross-entropy loss to measure the accuracy of the model's predictions.
   - **Epochs**: The model was trained for 5 epochs to balance between learning efficiency and computational resource usage.
   - **Batch Size**: A batch size of 8 was chosen based on available computational resources.
3. **Evaluation**: Regularly evaluate the model on the validation set to monitor performance and adjust parameters as necessary.

### Evaluation
 **Sample Generation**: Generate sample texts to manually check for coherence and adherence to the lipogram constraint.

## Pipeline Overview
1. **Data Loading**: Load and preprocess the training data.
2. **Tokenization**: Convert text into tokens.
3. **Fine-Tuning**: Train the model on the preprocessed dataset.
4. **Text Generation**: Generate lipogram texts using the fine-tuned model.
5. **Post-Processing**: Ensure the generated text adheres to the lipogram constraint by filtering out any unwanted characters.

## Parameters and Configurations
- **Learning Rate**: 5e-5
- **Epochs**: 5
- **Batch Size**: 8
- **Optimizer**: Adam
- **Loss Function**: Cross-entropy
- **Model**: Pre-trained GPT-2

## Results
After 5 epochs of training, the model was able to generate texts that adhered to the lipogram constraint. Here is a sample result:

“C'un pouvoir qu'il avait fait un mot, mais il nous a dans lui aussi mal. Il nul fut plus tard (saurait-on) du moins vingt ans; on aurait aujourd'hirai à jamais la mort (quoiquant cru parfois sa disparition). On alluma illico: il s'agissait pour là son fils d'Amaury Consolu! Qui? Ou plutôt pas l'avocat?..”

- The generated text is a lipogram, excluding the letter 'e'.
- The coherence of the text varies, with some segments making more sense than others.

## Challenges and Future Work
- **Text Length and Token Limit**: The GPT-2 model has a maximum token limit of 1024, requiring the text to be generated in segments.
- **Coherence**: Further fine-tuning and increasing the number of epochs may improve the semantic coherence of the generated texts.
- **Computational Resources**: Training larger models or extending training durations demands significant computational power and time.
- **Generalization**: Improving the model’s ability to generalize to other types of lipograms or tasks.


## License

Copyright (c) 2024 [Loujain Liekah]
