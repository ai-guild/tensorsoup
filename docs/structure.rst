

The object of consideration is the task. A task is defined by a dataset and its associated problem statement.
For instance, babi
- is a collection of stories, question and answer forming a dataset and
- has 20 different types of reasoning problems to solve over the same
    
In terms of implementation, a task consists of three important modules of code.
a model module,
a dataset module and
an application module

Model:
A function or a class that builds the computational graph with parameters to be trained

Dataset:
Set of utility functions to preprocess the datset and helper functions to feed the data into the model in the structure it would expect.

Application:
Module to control the training, and hyperparameters. This is where invoke the actual training process. This will encapsulate the process of building the model and feed data.

