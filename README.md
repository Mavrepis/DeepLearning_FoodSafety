## A Deep Learning Approach using Natural Language Processing and Time-series Forecasting towards enhanced Food Safety

### Description
This project is the implementation of the manuscript "A Deep Learning Approach 
using Natural Language Processing and Time-series Forecasting towards enhanced Food Safety".
There are two components composing this project, namely:
- Named Entity Recognition (NER)
- Time-series recall prediction using Reinforcement-Learning (RL)

The first component is responsible for the extraction and annotation of products 
from food recall announcements using a custom-trained Named Entity Recognition model built
with SpaCy, while the second component utilizes data produced from the previous process in a
matrix representation to predict the future recalls for a product category.

### Dataset
The dataset used for this project was provided by Agroknow but since it is a private 
one we could not publish this
and created a demonstration model with artificial dummy data.

The original dataset for the NER has the following structure:

 Index| Announcement | Label |
--- | --- | --- | 
0 | atropine 47 ppb scopolamine 30 ppb organic buckwheat flour france | organic buckwheat flour |
1 |dead insects live insects glucosamine sulphate china | glucosamine sulphate

While for the Time-series prediction:

 Index| # Recalls | Date | product |
--- | --- | --- | --- |
0 | 1 | 1985-07-12 |alcoholic beverages
1 |0 | 1985-07-13 | alcoholic beverages

Both of these datasets were heavily preprocessed as described in the manuscript to enhance 
their quality and usability.

### Installation
In order to install this project you need to clone the repository and install
the following:
```
pip install spacy==2.3.7
pip install spacy-lookups-data==1.0.3
pip install scikit-learn==0.23.2
```
### Usage
To use this project you should run main.py with the appropriate flags.

Flag| Description | Default Value |
--- | --- | --- | 
model_path | Directory in which the model will be saved | ./model |
data_path | Directory in which the data are located | ./data |
epochs | Number of epochs used to train the model | 100 |
demonstration | Signifies whether real or dummy data should be used | True


### Support 
For any questions regarding this project please contact
[Georgios Makridis](mailto:geomac.mac@gmail.com) and/or 
[Philip Mavrepis](mailto:philipmavrepis@gmail.com)

### Authors and acknowledgment
The research leading to the results presented in this paper has received funding from the 
European Union’s Project CYBELE under grant agreement no 825355.
#### Funding
The authors declare that, the research leading to the results presented in this paper has 
received funding from the European Union’s Project CYBELE under grant agreement no 825355.