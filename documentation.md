# **SAAS x [24]7AI: KGQA Project**


# Overview
The goal of this project was to build a knowledge graph question answering system using SOTA techniques and research. We decided to base our KGQA system on the paper [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://malllabiisc.github.io/publications/papers/final_embedkgqa.pdf) because of the benefits for multihop question answering, incompleteness, and the existing Github/implementation as a starting point. 


## High-Level Summary of the Paper
Knowledge graphs are built from a collection of triples in the form `(head entity, relation, tail entity)`. The goal of a KGQA is to use an underlying knowledge graph to answer an input question; at a high level, the input question will be split into a head entity and relation, and the knowledge graph is used to find the corresponding tail entity. The core idea of this paper is representing knowledge graphs (more specifically, the entities and relations that make up the knowledge graph) with embeddings. These embeddings are created such that, given some function `f(head_embedding, relation_embedding, tail_embedding)`, the function will output a positive value if this triple is indeed a true relationship, and a negative value otherwise. This embedding method and the corresponding function are defined in the paper [Complex Embeddings for Simple Link Prediction](https://arxiv.org/abs/1606.06357).  

After these embeddings are created, a separate question-embedding model will create an embedding for any given question, and the answer to that question will be the `tail` that results in the highest `f(head_embedding, question_embedding, tail_embedding)`, where `head_embedding` is created from the head entity extracted from the question. 


# Modules
We primarily split the code into three distinct modules, based on the paper with some small changes/additions: (1) the KG embedding module, (2) the question-embedding module, and (3) the answer-selection module. The KG embedding module creates and stores embeddings for all entities. The question-embedding module trains a model that can input a question and output an embedding representation for that question. The answer-selection module brings everything together, using the entity embeddings and question-embedding module to generate answers for a natural language question.


## Module 1: Knowledge Graph Embedding Module (aka Entity Embedding Module)
This module creates and stores embeddings for all entities. The majority of the code was pulled directly from the [LibKGE library](https://github.com/uma-pi1/kge), a PyTorch-based library for training knowledge graph embeddings. Most of the changes made to the source code were regarding file paths to access data and read/write checkpoints.

The EmbedKGQA paper determined that the best type of KG-embedding was ComplEx embeddings. Given all head entities, relations, and tail entities, ComplEx generates embeddings $e_h$, $e_r$, $e_t$ $\in C^d$ with the scoring function $\phi(h, r, t)$ = $Re(\langle e_h, e_r, \bar e_t\rangle)$. Using complex embedding vectors ensures that relationships are not commutative (ex. "A dog is a mammal" and "A mammal is a dog" are clearly not equivalent) and the use of inner products helps to capture information about multi-hop relations.


### Instructions
- Inside the `data` directory, create a folder with the relevant dataset files. `data/fb_natural_language_full` is an example of such a folder, with ~2 million sets of training data triples from the Freebase database. However, this file currently contains many files that are created during the module 1 or module 2 training processes.
Only the required files are required for training embeddings:
  - `train.txt` - training data of triples in the form "head_entity relation tail_entity" on each line
  - `valid.txt` - validation data of triples
  - `test.txt` - test/evaluation data of triples

- Run `preprocess_edited.py` on this data folder, which creates a `dataset.yaml` file and del files from the txt files: `!python3 preprocess_edited.py <file path to data folder>`. The del files contain the triples represented with integers instead of words. These created files will be used in the entity embedding training below.

- Create a folder inside `kge_module/embeddings/` for the embeddings that you are creating, an example for the `fb_natural_language_full` dataset would be `kge_module/embeddings/ComplEx_fbwq_full_natural_language`. This is where the PyTorch checkpoint files will be placed during training.
- Create a `config.yaml` inside the above folder, and set `dataset.name` to the corresponding dataset folder created earlier (ex. `fb_natural_language_full`)

- Make sure that the version of Python in Conda is >=3.7 and <3.9. We used Python 3.8 which is not the default in AWS EC2, but this can be changed by running the following
  - `conda activated /opt/conda`
  - `conda uninstall python`
  - `conda install python=3.8`

- Inside the `kge_module` directory, run the following, which will start training on GPU if available, else CPU. 
  - `pip install -e .`
  - `kge resume embeddings/model_folder_name`
  - If using multiple GPUs, `--search.device_pool` and `--search.num_workers` parameters can be set
    - ex. `kge resume embeddings/ComplEx_fbwq_full_natural_language —search.device_pool cuda:0,cuda:1,cuda:2,cuda:3 —search.num_workers 4`

- After training, the checkpoint files for the embeddings will be in `kge_module/embeddings/model_folder_name`


## Module 2: Question Embedding Module
The Question Embedding Module creates the PyTorch model used to generate question embeddings, which are used in the answer selection module to answer natural language questions. The code is adapted from the EmbedKGQA code repository, with significant changes with regards to structure and organization. Below are the files that are relevant to this module:
- `train.py` - the file that trains the model, contains many helper functions from the original repo that weren't used, but are left here in case [24]7 AI has a use for them later. Note that the hyperparameters and file paths are hard coded in to this file (meaning `python3 train.py` is all that's needed to train the model)
- `model.py` - contains the `RelationExtractor ` class that subclasses `nn.Module`, contains the structure of the question embedding model
- `dataloader.py` - contains the `DatasetFB` class that's used to load in and process data for training

The question embedding model has the following structure, centered around fine-tuning a pretrained RoBERTa model:
- input a tokenized question and attention mask into RoBERTa and obtain an embedding from the last hidden states
- pass the embedding into four fully connected layers with ReLU activation
- project that output onto $C^d$, in our case $d=50$

"The model is learned by minimizing the binary cross entropy loss between the sigmoid of the scores [output of the $\phi(head, relation, tail)$ function] and
the target labels, where the target label is 1 for the correct answers and 0 otherwise." (EmbedKGQA paper)


### Instructions:
- Set hyperparameters in `train.py` as desired
- Set the `data_path`, `valid_data_path`, and `test_data_path` variables (at the top of `train.py`) to the same dataset used in module 1
- Set the `model_folder` variable (at the top of `train.py`) to the name of the folder that stored the embeddings in module 1 (ex. `ComplEx_fbwq_full_natural_language`)
- Create a folder in `question_embedding_module/models` with the same name as the folder that stored the embeddings in module 1 (ex. `ComplEx_fbwq_full_natural_language`)
- Run the `train.py` file in the `question_embedding_module` directory, ensuring that the system is also on Python 3.8 (I believe Python 3.6 and older don't work, Python 3.7 and 3.9 might, but if you run this on the same system used to run module 1, then it should already be Python 3.8)
  - `python3 train.py`
- The .pt file for the question embedding model will be stored in the folder specified in the above steps: `question_embedding_module/model/model_folder`


## Module 3: Answer Selection Module
Given the entity embeddings in the form of a checkpoint file, the question embeddings, and the `entity_ids.del` file for the data, the answer selection module takes in natural language questions and outputs answers.
 
- The `getEntityEmbeddings` function returns the entity embeddings as a dictionary with the keys as the entity name strings and the `prepare_embeddings` function creates dictionaries that allow us to access the entity name once we get the answer. 
 
- The `DatasetMetaQA` class contains methods that process and tokenize questions.
 
- The `RelationExtractor` class contains the structure of the question embedding module. We initialize the model by calling `RelationExtractor`, then fit the model using the question embeddings that were created in the previous module.
 
- The `get_answer` function takes in the following arguments: the question as a string, the number of most similar entities in the knowledge the extracted entity is mapped to, the number of answers the module should return, the device, dataloader, model, and dictionaries that convert from entity to index and index to entity. This function works by looping through each possible head entity, adding scores for each possible answer for each head entity, and then outputting the overall k highest scoring answer entities and scores. 
 
The scores are created by taking the function $\phi(e_h, e_q, e_{a’})$, where $e_h$ is the head entity embedding, $e_q$ is the question embedding, and $e_{a’}$ is a possible answer entity embedding. The model scores the head and question against all possible answers, and the answer entity is the entity that produces the score that maximizes the function.
 
- The `get_head_entities` function returns a list of head entities given a question. It takes in the natural language question(string) and the k_closest_per_entity (int) which specifies how many k most similar strings we want to return. We used a pre-trained NLP English pipeline from SpaCy that parses the input question and extracts the entities. We iterate through each of these entities and find the closest k strings for each entity using the `get_closest_string` function. Similarly, we have the 
`get_head_entities_roberta` function, which does the same thing, except using the `roberta_closest_strings` function to find the closest k strings.
 
- The `get_closest_string` returns a list of k closest strings given a target string.  We tried two different pretrained models to find the closest strings, Fuzzy Matching and Semantic Similarity Matching using Transformers. The Fuzzy Matching defines similarity as matching the pattern of the strings while the Semantic Matching Transformer defines similarity as matching the semantic meaning of the strings. The SentenceTransformer uses ROBERTA-large as the base model and mean-pooling, as it is the best model for the task of semantic similarity. Fuzzy Matching and the SentenceTransformer both perform similarly in terms of returning the accurate closest words. The comparison result between the two can be found in the notebook `fuzzy_vs_roberta_get_closest_string.ipynb`. The `roberta_closest_strings` function virtually follows the same protocol as , except using `get_closest_string` except using the RoBERTa matching model.
 

### Instructions
In order to run the question-answering module, refer to the files `answer_selection.py` and `question_answer.py`. First, the top of the `answer_selection.py` file lists the python commands to run in order to install any dependencies to be used within the module. Once those are loaded, run the file as normal, and it will preprocess the information needed to be used within the `question_answer.py` file. Then, run the `question_answer.py` file, and it will load in the information that was preprocessed previously, as well as define any necessary functions. On the command line, an instruction will be printed, prompting you to answer the number of answers you'd like to output. Then, you'll be asked whether you’d like to use the RoBERTA or Fuzzy matching model. Type 'y' to use RoBERTA, otherwise any key for Fuzzy, and press enter. The next input will ask you to enter a natural language question that you’d like to be answered, and once you input the question and press enter, it will ask you whether you'd like to provide the head entity, or have the model extract it. Then, it will run a series of functions to parse the input, extract entities, and score potential answers, finally printing out the answer and score associated with it. You can continue asking questions and receiving answers, and once you’re done using the model, type ‘exit’ when the CLI asks you to enter a question. This file can be re-run without having to previously run `answer_selection.py` again; that file only needs to be rerun when there’s new data to be added.


## Notes on Preprocessing Data
Originally, the entity ids were in the form of mids, which are unique ids for every object in Freebase (for example, m.03xyz) that maps to a natural language version of the id. We proceeded to replace the mids with the natural language version instead, as we wanted our answer selection module to return the answer in the form of natural language where it is more easily interpretable. We found the Freebase mappings `mid2name.tsv` through a Github repository and proceeded to match the mids in our `train.txt`, `valid.txt`, and `train.txt` files to the natural language words through the mappings. The code for this can be found in `converting_train_natural_language.ipynb`.
 
For the rows in our dataset where we were not able to find the mapping to the natural language version, we discarded those entries. The size of our train dataset was reduced from around 6 million entries to ~2.5 million entries. We proceeded to further ensure that each row consists of the tuples (head entity, relation, tail entity) and discarded rows that didn’t have all 3 parts. This removed around 4,000 rows. 


## Notes on Training Time and Costs
Models were primarily trained on AWS EC2 instances, occasionally using AWS Sagemaker for easier debugging/experimentation.

We trained two sets of models, one on half of the data (triples) and the corresponding Web Questions data, and one on the full set of data. This was because the latter would take much more time and/or money to train, so the halved dataset was a better first step.


### Training Half-Data Models:
All times below purely count the time spent training the final models, not debugging or experimentation.

The halved dataset had ~1.3 million triples (to train entity embeddings) and 886 question-answer pairs for training (to train question embedding model, doesn't include validation set).
- Both models were trained on a AWS EC2 p3.2xlarge instance at $3.06 per hour
  - uses a single NVIDIA V100 GPU
- Entity embedding training took ~11 minutes/epoch, at 20 total epochs
  - ~3.7 hours and $11.22
- Question embedding training took ~2.5 minutes/epoch, at 50 total epochs
  - ~2 hours and $6.06
  - Note: the dimension of question embeddings was reduced to 50 from 200 (as used in the paper) because we believed it would be marginal benefit for very significant training time/cost increase
  - Note: we reduced num_epochs to 50 from 200 (as used in paper), since the dataset was small and we reduced the dimension size

The full dataset had ~2.5 million triples (to train entity embeddings) and 2300 question-answer pairs for training (to train question embedding model, doesn't include validation set).
- Both models were trained on a AWS EC2 p3.8xlarge instance at $12.24 per hour
  - uses four single NVIDIA V100 GPU
  - much higher memory than p3.2, we were originally concerned about memory because it was an issue in Google Colab (Pro) but uncertain if p3.2xlarge -> p3.8xlarge was necessary
- Entity embedding training on a single GPU (on p3.2xlarge) took ~25 minutes/epoch, at 20 total epochs
  - paused about 1/3 way through, switched to p3.8xlarge instance
- Entity embedding training on all four GPUs (on p3.8xlarge) took ~15 minutes/epoch, at 20 total epochs
  - ~5 hours and $61.20
  - achieved this by increasing batch size by a factor of 4 and having four workers at a time, one on each device
  - probably could have utilized the GPUs even more, but didn't think the time required to experiment was worth it
- Question embedding training took ~2.5 minutes/epoch, at 200 total epochs
  - due to our computational power, we stopped at 79 epochs
  - ~3.3 hours and $40.39 (with 79 epochs)
  - note: the dimension of question embeddings was reduced to 50 from 200 (as used in the paper) because we believed it would be marginal benefit for very significant training time/cost increase
  - achieved this by increasing batch size by a factor of 4 and parallelizing computations during training only for the RoBERTa "layer"
  - probably could have utilized the GPUs even more, but we didn't think the time required to experiment was efficient
    - perhaps parallelizing computations for each of the fully connected layers would have helped, but we were concerned it might increase training time due to overhead I/O costs


## Notes on Updating the Model
The client has mentioned before that they would like some way to update the model.

First, some limitations regarding updating models with new data: the main blocker is that the created entity embedding for any entity depends on all the other entities and triples in the dataset. The entity embedding module (module 1) outputs a lookup table, which means we cannot simply generate a new embedding with some input data (specifically, an entity), as we are able to with the question embedding model. This also means that if we want to update the entity embeddings with some new data, this would most likely change ALL of the other entity embeddings as well.

This is a problem because the question embedding model is created using the question-answer dataset and also the existing entity embeddings. Thus, if the entity embeddings change at all, then the question embedding model must be completely retrained.

While we did not implement some sort of way to update models with new training data, here are some potential options:
1. Add new data into the KGE/entity embedding training data, resume training on the entity embeddings until the model converges again (or hit num_epochs), then take this new output and then resume training the question embedding model until it converges again.
    - This method is good if you want the new data to be equally important as the past data
    - Even though you won't be training the model from scratch, it could take many epochs before training converges again, and each epoch would still have to make a full pass through all the training data. As you add more and more data, this gets especially costly.
    - The time it takes to resume training the question embedding model depends on how much the embeddings have changed
2. Treat the current model as a "pretrained model" and use the new training data to "finetune" this model. Then resume training the question embedding model until it converges again
    - This method may cause the new training data to have more weight in the model -- this is a pro/con depending on context.
    - Each epoch would only make a full pass through the new data, not all of the data.
    - The time it takes to resume training the question embedding model depends on how much the embeddings have changed.

The method used to update the model seems to be very dependent on the use-case of the KGQA system and the context of the data, which is uncertain to us at this point. Nonetheless, because updating seems to be a fairly costly operation, we would suggest this happens only after receiving large batches of new data.


## Limitations and Next Steps
Some potential ideas to pursue in the future:
- create/find a dataset where the entities in the dataset are closer to what would be extracted from a question (ex. using fuzzy or bert)
- create a knowledge graph with better defined entities
- more question/answer training data
- more context-specific data
- if sufficient resources: hyperparameter tuning on complex embedding


# Paper Citation
```
@inproceedings{saxena2020improving,
  title={Improving multi-hop question answering over knowledge graphs using knowledge base embeddings},
  author={Saxena, Apoorv and Tripathi, Aditay and Talukdar, Partha},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={4498--4507},
  year={2020}
}
```