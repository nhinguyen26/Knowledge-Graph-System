import spacy
import dill
import torch
from rapidfuzz import process, fuzz
from transformers import RobertaTokenizer
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
 
################### DEFINING FUNCTIONS ###############################

def get_head_entities(question, k_closest_per_entity):
    nlp_trf = spacy.load("en_core_web_trf")
    doc = nlp_trf(question)
    if not doc:
        nlp_sm = spacy.load("en_core_web_sm")
        doc = nlp_sm(question)
    head_entities = []
    for entity in doc.ents:
        head_entities.extend(get_closest_strings(str(entity), k_closest_per_entity))
    return head_entities

def get_head_entities_roberta(question, k_closest_per_entity):
    nlp_trf = spacy.load("en_core_web_trf")
    doc = nlp_trf(question)
    if not doc:
        nlp_sm = spacy.load("en_core_web_sm")
        doc = nlp_sm(question)
    head_entities = []
    for entity in doc.ents:
        head_entities.extend(roberta_closest_strings(str(entity), k_closest_per_entity, choices_list, full_choices_embed))
    return head_entities 

def ner(question):
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(question)
    return doc.ents

def roberta_closest_strings(target_string, top_k, choices_list, choices_embeddings):
    string_embedding = robertamodel.encode(target_string , convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(string_embedding.to("cpu"), choices_embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    closest_k_strings = dict()
    for idx in top_results[0:top_k]:
        closest_k_strings[choices_list[idx]] = cos_scores[idx].item()
    return closest_k_strings

def get_closest_strings(string, n):
    strings = [choices[i[0]] for i in process.extract(string, choices.keys(), scorer=fuzz.partial_ratio, limit=n)]
    return strings

class DatasetMetaQA(Dataset):
    def __init__(self, data, entities, entity2idx):
        self.data = data
        self.entities = entities
        self.entity2idx = entity2idx
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = 'roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)

    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = []
        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            if tail_name in self.entity2idx:
                tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)
        return question_tokenized, attention_mask, head_id, tail_onehot 

    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

# return a dictionary with the highest scoring entities as keys and scores as values
def get_answer(question, head_entities, k_answers, device, dataloader, model, entity2idx, idx2entity):
    scores_dict = dict()
    head_en = head_entities
    for entity in head_en:
        question_tokenized, attention_mask = dataloader.tokenize_question(question)
        head = torch.tensor(entity2idx[entity], dtype = torch.long).to(device) 
        question_tokenized = question_tokenized.to(device)
        attention_mask = attention_mask.to(device)
        scores = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask)[0]
        mask = torch.zeros(len(entity2idx)).to(device)
        mask[head] = 1
        new_scores = scores - (mask*99999)

        # add scores for each head entity to a dictionary
        for i in range(len(new_scores)):
            scores_dict[new_scores[i].item()] = i
    answers_dict = dict()
    # take the highest scoring answer entities and scores from the dictionary
    for _ in range(min(len(scores_dict.keys()), k_answers)):
        highest_score = max(scores_dict.keys())
        max_answer_idx = scores_dict[highest_score]
        answers_dict[idx2entity[max_answer_idx]] = highest_score
        del scores_dict[highest_score]
    return answers_dict

######################################################################
############## LOADING IN PRE-SAVED VARIABLES/MODELS #################

with open('choices.pkl', 'rb') as f:
    choices = dill.load(f)

with open('entity2idx.pkl', 'rb') as f:
    entity2idx = dill.load(f)

with open('device.pkl', 'rb') as f:
    device = dill.load(f)

with open('model2.pkl', 'rb') as f:
    model2 = dill.load(f)

with open('idx2entity.pkl', 'rb') as f:
    idx2entity = dill.load(f)

e = torch.load("e.pt")

######################################################################
################### INITIALIZING INTERACTIVE QA ######################

num_answers = int(input("Enter the number of answers to ouput for each question: "))

roberta = input("Type 'y' to use the Roberta model, else press any key(s) to use Fuzzy: ") == 'y'

question = input("Enter your question, or type 'exit' to end the session: ")

if roberta:
    full_choices_embed = torch.load('full_choices_embeddings.pt', map_location=torch.device('cpu'))
    robertamodel = SentenceTransformer('stsb-roberta-large')
    with open('choices_list.pkl', 'rb') as f:
        choices_list = dill.load(f)

# type 'exit' in order to end the interactive question-answering
while question != "exit":
    head_en = input("Enter the head entity manually, or press 'n' to generate it using the question: ")
    if head_en == 'n':
        if roberta:
            matched_entity = get_head_entities_roberta(question, 2) # finding the most similar, comparing the extracted entity to entities in kg
        else:
            matched_entity = get_head_entities(question, 2) # finding the most similar, comparing the extracted entity to entities in kg
    else:
        matched_entity = [head_en]
    print(matched_entity)
    dataloader = DatasetMetaQA(question, e, entity2idx)
    ans = get_answer(question, matched_entity, num_answers, device, dataloader, model2, entity2idx, idx2entity)
    print(ans)
    question = input("Enter your question: ")

######################################################################