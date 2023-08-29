# pip install transformers
# pip install path
# pip install rapidfuzz
# pip install torch
# pip install spacy
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_trf
# pip install sentence-transformers

import torch
import dill
from kge.model.kge_model import KgeModel
from kge.util.io import load_checkpoint
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import RobertaTokenizer
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from transformers import RobertaModel
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util

################### DEFINING FUNCTIONS ###############################

# returns entity embeddings 
def getEntityEmbeddings(kge_model, entity_dict):
    e = {}
    embedder = kge_model._entity_embedder
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = embedder._embeddings(torch.LongTensor([ent_id]))[0]
    f.close()
    return e

# creates dictionaries that allow us to access entity name once we get the answer
def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix

# contains question embedding process and score ranking functions
class RelationExtractor(nn.Module):
    def __init__(self, embedding_dim, relation_dim, num_entities, pretrained_embeddings, device,
    entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0, model="ComplEx", ls=0.0, do_batch_norm=True, freeze=True):
        super(RelationExtractor, self).__init__()
        self.device = device
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm
        if not self.do_batch_norm:
            print("Not doing batch norm")
        self.roberta_pretrained_weights = "roberta-base"
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        if self.model == "ComplEx":
            multiplier = 2
            self.getScores = self.ComplEx
        else:
            print("Incorrect model specified:", self.model)
            exit(0)
        self.hidden_dim = 768
        self.relation_dim = relation_dim * multiplier
        self.num_entities = num_entities
        self.loss = self.kge_loss
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.fcnn_dropout = torch.nn.Dropout(0.1)
        self.embedding = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings, dim=0), freeze=self.freeze)
        self.mid1 = 512
        self.mid2 = 512
        self.mid3 = 512
        self.mid4 = 512
        self.lin1 = nn.Linear(self.hidden_dim, self.mid1)
        self.lin2 = nn.Linear(self.mid1, self.mid2)
        self.lin3 = nn.Linear(self.mid2, self.mid3)
        self.lin4 = nn.Linear(self.mid3, self.mid4)
        self.hidden2rel = nn.Linear(self.mid4, self.relation_dim)
        self.bn0 = torch.nn.BatchNorm1d(multiplier)
        self.bn2 = torch.nn.BatchNorm1d(multiplier)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self._klloss = torch.nn.KLDivLoss(reduction="sum")

    def set_bn_eval(self):
        self.bn0.eval()
        self.bn2.eval()

    def kge_loss(self, scores, targets):
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim =1)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)
        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        pred = score
        return pred

    # Inputs a tokenized question and attention mask, returns a question embedding
    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        return question_embedding

    def forward(self, question_tokenized, attention_mask, p_head, p_tail):
        # create question embedding with roBERTa, 768 dim
        question_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)
        # NN that sends question_embedding to a vector with dimension = relation_dim
        rel_embedding = self.applyNonLinear(question_embedding)
        # embedding of the entity in the question
        p_head = self.embedding(p_head)
        pred = self.getScores(p_head, rel_embedding)
        actual = p_tail
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1))
        loss = self.loss(pred, actual)
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        return loss

    def applyNonLinearEval(self, outputs):
        # linear layer with dropout
        outputs = self.lin1(outputs)
        # reLU activation layer
        outputs = F.relu(outputs)
        # linear layer with dropout
        outputs = self.lin2(outputs)
        # reLU activation layer
        outputs = F.relu(outputs)
        # linear layer
        outputs = self.lin3(outputs)
        # reLU activation layer
        outputs = F.relu(outputs)
        # linear layer
        outputs = self.lin4(outputs)
        # reLU activation layer
        outputs = F.relu(outputs)
        # linear layer
        outputs = self.hidden2rel(outputs)
        return outputs

    def get_score_ranked(self, head, question_tokenized, attention_mask):
        question_embedding = self.getQuestionEmbedding(question_tokenized.unsqueeze(0), attention_mask.unsqueeze(0))
        rel_embedding = self.applyNonLinearEval(question_embedding)
        head = self.embedding(head).unsqueeze(0)
        scores = self.getScores(head, rel_embedding)
        return scores

######################################################################
############## CREATING + SAVING VARIABLES AND MODELS ################

# best entity embedding weights trained
checkpoint = load_checkpoint('checkpoint_best.pt')
model = KgeModel.create_from(checkpoint)

e = getEntityEmbeddings(model, "data/fb_natural_language_data_full/entity_ids.del")

entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)

device = torch.device("cpu")

#initialize model with question embedding and score ranking functions that does answer selection
model2 = RelationExtractor(embedding_dim=256, num_entities = len(idx2entity), relation_dim=50, 
                              pretrained_embeddings=embedding_matrix, device=device)

fname = "best_score_model.pt"

model2.load_state_dict(torch.load(fname, map_location=torch.device('cpu')), strict=False)

lines = open("data/fb_natural_language_data_full/entity_ids.del", 'r').readlines()

count = 0
choices = {}
for line in lines:
    count += 1
    entity_name = line.split("\t")[1][:-1]
    if entity_name not in choices:
        choices[str.lower(entity_name)] = entity_name
    else:
        choices[str.lower(entity_name) + "."] = entity_name + "."

choices_list = list(choices.values())

with open('choices_list.pkl', 'wb') as f:
    dill.dump(choices_list, f)

with open('choices.pkl', 'wb') as f:
    dill.dump(choices, f)

with open('entity2idx.pkl', 'wb') as f:
    dill.dump(entity2idx, f)

with open('device.pkl', 'wb') as f:
    dill.dump(device, f)

with open("model2.pkl", "wb") as f:
    dill.dump(model2, f)

with open('idx2entity.pkl', 'wb') as f:
    dill.dump(idx2entity, f)

torch.save(e, "e.pt")

######################################################################
