from application.util.exceptions import ModelWeightsNotFound
from application.ml.what2wear.model import FullBiLSTM as bilstm_vse
from application.util.model_versioning import ModelName, MODELS
from application.util.helpers import load_model_weights
from application.bootstrap.constants import DEVICE, LSTMDirection
import json
import os
import torch

MODEL_VOCAB_PATH = 'application/ml/what2wear/model_vocab.json'
MAX_LSTM_PASSES = 10
WHAT2WEAR_MODEL = MODELS[ModelName.WHAT2WEAR]



def run_lstm(items_in_sequence, candidate_items, model,
                      occasion_item_embeddings, candidate_indeces, direction):

    candidate_embeddings = candidate_items.compute_normalized_embeddings()
    sequence = []
    probabilities = []
    while len(items_in_sequence) < MAX_LSTM_PASSES:
        max_probability, idx = predict_single_direction(
            items_in_sequence, candidate_embeddings, LSTMDirection.BACKWARD, model,
            candidate_indeces)
        max_prob_img = candidate_items.get_item_by_index(idx)
        new_sequence_item = candidate_embeddings[idx].unsqueeze(0)

        if direction == LSTMDirection.FORWARD:
            sequence.append(max_prob_img)
            probabilities.append(max_probability)
            items_in_sequence = torch.cat((items_in_sequence, new_sequence_item))
        else:
            sequence.insert(0, max_prob_img)
            probabilities.insert(0, max_probability)
            items_in_sequence = torch.cat((new_sequence_item, items_in_sequence))

    return sequence, probabilities

def run_lstm_on_sequence(feats, direction, model, hidden=None):
    if not hidden:
        out, hidden = model.lstm(torch.autograd.Variable(feats).unsqueeze(0))
    else:
        out, hidden = model.lstm(
            torch.autograd.Variable(feats).unsqueeze(0), hidden)
    out = out.data
    if direction == LSTMDirection.FORWARD:
        return out[0, :feats.size(0), :out.size(2) // 2][-1].view(1,
                                                                  -1), hidden
    elif direction == LSTMDirection.BACKWARD:
        return out[0, :feats.size(0), out.size(2) // 2:][0].view(1, -1), hidden
    else:
        print("Specifiy a direction for lstm inference")
        return None


def load_bilstm_vse():
    input_dim = hidden_dim = 512
    vocab_size = 2480
    model = bilstm_vse(input_dim,
                       hidden_dim,
                       vocab_size,
                       batch_first=False,
                       dropout=0,
                       freeze=False)
    weights_path = WHAT2WEAR_MODEL.get_weights_file_path()
    model_weights = load_model_weights(weights_path,
                                       WHAT2WEAR_MODEL.name.value)
    model.load_state_dict(model_weights)
    model.eval()
    return model


def predict_single_direction(sequence_embeddings, candidate_embeddings,
                             direction, model, candidate_indeces):
    w_hidden, _ = run_lstm_on_sequence(sequence_embeddings, direction, model)
    hidden_tensor = torch.autograd.Variable(w_hidden)
    candidates_tensor = torch.autograd.Variable(candidate_embeddings)
    probability_scores = torch.nn.functional.softmax(torch.mm(
        hidden_tensor, candidates_tensor.permute(1, 0)),
                                                     dim=1)
    max_probability, idx = find_max_feasible_probability(
        probability_scores, candidate_indeces)
    return max_probability, idx


def find_max_feasible_probability(probability_scores, candidate_indeces):
    probability_scores = probability_scores.squeeze()
    max_probability = 0
    max_probability_idx = None
    for idx in candidate_indeces:
        probability = probability_scores[idx].item()
        if probability > max_probability:
            max_probability = probability
            max_probability_idx = idx
    return max_probability, max_probability_idx


def load_vocab():
    with open(MODEL_VOCAB_PATH) as f:
        data = json.load(f)
    return data
