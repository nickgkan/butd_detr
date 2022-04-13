from transformers import RobertaModel, RobertaTokenizerFast
import numpy as np
import ipdb
st = ipdb.set_trace

object_file = 'objects_vocab.txt'
output_path = '/projects/katefgroup/language_grounding/class_embeddings.npy'

tokenizer = RobertaTokenizerFast.from_pretrained(
            'roberta-base')
text_encoder = RobertaModel.from_pretrained(
    'roberta-base')

with open(object_file, 'r') as f:
    object_list = [line.strip() for line in f.readlines()]

tokenized = tokenizer.batch_encode_plus(object_list, padding="longest", return_tensors="pt")
encoded_text = text_encoder(**tokenized)
object_embeddings = encoded_text.last_hidden_state.mean(1)
np.save(output_path, object_embeddings.detach().numpy())
