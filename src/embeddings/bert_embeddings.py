import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List
from ..utils.cache import get_cache_key, cache

class BertEmbeddings:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(self.device)
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get BERT embeddings for a single text."""
        cache_key = get_cache_key(text, 'bert_embedding')
        
        if cache_key in cache:
            return cache[cache_key]
        
        # Tokenize and get embedding
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        cache[cache_key] = embedding[0]  # Store the flattened embedding
        return embedding[0]
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get BERT embeddings for a list of texts."""
        return np.array([self.get_embedding(text) for text in texts]) 