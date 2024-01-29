import os
import random
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import torch
from d3l.indexing.lsh.lsh_index import LSHIndex
from d3l.indexing.similarity_indexes import SimilarityIndex
from d3l.utils.functions import is_numeric
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, T5Model, T5Tokenizer, utils
utils.logging.set_verbosity(40) # Only report errors

from util.data_loader import NextiaJDCSVDataLoader, SpiderCSVDataLoader


class BertTransformer:
    def __init__(
        self,
        model_name: str,
        num_samples: int = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Instantiate a new embedding-based transformer
        Parameters
        ----------
        token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        max_df : float
            Percentage of values the token can appear in before it is ignored.
        stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        model_name : str
            The embedding model name to download from Stanford's website.
            It does not have to include to *.zip* extension.
            By default, the *Common Crawl 42B* model will be used.
        cache_dir : Optional[str]
            An exising directory path where the model will be stored.
            If not given, the current working directory will be used.
        """

        self._model_name = model_name
        self._num_samples = num_samples
        self._cache_dir = (
            cache_dir if cache_dir is not None and os.path.isdir(cache_dir) else None
        )

        self._tokenizer, self._embedding_model, self._device = self.get_embedding_model()

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k != "_embedding_model" and k != "_tokenizer"}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        self._tokenizer, self._embedding_model, self._device = self.get_embedding_model()

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    def get_embedding_model(self):
        # Attempt to load cached tokenizer and model from local
        if self._model_name.startswith("bert"):
            try:
                tokenizer = BertTokenizer.from_pretrained(self._model_name, local_files_only=True)
                model = BertModel.from_pretrained(self._model_name, local_files_only=True)
            except OSError:
                tokenizer = BertTokenizer.from_pretrained(self._model_name)
                model = BertModel.from_pretrained(self._model_name)
        elif self._model_name.startswith("t5"):
            try:
                tokenizer = T5Tokenizer.from_pretrained(self._model_name, local_files_only=True)
                model = T5Model.from_pretrained(self._model_name, local_files_only=True)
            except OSError:
                tokenizer = T5Tokenizer.from_pretrained(self._model_name)
                model = T5Model.from_pretrained(self._model_name)

        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return tokenizer, model, device

    def get_embedding_dimension(self):
        return self._embedding_model.config.hidden_size

    @torch.no_grad()
    def transform(self, input_values: Iterable[str]) -> np.ndarray:
        """
        Extract a column embedding for each column in the table
        Given that the underlying embedding model is a n-gram based one,
        the number of out-of-vocabulary tokens should be relatively small or zero.
        Parameters
        ----------
        table: pd.DataFrame
            The table to extract column embeddings from.

        Returns
        -------
        np.ndarray
            A Numpy vector representing the mean of all token embeddings.
        """
        if self._num_samples > 0 and self._num_samples < len(input_values):
            input_values = random.sample(input_values, self._num_samples)

        batch_size = 256
        num_batches = len(input_values) // batch_size if (len(input_values) % batch_size) == 0 else (len(input_values) // batch_size) + 1
        col_embedding = torch.zeros(self.get_embedding_dimension()).to(self._device)

        for i in range(0, num_batches):
            batch_inputs = input_values[i*batch_size:(i+1)*batch_size]
            encodings = self._tokenizer(batch_inputs, truncation=True, max_length=128, padding=True, return_tensors="pt").to(self._device)
            
            if self._model_name.startswith("bert"):
                batch_embeddings = self._embedding_model(**encodings).last_hidden_state
            elif self._model_name.startswith("t5"):
                batch_embeddings = self._embedding_model(input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"], decoder_input_ids=encodings["input_ids"]).last_hidden_state
            col_embedding += torch.mean(batch_embeddings[:, 0, :], dim=0)
        
        col_embedding /= num_batches
        return col_embedding.detach().cpu().numpy()


class BertEmbeddingIndex(SimilarityIndex):
    def __init__(
        self,
        model_name: str,
        dataloader: Union[NextiaJDCSVDataLoader, SpiderCSVDataLoader],
        num_samples: Optional[int] = None,
        data_root: Optional[str] = None,
        index_hash_size: int = 1024,
        index_similarity_threshold: float = 0.5,
        index_fp_fn_weights: Tuple[float, float] = (0.5, 0.5),
        index_seed: int = 12345,
        index_cache_dir: Optional[str] = None
    ):
        """

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader object used to read the data.
        data_root : Optional[str]
            A schema name if the data is being loaded from a database.
        transformer_token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        transformer_max_df : float
            Percentage of values the token can appear in before it is ignored.
        transformer_stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        transformer_embedding_model_lang : str
            The embedding model language.
        index_hash_size : int
            The expected size of the input hashcodes.
        index_similarity_threshold : float
            Must be in [0, 1].
            Represents the minimum similarity score between two sets to be considered similar.
            The similarity type is given by the type of hash used to generate the index inputs.
            E.g.,   *MinHash* hash function corresponds to Jaccard similarity,
                    *RandomProjections* hash functions corresponds to Cosine similarity.
        index_fp_fn_weights : Tuple[float, float]
            A pair of values between 0 and 1 denoting a preference for high precision or high recall.
            If the fp weight is higher then indexing precision is preferred. Otherwise, recall is preferred.
            Their sum has to be 1.
        index_seed : int
            The random seed for the underlying hash generator.
        index_cache_dir : str
            A file system path for storing the embedding model.

        """
        super(BertEmbeddingIndex, self).__init__(dataloader=dataloader, data_root=data_root)

        self.model_name = model_name
        self.num_samples = num_samples
        self.index_hash_size = index_hash_size
        self.index_similarity_threshold = index_similarity_threshold
        self.index_fp_fn_weights = index_fp_fn_weights
        self.index_seed = index_seed
        self.index_cache_dir = index_cache_dir

        self.transformer = BertTransformer(
            model_name=self.model_name,
            num_samples=self.num_samples,
            cache_dir=self.index_cache_dir
        )
        self.lsh_index = self.create_index()

    def create_index(self) -> LSHIndex:
        """
        Create the underlying LSH index with data from the configured dataloader.

        Returns
        -------
        LSHIndex
            A new LSH index.
        """

        lsh_index = LSHIndex(
            hash_size=self.index_hash_size,
            dimension=self.transformer.get_embedding_dimension(),
            similarity_threshold=self.index_similarity_threshold,
            fp_fn_weights=self.index_fp_fn_weights,
            seed=self.index_seed,
        )

        for table_name in tqdm(self.dataloader.get_table_names()):
            table_data = self.dataloader.read_table(table_name)

            column_signatures = [
                (c, self.transformer.transform(table_data[c].dropna().tolist()))
                for c in table_data.columns
                if not is_numeric(table_data[c]) and table_data[c].count() > 0
            ]

            for c, signature in column_signatures:
                if len(signature) > 0:
                    lsh_index.add(input_id=str(table_name) + "!" + str(c), input_set=signature)

        return lsh_index

    def query(
        self, query: Iterable[Any], k: Optional[int] = None
    ) -> Iterable[Tuple[str, float]]:
        """

        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query : Iterable[Any]
            A collection of values representing the query set.
        k : Optional[int]
            Only the top-k neighbours will be retrieved.
            If this is None all results are retrieved.

        Returns
        -------
        Iterable[Tuple[str, float]]
            A collection of (item id, score value) pairs.
            The item ids typically represent pre-indexed column ids.
            The score is a similarity measure between the query set and the indexed items.

        """
        if is_numeric(query):
            return []

        query_signature = self.transformer.transform(query)
        if len(query_signature) == 0:
            return []
        return self.lsh_index.query(
            query_id=None, query=query_signature, k=k, with_scores=True
        )


if __name__ == "__main__":
    transformer = BertTransformer("bert-base-uncased")
    tokenizer, model = transformer.get_embedding_model()
    x = ["I am", "apple microsoft google"]
    inputs = tokenizer(x, padding=True, truncation=True, return_tensors="pt")
    # rint(inputs)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)
    last_hidden_states = last_hidden_states[:, 0, :]
    print(last_hidden_states.shape)
    last_hidden_states = torch.mean(last_hidden_states, dim=0)
    print(last_hidden_states.shape)