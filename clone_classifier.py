
# import tempfile
# import multiprocessing

# import torch
# import numpy as np
# import pandas as pd

# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModel, TrainingArguments

# from utils.encoder import Encoder
# from utils.trainer import ImprovedRDropTrainer
# from utils.collator import DataCollatorWithPadding
# from utils.preprocessor import AnnotationPreprocessor, FunctionPreprocessor


# class CloneClassifier:
#     """
#     A class that integrates data preprocessing, input tokenization, and model
#     inferencing. It takes in a pandas dataframe with two columns: "code1" and
#     "code2", and returns the predictions as a dataframe with similarity scores.
#     """

#     PLM = "Lazyhope/python-clone-detection"  # HuggingFace model name

#     def __init__(
#         self,
#         max_token_size=512,
#         fp16=False,
#         per_device_eval_batch_size=32,
#     ):
#         # -- Tokenizing & Encoding
#         self.tokenizer = AutoTokenizer.from_pretrained(self.PLM)
#         self.encoder = Encoder(self.tokenizer, max_input_length=max_token_size)

#         # -- Collator
#         self.data_collator = DataCollatorWithPadding(
#             tokenizer=self.tokenizer, max_length=max_token_size
#         )

#         # -- Config & Model
#         self.model = AutoModel.from_pretrained(self.PLM, trust_remote_code=True)

#         with tempfile.TemporaryDirectory() as tmpdirname:
#             training_args = TrainingArguments(
#                 output_dir=tmpdirname,  # output_dir is not needed for inference
#                 per_device_eval_batch_size=per_device_eval_batch_size,
#                 fp16=fp16,
#                 remove_unused_columns=False,
#             )
#             self.trainer = ImprovedRDropTrainer(
#                 model=self.model,
#                 args=training_args,
#                 data_collator=self.data_collator,
#             )

#     def prepare_inputs(self, df: pd.DataFrame):
#         """Data preprocessing and tokenization."""
#         # -- Loading datasets
#         dset = Dataset.from_pandas(df)

#         # -- Preprocessing datasets
#         CPU_COUNT = multiprocessing.cpu_count() // 2

#         fn_preprocessor = FunctionPreprocessor()
#         dset = dset.map(fn_preprocessor, batched=True, num_proc=CPU_COUNT)

#         an_preprocessor = AnnotationPreprocessor()
#         dset = dset.map(an_preprocessor, batched=True, num_proc=CPU_COUNT)

#         dset = dset.map(
#             self.encoder,
#             batched=True,
#             num_proc=multiprocessing.cpu_count(),
#             remove_columns=dset.column_names,
#         )

#         return dset

#     def predict(
#         self, df: pd.DataFrame, save_path: str = None
#     ) -> pd.DataFrame:
#         """
#         Perform model inference and return predictions as a dataframe with similarity scores.

#         Args:
#             df (pd.DataFrame): DataFrame containing 'code1' and 'code2' columns.
#             save_path (str, optional): Path to save the results as a CSV file. Defaults to None.

#         Returns:
#             pd.DataFrame: DataFrame with 'code1', 'code2', 'predictions', and 'similarity_score' columns.
#         """
#         # -- Preparing inputs
#         dset = self.prepare_inputs(df)

#         # -- Inference
#         # outputs = self.trainer.predict(dset)[0]  # logits output
#         # scores = torch.Tensor(outputs).softmax(dim=-1).numpy()  # probability output

#         outputs = self.trainer.predict(dset)  # This gives a tuple (predictions, label_ids, metrics)
#         logits = outputs[0]  # logits output
#         scores = torch.Tensor(logits).softmax(dim=-1).numpy()  # probability output

#         results = df[["code1", "code2"]].copy()
#         results["predictions"] = np.argmax(scores, axis=-1)
#         # score of positive class
#         if scores.ndim == 1:
#             results["similarity_score"] = scores[1] * 100  # Single pair
#         else:
#             results["similarity_score"] = scores[:, 1] * 100  # Multiple pairs

#         if save_path is not None:
#             results.to_csv(save_path, index=False)

#         return results
    

#     def get_similarity_score(self, target_code: str, compare_code: str) -> float:
#         """
#         Calculate the similarity score between two code snippets.

#         Args:
#             target_code (str): The target code snippet.
#             compare_code (str): The code snippet to compare against.

#         Returns:
#             float: Similarity score as a percentage.
#         """
#         df = pd.DataFrame({
#             'code1': [target_code],
#             'code2': [compare_code]
#         })
#         result = self.predict(df)
#         return result['similarity_score'].iloc[0]

#     def get_top_k_similar(self, target_code: str, code_repository: list, k: int = 5) -> pd.DataFrame:
#         """
#         Retrieve the top k similar code snippets from the repository based on similarity scores.

#         Args:
#             target_code (str): The target code snippet.
#             code_repository (list): List of code snippets to compare against.
#             k (int, optional): Number of top similar codes to retrieve. Defaults to 5.

#         Returns:
#             pd.DataFrame: DataFrame containing the top k similar codes and their similarity scores.
#         """
#         # Prepare DataFrame for prediction
#         df = pd.DataFrame({
#             'code1': [target_code] * len(code_repository),
#             'code2': code_repository
#         })

#         # Predict similarity scores
#         results = self.predict(df)

#         # Sort by similarity score in descending order and select top k
#         top_k = results.sort_values(by='similarity_score', ascending=False).head(k).reset_index(drop=True)

#         return top_k


import tempfile
import multiprocessing

import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments

from utils.encoder import Encoder
from utils.trainer import ImprovedRDropTrainer
from utils.collator import DataCollatorWithPadding
from utils.preprocessor import AnnotationPreprocessor, FunctionPreprocessor


class CloneClassifier:
    """
    A class that integrates data preprocessing, input tokenization, and model
    inferencing. It takes in a pandas dataframe with two columns: "code1" and
    "code2", and returns the predictions as a dataframe with similarity scores.
    """

    PLM = "Lazyhope/python-clone-detection"  # HuggingFace model name

    def __init__(
        self,
        max_token_size=512,
        fp16=False,
        per_device_eval_batch_size=32,
    ):
        # -- Tokenizing & Encoding
        self.tokenizer = AutoTokenizer.from_pretrained(self.PLM)
        self.encoder = Encoder(self.tokenizer, max_input_length=max_token_size)

        # -- Collator
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, max_length=max_token_size
        )

        # -- Config & Model
        self.model = AutoModel.from_pretrained(self.PLM, trust_remote_code=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            training_args = TrainingArguments(
                output_dir=tmpdirname,  # output_dir is not needed for inference
                per_device_eval_batch_size=per_device_eval_batch_size,
                fp16=fp16,
                remove_unused_columns=False,
            )
            self.trainer = ImprovedRDropTrainer(
                model=self.model,
                args=training_args,
                data_collator=self.data_collator,
            )

    def prepare_inputs(self, df: pd.DataFrame):
        """Data preprocessing and tokenization."""
        # -- Loading datasets
        dset = Dataset.from_pandas(df)

        # -- Preprocessing datasets
        CPU_COUNT = multiprocessing.cpu_count() // 2

        fn_preprocessor = FunctionPreprocessor()
        dset = dset.map(fn_preprocessor, batched=True, num_proc=CPU_COUNT)

        an_preprocessor = AnnotationPreprocessor()
        dset = dset.map(an_preprocessor, batched=True, num_proc=CPU_COUNT)

        dset = dset.map(
            self.encoder,
            batched=True,
            num_proc=multiprocessing.cpu_count(),
            remove_columns=dset.column_names,
        )

        return dset

    def predict(
        self, df: pd.DataFrame, save_path: str = None
    ) -> pd.DataFrame:
        """
        Perform model inference and return predictions as a dataframe with similarity scores.

        Args:
            df (pd.DataFrame): DataFrame containing 'code1' and 'code2' columns.
            save_path (str, optional): Path to save the results as a CSV file. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with 'code1', 'code2', 'predictions', and 'similarity_score' columns.
        """
        # -- Preparing inputs
        dset = self.prepare_inputs(df)

        # -- Inference
        outputs = self.trainer.predict(dset)[0]  # logits output

        print(outputs)
        scores = torch.Tensor(outputs).softmax(dim=-1).numpy()  # probability output
        print(scores)
        results = df[["code1", "code2"]].copy()
        results["predictions"] = np.argmax(scores, axis=-1)
        # score of positive class
        if scores.ndim == 1:
            results["similarity_score"] = scores[0] * 100  # Single pair
        else:
            results["similarity_score"] = scores[:, 1] * 100  # Multiple pairs

        if save_path is not None:
            results.to_csv(save_path, index=False)

        return results


    def predict_1(
        self, df: pd.DataFrame, save_path: str = None
    ) -> pd.DataFrame:
        """
        Perform model inference and return predictions as a dataframe with similarity scores.

        Args:
            df (pd.DataFrame): DataFrame containing 'code1' and 'code2' columns.
            save_path (str, optional): Path to save the results as a CSV file. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with 'code1', 'code2', 'predictions', and 'similarity_score' columns.
        """
        # -- Preparing inputs
        dset = self.prepare_inputs(df)

        # -- Inference
        outputs = self.trainer.predict(dset)[0]  # logits output

        print(outputs)
        scores = torch.Tensor(outputs).softmax(dim=-1).numpy()  # probability output
        print(scores)
        results = df[["code1", "code2"]].copy()
        results["predictions"] = np.argmax(scores, axis=-1)
        # score of positive class
        if scores.ndim == 1:
            results["similarity_score"] = scores[0] * 100  # Single pair
        else:
            results["similarity_score"] = scores[:, 1] * 100  # Multiple pairs
        results=results.iloc[:1]
        if save_path is not None:
            results.to_csv(save_path, index=False)

        return results

    def get_similarity_score(self, target_code: str, compare_code: str) -> float:
        """
        Calculate the similarity score between two code snippets.

        Args:
            target_code (str): The target code snippet.
            compare_code (str): The code snippet to compare against.

        Returns:
            float: Similarity score as a percentage.
        """
        df = pd.DataFrame({
            'code1': [target_code],
            'code2': [compare_code]
        })
        result = self.predict(df)
        return result['similarity_score'].iloc[0]

    def get_top_k_similar(self, target_code: str, code_repository: list, k: int = 5) -> pd.DataFrame:
        """
        Retrieve the top k similar code snippets from the repository based on similarity scores.

        Args:
            target_code (str): The target code snippet.
            code_repository (list): List of code snippets to compare against.
            k (int, optional): Number of top similar codes to retrieve. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing the top k similar codes and their similarity scores.
        """
        # Prepare DataFrame for prediction
        df = pd.DataFrame({
            'code1': [target_code] * len(code_repository),
            'code2': code_repository
        })

        # Predict similarity scores
        results = self.predict(df)

        # Sort by similarity score in descending order and select top k
        top_k = results.sort_values(by='similarity_score', ascending=False).head(k).reset_index(drop=True)

        return top_k