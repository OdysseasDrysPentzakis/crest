from functools import partial
from itertools import chain
import datasets as hf_datasets
import nltk
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule


class ImdbDataModule(BaseDataModule):
    """DataModule for IMDB Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params)
        # Remove or update the path variable
        self.path = None  # No longer needed
        self.is_multilabel = True
        self.nb_classes = 2  # neg, pos

        # hyperparams
        self.batch_size = d_params.get("batch_size", 64)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.max_seq_len = d_params.get("max_seq_len", 99999999)
        self.max_dataset_size = d_params.get("max_dataset_size", None)
        self.create_validation_split = d_params.get("create_validation_split", True)

        # objects
        self.dataset = None
        self.label_encoder = None  # no label encoder for this dataset
        self.tokenizer = tokenizer
        self.tokenizer_cls = partial(
            StaticTokenizerEncoder,
            tokenize=nltk.wordpunct_tokenize,
            min_occurrences=self.vocab_min_occurrences,
            reserved_tokens=[
                constants.PAD,
                constants.UNK,
                constants.EOS,
                constants.SOS,
                "<copy>",
            ],
            padding_index=constants.PAD_ID,
            unknown_index=constants.UNK_ID,
            eos_index=constants.EOS_ID,
            sos_index=constants.SOS_ID,
            append_sos=False,
            append_eos=False,
        )

    def _collate_fn(self, samples: list, are_samples_batched: bool = False):
        """
        :param samples: a list of dicts
        :param are_samples_batched: in case a batch/bucket sampler are being used
        :return: dict of features, label (Tensor)
        """
        if are_samples_batched:
            # dataloader batch size is 1 -> the sampler is responsible for batching
            samples = samples[0]

        # convert list of dicts to dict of lists
        collated_samples = collate_tensors(samples, stack_tensors=list)

        # pad and stack input ids
        def pad_and_stack_ids(x):
            x_ids, x_lengths = stack_and_pad_tensors(x, padding_index=constants.PAD_ID)
            return x_ids, x_lengths

        def stack_labels(y):
            if isinstance(y, list):
                return torch.stack(y, dim=0)
            return y

        input_ids, lengths = pad_and_stack_ids(collated_samples["input_ids"])
        labels = stack_labels(collated_samples["label"])

        # keep tokens in raw format
        tokens = collated_samples["text"]

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "lengths": lengths,
            "tokens": tokens,
            "labels": labels,
        }
        return batch

    def prepare_data(self):
        # No need to download data
        pass

    def setup(self, stage: str = None):
        # Load the preprocessed CSV files
        train_path = "imdb_train_preprocessed.csv"
        test_path = "imdb_test_preprocessed.csv"

        # Load datasets from CSV files
        self.dataset = hf_datasets.load_dataset(
            "csv",
            data_files={
                "train": train_path,
                "test": test_path,
            }
        )

        # Remove unnecessary data (if any)
        if 'unsupervised' in self.dataset:
            del self.dataset['unsupervised']

        # Create validation split if required
        if self.create_validation_split:
            modified_dataset = self.dataset["train"].train_test_split(test_size=0.1)
            self.dataset["train"] = modified_dataset["train"]
            self.dataset["validation"] = modified_dataset["test"]

        # Cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(self.max_dataset_size))
            if self.create_validation_split:
                self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

        # Build tokenizer and label encoder
        if self.tokenizer is None:
            # Build tokenizer info (vocab + special tokens) based on train and validation set
            if self.create_validation_split:
                tok_samples = chain(
                    self.dataset["train"]["text"],
                    self.dataset["validation"]["text"]
                )
            else:
                tok_samples = self.dataset["train"]["text"]
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # Function to map strings to ids
        def _encode(example: dict):
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                example["input_ids"] = self.tokenizer(
                    example["text"].strip(),
                    padding=False,  # do not pad, padding will be done later
                    truncation=True,  # truncate to max length accepted by the model
                )["input_ids"]
            else:
                example["input_ids"] = self.tokenizer.encode(example["text"].strip())
            return example

        # Function to filter out examples longer than max_seq_len
        def _filter(example: dict):
            return len(example["input_ids"]) <= self.max_seq_len

        # Apply encode and filter
        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(_filter)

        # Convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "label"],
            output_all_columns=True,
        )