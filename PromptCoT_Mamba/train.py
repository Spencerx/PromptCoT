from datasets import load_dataset, load_from_disk
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union, Dict, Any, List, Tuple, Sequence

import sys
import copy
import torch
import datetime

torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=12000))


import random
import yaml
import transformers
from transformers.hf_argparser import DataClass, DataClassType
import os

from trl import SFTTrainer

IGNORE_INDEX = -100

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = field(default="/ossfs/workspace/nas/xueliang/hf_models/Meta-Llama-3.1-8B")
    tokenizer_path: Optional[str] = field(default="/ossfs/workspace/nas/xueliang/hf_models/Meta-Llama-3.1-8B")


@dataclass
class SFTConfig(transformers.TrainingArguments):
    # Parameters that control the model
    model_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the column that contains text data in the dataset."},
    )
    dataset_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments for the dataset preparation. The only supported key is "
            "`skip_prepare_dataset`."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    pad_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from"
            "the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to pack multiple sequences into a fixed-length format. Uses `max_length` to define "
            "sequence length."
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, "
            "this is only supported with the `flash_attention_2` attention implementation, which can efficiently "
            "handle the flattened batch structure."
        },
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=2.0e-5,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`TrainingArguments`."
        },
    )

    # Deprecated parameters
    dataset_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. You can safely remove this "
            "parameter from your configuration."
        },
    )
    num_of_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. Use `max_length` instead, "
            "which specifies the maximum length of the tokenized sequence, unlike `num_of_sequences`, which referred "
            "to string sequences."
        },
    )
    chars_per_token: Optional[float] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. If you want to customize the "
            "packing length, use `max_length`."
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.20.0. Use `max_length` instead."
        },
    )
    use_liger: Optional[bool] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. Use `use_liger_kernel` "
            "instead."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.dataset_batch_size is not None:
            warnings.warn(
                "`dataset_batch_size` is deprecated and will be removed in version 0.18.0. You can safely remove this "
                "parameter from your configuration.",
                DeprecationWarning,
            )

        if self.num_of_sequences is not None:
            warnings.warn(
                "`num_of_sequences` is deprecated and will be removed in version 0.18.0. Use `max_length` instead, "
                "which specifies the maximum length of the tokenized sequence, unlike `num_of_sequences`, which "
                "referred to string sequences.",
                DeprecationWarning,
            )

        if self.chars_per_token is not None:
            warnings.warn(
                "`chars_per_token` is deprecated and will be removed in version 0.18.0. If you want to customize the "
                "packing length, use `max_length`.",
                DeprecationWarning,
            )

        if self.max_seq_length is not None:
            warnings.warn(
                "`max_seq_length` is deprecated and will be removed in version 0.20.0. Use `max_length` instead.",
                DeprecationWarning,
            )
            self.max_length = self.max_seq_length

        if self.use_liger is not None:
            warnings.warn(
                "`use_liger` is deprecated and will be removed in version 0.18.0. Use `use_liger_kernel` instead.",
                DeprecationWarning,
            )
            self.use_liger_kernel = self.use_liger


class TrlParser(transformers.HfArgumentParser):

    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self, args: Optional[Iterable[str]] = None, return_remaining_strings: bool = False
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        args = list(args) if args is not None else sys.argv[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> List[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments.

        Any argument with an updated default will also be marked as not required
        if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """
        # If an argument is in the kwargs, update its default and set it as not required
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs.pop(action.dest)
                action.required = False
        remaining_strings = [item for key, value in kwargs.items() for item in [f"--{key}", str(value)]]
        return remaining_strings


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSuperviseDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, items: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([item[key] for item in items] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )


class SkipBadBF16BatchTrainer(SFTTrainer):
    def __init__(self, *args, max_bad_skips: int = 10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.bad_skips = 0
        self.max_bad_skips = max_bad_skips

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            # forward + backward + (on accumulation boundary) engine.step()
            return super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            msg = str(e).lower()
            if "expected a floating point or complex tensor as input" in msg and "got long" in msg:
                # 1) count & warn
                self.bad_skips += 1
                warnings.warn(
                    f"[SkipBF16] Encountered Long‐tensor in vector_norm (batch #{self.bad_skips}), skipping.",
                    UserWarning,
                )

                # 2) clear any gradients to avoid leakage
                model.zero_grad()
                if hasattr(self, "optimizer") and self.optimizer is not None:
                    self.optimizer.zero_grad()

                # 3) safety bail if too many
                if self.bad_skips >= self.max_bad_skips:
                    raise RuntimeError(
                        f"Too many bf16_vector_norm skips ({self.bad_skips}). Aborting to avoid infinite loop."
                    )

                # 4) return zero‐loss so Trainer moves to next batch
                return torch.tensor(0.0, device=self.args.device)
        # any other error should still bubble up
        raise


def train_tokenize_function(examples, tokenizer):
    sources = [prompt for prompt in examples["prompt"]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples["completion"]]
    return preprocess(sources, targets, tokenizer)


def train():
    parser = TrlParser((DataArguments, SFTConfig, ModelConfig))
    data_args, training_args, model_args = parser.parse_args_and_config()

    if "gemma" in model_args.model_name_or_path:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype="auto",
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # attn_implementation="flash_attention_2",
            # torch_dtype=torch.float16,  # works for first 2000 steps
            torch_dtype="auto",
            use_cache=False if training_args.gradient_checkpointing else True,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        model_max_length=training_args.max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_train_dataset = load_dataset(
        "json",
        data_files=data_args.data_path,
        split="train",
    )

    if training_args.local_rank > 0:
        torch.distributed.barrier()

    train_dataset = raw_train_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=4096,
        num_proc=1,
        remove_columns=raw_train_dataset.column_names,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
        }
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()

    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForSuperviseDataset(tokenizer=tokenizer)
    trainer = SkipBadBF16BatchTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
