# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Dict, List, Optional, Union
import dataclasses
import json
import sys
import time
import traceback

# Third Party
import torch
from huggingface_hub.utils._validators import HFValidationError
from peft.utils.other import fsdp_auto_wrap_policy
from torch.cuda import OutOfMemoryError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)
from transformers.utils import is_accelerate_available, logging
from trl import SFTConfig, SFTTrainer
import fire
import transformers
import datasets

# Local
from tuning.config import configs, peft_config
from tuning.config.acceleration_configs import (
    AccelerationFrameworkConfig,
    FusedOpsAndKernelsConfig,
    QuantizedLoraConfig,
)
from tuning.config.tracker_configs import (
    AimConfig,
    FileLoggingTrackerConfig,
    TrackerConfigFactory,
)
from tuning.data import tokenizer_data_utils
from tuning.trackers.tracker_factory import FILE_LOGGING_TRACKER, get_tracker
from tuning.trainercontroller import TrainerControllerCallback
from tuning.utils.config_utils import get_hf_peft_config, get_json_config
from tuning.utils.data_type_utils import get_torch_dtype
from tuning.utils.error_logging import (
    INTERNAL_ERROR_EXIT_CODE,
    USER_ERROR_EXIT_CODE,
    write_termination_log,
)
from tuning.utils.preprocessing_utils import (
    format_dataset,
    get_data_collator,
    validate_data_args,
)


def train(
    model_args: configs.ModelArguments,
    data_args: configs.DataArguments,
    train_args: configs.TrainingArguments,
    peft_config: Optional[  # pylint: disable=redefined-outer-name
        Union[peft_config.LoraConfig, peft_config.PromptTuningConfig]
    ] = None,
    trainer_controller_args: configs.TrainerControllerArguments = None,
    tracker_configs: Optional[TrackerConfigFactory] = TrackerConfigFactory(
        file_logger_config=FileLoggingTrackerConfig()
    ),
    additional_callbacks: Optional[List[TrainerCallback]] = None,
    exp_metadata: Optional[Dict] = None,
    quantized_lora_config: Optional[QuantizedLoraConfig] = None,
    fusedops_kernels_config: Optional[FusedOpsAndKernelsConfig] = None,
    packing_mode:str = None,
    use_hf_trainer:bool = False,
    num_samples:int = 0,
    goldfish_prob:float = 0.,
    attention_dropout:float = 0.,
    dont_freeze:list = [],
    trust_remote_code:bool = False,
    pemoet_path:str = "",
    is_calm:bool=False,
):
    """Call the SFTTrainer

    Args:
        model_args: tuning.config.configs.ModelArguments
        data_args: tuning.config.configs.DataArguments
        train_args: tuning.config.configs.TrainingArguments
        peft_config: peft_config.LoraConfig for Lora tuning | \
        peft_config.PromptTuningConfig for prompt tuning | \
        None for fine tuning
            The peft configuration to pass to trainer
        trainer_control_args: configs.TrainerControllerArguments \
            for controlling the training loop using policy rules
        tracker_configs: An instance of tuning.config.tracker_configs.TrackerConfigFactory \
                         which represents the configuration for various trackers\
                         Note, trackers need to be enabled to use this \
                         for e.g. --tracker(s) aim \
        additional_callbacks: List of callbacks to attach with SFTtrainer,\
                              besides those associated with experiment trackers \
                              or TrainerControllers. Callbacks associated with \
                              tracker with automatically be added.
        exp_metadata: Dict of key value pairs passed to train to be recoreded by the tracker.
        quantized_lora_config: tuning.config.acceleration_configs.QuantizedLoraConfig \
            Should be used in combination with peft_config.LoraConfig for Lora tuning \
        fusedops_kernels_config: tuning.config.acceleration_configs.FusedOpsAndKernelsConfig \
            Should be used in combination with quantized_lora_config. Also currently 
            fused_lora and fast_kernels must used together (may change in future). \
    """

    logger = logging.get_logger("sft_trainer")

    # Validate parameters
    if (not isinstance(train_args.num_train_epochs, (float, int))) or (
        train_args.num_train_epochs <= 0
    ):
        raise ValueError("num_train_epochs has to be an integer/float >= 1")
    if (not isinstance(train_args.gradient_accumulation_steps, int)) or (
        train_args.gradient_accumulation_steps <= 0
    ):
        raise ValueError("gradient_accumulation_steps has to be an integer >= 1")

    task_type = "CAUSAL_LM"
    additional_metrics = {}

    # Initialize Trackers And Callbacks
    trackers = []
    trainer_callbacks = []

    if exp_metadata and (not isinstance(exp_metadata, dict)):
        raise ValueError("exp metadata passed should be a dict with valid json")

    if train_args.trackers is not None:
        requested_trackers = set(train_args.trackers)
    else:
        requested_trackers = set()

    # Ensure file logging is present
    if FILE_LOGGING_TRACKER not in requested_trackers:
        requested_trackers.add(FILE_LOGGING_TRACKER)

    if not isinstance(tracker_configs, TrackerConfigFactory):
        raise ValueError(
            "tracker configs should adhere to the TrackerConfigFactory type"
        )

    # Now initialize trackers one by one
    for name in requested_trackers:
        t = get_tracker(name, tracker_configs)
        cb = t.get_hf_callback()
        if cb is not None:
            trainer_callbacks.append(cb)
            trackers.append(t)

    # Now add trainer controller callbacks if requested
    if (trainer_controller_args is not None) and (
        trainer_controller_args.trainer_controller_config_file is not None
    ):
        tc_callback = TrainerControllerCallback(
            trainer_controller_args.trainer_controller_config_file,
        )
        trainer_callbacks.append(tc_callback)

    # Add any extra callback if passed by users
    if additional_callbacks is not None:
        for cb in additional_callbacks:
            if not isinstance(cb, TrainerCallback):
                raise ValueError(
                    "additional callbacks should be of type TrainerCallback"
                )
            trainer_callbacks.append(cb)

    framework = AccelerationFrameworkConfig.from_dataclasses(
        quantized_lora_config, fusedops_kernels_config
    ).get_framework()

    model_loader = AutoModelForCausalLM.from_pretrained
    if framework is not None and framework.requires_custom_loading:
        model_loader = framework.model_loader  # drop-in new loader
    model_load_time = time.time()

    if not is_calm:
        model = model_loader(
            model_args.model_name_or_path,
            cache_dir=train_args.cache_dir,
            torch_dtype=get_torch_dtype(model_args.torch_dtype),
            attn_implementation="flash_attention_2" if model_args.use_flash_attn else None,
            attention_dropout=attention_dropout,
            trust_remote_code=trust_remote_code
        )
    else:
        from model import calm
        config = calm.CALMConfig.from_pretrained(args.model_path)
        model = calm.CALM.from_pretrained(args.model_path, torch_dtype=dtype, config=config)

    if pemoet_path:
        import pemoet
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, pemoet_path,
        )

    # TODO: Move these to a config as well
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name_or_path
            if model_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        cache_dir=train_args.cache_dir,
        use_fast=True,
    )

    # Calculate and save additional metrics to track later.
    additional_metrics["model_load_time"] = time.time() - model_load_time

    peft_config = get_hf_peft_config(
        task_type,
        peft_config,
        (
            model_args.tokenizer_name_or_path
            if model_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
    )

    # add special tokens only when a custom tokenizer is not passed
    if not model_args.tokenizer_name_or_path:
        # TODO: understand if we need to hardcode these here or just use defaults in model
        if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            tokenizer.add_special_tokens(
                {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                }
            )
        elif isinstance(tokenizer, (GPT2Tokenizer, GPTNeoXTokenizerFast)):
            tokenizer.add_special_tokens(
                {
                    "pad_token": "<pad>",
                }
            )

    max_seq_length = min(train_args.max_seq_length, tokenizer.model_max_length)
    logger.info("Max sequence length is %s", max_seq_length)
    if train_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "max_seq_length %s exceeds tokenizer.model_max_length \
            %s, using tokenizer.model_max_length %s",
            train_args.max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )

    # add special tokens only when a custom tokenizer is not passed
    special_tokens_dict = {}
    if not model_args.tokenizer_name_or_path:
        # TODO: we need to change this, perhaps follow what open instruct does?
        if tokenizer.pad_token is None:
            logger.warning("PAD token set to default, missing in tokenizer")
            special_tokens_dict["pad_token"] = configs.DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            logger.warning("EOS token set to default, missing in tokenizer")
            special_tokens_dict["eos_token"] = configs.DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            logger.warning("BOS token set to default, missing in tokenizer")
            special_tokens_dict["bos_token"] = configs.DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            logger.warning("UNK token set to default, missing in tokenizer")
            special_tokens_dict["unk_token"] = configs.DEFAULT_UNK_TOKEN

    # TODO: lower priority but understand if resizing impacts inference quality and why its needed.
    # It makes sense if we manipulate tokenizer that we also save it and provide it to inference.
    tokenizer_data_utils.tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
        multiple_of=model_args.embedding_size_multiple_of,
    )

    trainer_data_kwargs = dict(
        formatting_func = lambda x:x,
        dataset_kwargs = {"skip_prepare_dataset": True},
    )

    def read_data(path):

        if path.endswith("json") or path.endswith("jsonl"):
            dataset = datasets.load_dataset("json", data_files=path)
        else:
            dataset = datasets.load_from_disk(path)
        train_dataset = dataset["train"]

        if data_args.validation_data_path:
            eval_dataset = datasets.load_dataset(
                "json", data_files=data_args.validation_data_path
            )['train']
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
        else:
            _ds = train_dataset.train_test_split(test_size=0.1, seed=train_args.seed)
            train_dataset = _ds["train"]
            eval_dataset = _ds["test"]

        if num_samples:
            train_dataset = train_dataset.select(range(min(len(train_dataset), num_samples)))
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), num_samples)))
        
        return train_dataset, eval_dataset
    
    if not "," in data_args.training_data_path:
        train_dataset, eval_dataset = read_data(data_args.training_data_path)
    else:
        data_paths = data_args.training_data_path.split(",")
        train_dataset_list, eval_dataset_list = [], []
        for dp in data_paths:
            train_dataset, eval_dataset = read_data(dp)
            train_dataset_list.append(train_dataset)
            eval_dataset_list.append(eval_dataset)
        train_dataset = datasets.concatenate_datasets(train_dataset_list)
        eval_dataset = datasets.concatenate_datasets(eval_dataset_list)

    def truncate(examples):
        return {
            "input_ids":[x[:max_seq_length] for x in examples["input_ids"]],
            "labels":[x[:max_seq_length] for x in examples["labels"]],
        }

    train_dataset = train_dataset.map(truncate, batched=True)
    eval_dataset = eval_dataset.map(truncate, batched=True)

    if packing_mode == "minibatch":
        from transformers import DataCollatorWithFlattening
        data_collator = DataCollatorWithFlattening(goldfish_prob=goldfish_prob)
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, max_length=max_seq_length
        )

    if data_args.validation_data_path:
        trainer_data_kwargs.update(dict(
            eval_dataset = datasets.load_dataset(
                "json", data_files=data_args.validation_data_path)['train']
            )
        )
    ### JSON file formatting ends here

    if framework is not None and framework.requires_agumentation:
        model, (peft_config,) = framework.augmentation(
            model, train_args, modifiable_args=(peft_config,)
        )

    # HACK - The SFT Trainer has internal validation which inspects the name of the class
    # being used for the HF training args; if it's a TrainingArguments class, which is
    # presumably from transformers, it tries to build it into an SFT Config.
    #
    # This is unfortunately a naming collision with one of our own classes, which has extra
    # fields, and therefore can't be used to initialize the SFT Config. For now, to sidestep
    # this validation, we just drop the things that aren't part of the SFT Config and build one
    # from our object directly. In the future, we should consider renaming this class and / or
    # not adding things that are not directly used by the trainer instance to it.
    transformer_train_arg_fields = [x.name for x in dataclasses.fields(SFTConfig)]
    transformer_kwargs = {
        k: v
        for k, v in train_args.to_dict().items()
        if k in transformer_train_arg_fields
    }
    transformer_kwargs["gradient_checkpointing_kwargs"] = {
        "use_reentrant": False
    }
    if transformer_kwargs["lr_scheduler_type"] == "steplr":
        transformer_kwargs["lr_scheduler_kwargs"] = {
            "gamma": 0.33,
            "step_size": len(train_dataset) // transformer_kwargs["gradient_accumulation_steps"] // transformer_kwargs["per_device_train_batch_size"] // torch.distributed.get_world_size()
        }

    if dont_freeze:
        for name, param in model.named_parameters():
            if not any([x in name for x in dont_freeze]):
                param.requires_grad = False
            else:
                logger.info("Training {}!".format(name))
                param.requires_grad = True
        logger.info("Frozen model number of parameters: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))
        #model.enable_input_require_grads()
        
    if not use_hf_trainer:
        training_args = SFTConfig(**transformer_kwargs)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            dataset_text_field=data_args.dataset_text_field,
            args=training_args,
            packing=False,
            max_seq_length=max_seq_length,
            callbacks=trainer_callbacks,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **trainer_data_kwargs
        )
    else:
        from transformers import TrainingArguments, Trainer
        dummy = TrainingArguments("")
        to_remove = []
        for k in transformer_kwargs:
            if not hasattr(dummy, k):
                logger.info(f"Removing `{k}` from `transformer_kwargs`")
                to_remove.append(k)
        for k in to_remove:
            transformer_kwargs.pop(k)
        training_args = TrainingArguments(**transformer_kwargs)
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )


    # We track additional metrics and experiment metadata after trainer object creation
    # this ensure that the process is not repeated multiple times for FSDP runs.
    if trainer.is_world_process_zero():
        # Currently tracked only on process zero.
        for tracker in trackers:
            try:
                for k, v in additional_metrics.items():
                    tracker.track(metric=v, name=k, stage="additional_metrics")
                    tracker.set_params(params=exp_metadata, name="experiment_metadata")
            except ValueError as e:
                logger.error(
                    "Exception while saving additional metrics and metadata %s",
                    repr(e),
                )

    if trainer.is_fsdp_enabled and peft_config is not None:
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(
            model
        )

    if framework is not None:
        accelerator = None if not is_accelerate_available else trainer.accelerator

        # ready for train may produce additional callbacks for the trainer
        for x in framework.get_callbacks_and_ready_for_train(model, accelerator):
            trainer.add_callback(x)

    o = trainer._evaluate(None, None, skip_scheduler=True)
    import numpy as np
    import os
    top1 = []
    top2 = []
    if trainer.args.world_size > 1:
        for layer_idx in range(len(model.model.layers)):
            _top1 = trainer.accelerator.reduce(model.model.layers[layer_idx].mlp.assignment_top1.cuda()).cpu().numpy()
            _top2 = trainer.accelerator.reduce(model.model.layers[layer_idx].mlp.assignment_top2.cuda()).cpu().numpy()
            top1.append(_top1)
            top2.append(_top2)
    else:
        for layer_idx in range(len(model.model.layers)):
            top1.append(model.model.layers[layer_idx].mlp.assignment_top1.int().cpu().numpy())
            top2.append(model.model.layers[layer_idx].mlp.assignment_top2.int().cpu().numpy())
    if trainer.args.distributed_state.is_main_process:
        top1 = np.stack(top1)
        top2 = np.stack(top2)
        np.save(os.path.join(train_args.output_dir, "top1.npy"), top1)
        np.save(os.path.join(train_args.output_dir, "top2.npy"), top2)
    trainer.train()


def get_parser():
    """Get the command-line argument parser."""
    parser = transformers.HfArgumentParser(
        dataclass_types=(
            configs.ModelArguments,
            configs.DataArguments,
            configs.TrainingArguments,
            configs.TrainerControllerArguments,
            peft_config.LoraConfig,
            peft_config.PromptTuningConfig,
            FileLoggingTrackerConfig,
            AimConfig,
            QuantizedLoraConfig,
            FusedOpsAndKernelsConfig,
        )
    )
    parser.add_argument(
        "--peft_method",
        type=str.lower,
        choices=["pt", "lora", None, "none"],
        default="none",
    )
    parser.add_argument(
        "--exp_metadata",
        type=str,
        default=None,
        help='Pass a json string representing K:V pairs to be associated\
              to the tuning run in the tracker. e.g. \'{"gpu":"A100-80G"}\'',
    )
    parser.add_argument(
        "--packing_mode",
        type=str,
        default=None,
        choices=["minibatch"],
    )

    parser.add_argument(
        "--goldfish_prob",
        type=float,
        default=0.,
    )

    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.,
    )

    parser.add_argument(
        "--use_hf_trainer",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--trust_remote_code",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dont_freeze",
        nargs="+",
        default=[]
    )

    parser.add_argument(
        "--pemoet_path",
        type=str,
        default="",
    )

    parser.add_argument(
        "--is_calm",
        default=False,
        action="store_true",
    )

    return parser


def parse_arguments(parser, json_config=None):
    """Parses arguments provided either via command-line or JSON config.

    Args:
        parser: argparse.ArgumentParser
            Command-line argument parser.
        json_config: dict[str, Any]
            Dict of arguments to use with tuning.

    Returns:
        ModelArguments
            Arguments pertaining to which model we are going to tune.
        DataArguments
            Arguments pertaining to what data we are going to use for training and evaluation.
        TrainingArguments
            Configuration for training model.
        TrainerControllerArguments
            Configuration for custom trainer controller such as early stopping or dynamic scaling.
        PromptTuningConfig/LoraConfig/None
            Configuration for running PEFT, different depending on type of PEFT.
        FileLoggingTrackerConfig
            Configuration for training log file.
        AimConfig
            Configuration for AIM stack.
        QuantizedLoraConfig
            Configuration for quantized LoRA (a form of PEFT).
        FusedOpsAndKernelsConfig
            Configuration for fused operations and kernels.
        dict[str, str]
            Extra AIM metadata.
    """
    if json_config:
        (
            model_args,
            data_args,
            training_args,
            trainer_controller_args,
            lora_config,
            prompt_tuning_config,
            file_logger_config,
            aim_config,
            quantized_lora_config,
            fusedops_kernels_config,
        ) = parser.parse_dict(json_config, allow_extra_keys=True)
        peft_method = json_config.get("peft_method")
        exp_metadata = json_config.get("exp_metadata")
    else:
        (
            model_args,
            data_args,
            training_args,
            trainer_controller_args,
            lora_config,
            prompt_tuning_config,
            file_logger_config,
            aim_config,
            quantized_lora_config,
            fusedops_kernels_config,
            additional,
            _,
        ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        peft_method = additional.peft_method
        exp_metadata = additional.exp_metadata
        packing_mode = additional.packing_mode
        use_hf_trainer = additional.use_hf_trainer
        num_samples = additional.num_samples
        goldfish_prob = additional.goldfish_prob
        attention_dropout = additional.attention_dropout
        dont_freeze = additional.dont_freeze
        trust_remote_code = additional.trust_remote_code
        pemoet_path = additional.pemoet_path
        is_calm = additional.is_calm

    if peft_method == "lora":
        tune_config = lora_config
    elif peft_method == "pt":
        tune_config = prompt_tuning_config
    else:
        tune_config = None

    return (
        model_args,
        data_args,
        training_args,
        trainer_controller_args,
        tune_config,
        file_logger_config,
        aim_config,
        quantized_lora_config,
        fusedops_kernels_config,
        exp_metadata,
        packing_mode,
        use_hf_trainer,
        num_samples,
        goldfish_prob,
        attention_dropout,
        dont_freeze,
        trust_remote_code,
        pemoet_path,
        is_calm
    )


def main(**kwargs):  # pylint: disable=unused-argument
    logger = logging.get_logger("__main__")

    parser = get_parser()
    job_config = get_json_config()
    logger.debug("Input args parsed: %s", job_config)
    # accept arguments via command-line or JSON
    try:
        (
            model_args,
            data_args,
            training_args,
            trainer_controller_args,
            tune_config,
            file_logger_config,
            aim_config,
            quantized_lora_config,
            fusedops_kernels_config,
            exp_metadata,
            packing_mode,
            use_hf_trainer,
            num_samples,
            goldfish_prob,
            attention_dropout,
            dont_freeze,
            trust_remote_code,
            pemoet_path,
            is_calm
        ) = parse_arguments(parser, job_config)
        logger.debug(
            "Input args parsed: \
            model_args %s, data_args %s, training_args %s, trainer_controller_args %s, \
            tune_config %s, file_logger_config, %s aim_config %s, \
            quantized_lora_config %s, fusedops_kernels_config %s, \
            exp_metadata %s, packing_mode %s",
            model_args,
            data_args,
            training_args,
            trainer_controller_args,
            tune_config,
            file_logger_config,
            aim_config,
            quantized_lora_config,
            fusedops_kernels_config,
            exp_metadata,
            packing_mode,
            use_hf_trainer,
            dont_freeze,
            trust_remote_code
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.error(traceback.format_exc())
        write_termination_log(
            f"Exception raised during training. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)

    # extra metadata passed via client
    metadata = None
    if exp_metadata is not None:
        try:
            metadata = json.loads(exp_metadata)
            if metadata is None or not isinstance(metadata, Dict):
                logger.warning(
                    "metadata cannot be converted to simple k:v dict ignoring"
                )
                metadata = None
        except ValueError as e:
            logger.error(
                "failed while parsing extra metadata. pass a valid json %s", repr(e)
            )

    combined_tracker_configs = TrackerConfigFactory()

    combined_tracker_configs.file_logger_config = file_logger_config
    combined_tracker_configs.aim_config = aim_config

    try:
        train(
            model_args=model_args,
            data_args=data_args,
            train_args=training_args,
            peft_config=tune_config,
            trainer_controller_args=trainer_controller_args,
            tracker_configs=combined_tracker_configs,
            additional_callbacks=None,
            exp_metadata=metadata,
            quantized_lora_config=quantized_lora_config,
            fusedops_kernels_config=fusedops_kernels_config,
            packing_mode=packing_mode,
            use_hf_trainer=use_hf_trainer,
            num_samples=num_samples,
            goldfish_prob=goldfish_prob,
            attention_dropout=attention_dropout,
            dont_freeze=dont_freeze,
            trust_remote_code=trust_remote_code,
            pemoet_path=pemoet_path,
            is_calm=is_calm
        )
    except (MemoryError, OutOfMemoryError) as e:
        logger.error(traceback.format_exc())
        write_termination_log(f"OOM error during training. {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)
    except FileNotFoundError as e:
        logger.error(traceback.format_exc())
        write_termination_log("Unable to load file: {}".format(e))
        sys.exit(USER_ERROR_EXIT_CODE)
    except HFValidationError as e:
        logger.error(traceback.format_exc())
        write_termination_log(
            f"There may be a problem with loading the model. Exception: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)
    except (TypeError, ValueError, EnvironmentError) as e:
        logger.error(traceback.format_exc())
        write_termination_log(
            f"Exception raised during training. This may be a problem with your input: {e}"
        )
        sys.exit(USER_ERROR_EXIT_CODE)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(traceback.format_exc())
        write_termination_log(f"Unhandled exception during training: {e}")
        sys.exit(INTERNAL_ERROR_EXIT_CODE)


if __name__ == "__main__":
    fire.Fire(main)
