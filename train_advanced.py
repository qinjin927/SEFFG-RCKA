import argparse
import ast
import logging
import os
import random
import re

import numpy as np
import torch

from datasets.registry import dataset_names, get_dataset_spec
from utils.logger import setlogger
from utils.train_utils_combines import train_utils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transfer_task(value):
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a task such as [[2],[3]]"
        ) from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or not all(isinstance(domain, list) and domain for domain in parsed)
        or not all(
            isinstance(index, int) for domain in parsed for index in domain
        )
    ):
        raise argparse.ArgumentTypeError("expected two non-empty integer lists")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the complete GSCRDA model"
    )

    parser.add_argument("--dataset", choices=dataset_names(), default="SJTU")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--transfer_task", type=transfer_task, default=[[0], [1]]
    )
    parser.add_argument("--normlizetype", default="mean-std")
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--checkpoint_dir", default="./results/")
    parser.add_argument("--run_id", default="")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_epoch", type=int, default=60)
    parser.add_argument("--middle_epoch", type=int, default=20)
    parser.add_argument("--print_step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher_seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)

    parser.add_argument("--bottleneck_num", type=int, default=256)
    parser.add_argument("--graph_k_neighbors", type=int, default=4)
    parser.add_argument("--max_criticality", type=float, default=0.35)
    parser.add_argument("--dvsca_rl_episodes", type=int, default=240)
    parser.add_argument("--dvsca_rl_min_reward_gain", type=float, default=0.001)
    parser.add_argument("--dvsca_rl_trust_budget", type=float, default=0.375)
    parser.add_argument("--adversarial_hidden_size", type=int, default=256)

    parser.add_argument("--scale_augmentation_weight", type=float, default=0.1)
    parser.add_argument("--target_information_weight", type=float, default=0.1)
    parser.add_argument("--target_mcc_weight", type=float, default=0.1)
    parser.add_argument("--conflict_adversarial_weight", type=float, default=1.0)
    parser.add_argument("--semantic_adversarial_weight", type=float, default=0.1)

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = get_dataset_spec(args.dataset).default_data_dir

    if args.batch_size < 1 or args.num_workers < 0 or args.bottleneck_num < 1:
        parser.error(
            "batch_size/bottleneck_num must be positive and "
            "num_workers nonnegative"
        )
    if args.graph_k_neighbors < 1:
        parser.error("graph_k_neighbors must be positive")
    if not 0.0 < args.max_criticality <= 1.0:
        parser.error("max_criticality must be in (0, 1]")
    if not 0 < args.middle_epoch < args.max_epoch:
        parser.error("middle_epoch must satisfy 0 < middle_epoch < max_epoch")
    if args.early_stopping_patience < 1:
        parser.error("early_stopping_patience must be positive")
    if args.early_stopping_min_delta < 0:
        parser.error("early_stopping_min_delta must be nonnegative")
    if args.dvsca_rl_episodes < 1:
        parser.error("dvsca_rl_episodes must be positive")
    if args.dvsca_rl_min_reward_gain < 0:
        parser.error("dvsca_rl_min_reward_gain must be nonnegative")
    if not 0 < args.dvsca_rl_trust_budget <= 0.5:
        parser.error("dvsca_rl_trust_budget must be in (0, 0.5]")
    if args.adversarial_hidden_size < 1:
        parser.error("adversarial_hidden_size must be positive")
    if any(
        weight < 0
        for weight in (
            args.scale_augmentation_weight,
            args.target_information_weight,
            args.target_mcc_weight,
            args.conflict_adversarial_weight,
            args.semantic_adversarial_weight,
        )
    ):
        parser.error("loss weights must be nonnegative")

    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device.strip()

    print(torch.__version__)
    print(f"[*] Experiment seed locked to: {args.seed}")

    task = "-".join(
        "".join(str(value) for value in side) for side in args.transfer_task
    )
    task = re.sub(r"[^a-zA-Z0-9_-]", "", task)
    model_name = "SoftEventEFFGCN_features"
    run_name = f"{model_name}-{args.dataset}-{task}-targetearlystop"
    if args.run_id:
        run_name += f"_run{args.run_id}"

    save_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    setlogger(os.path.join(save_dir, "train.log"))

    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)
    logging.info(
        "protocol: target_validation_early_stopping | model: %s | data: %s | "
        "backbone: full | adaptation: rl_ma_dvsca_rgda | "
        "target_train: unlabeled_adaptation_only | "
        "target_val_inputs_and_labels: per_epoch_early_stopping_and_final_reporting",
        model_name,
        args.dataset,
    )

    trainer = train_utils(args, save_dir)
    trainer.setup()
    return trainer.train()


if __name__ == "__main__":
    main()
