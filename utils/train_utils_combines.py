
import copy
import logging
import math
import os
import random
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from datasets.registry import get_dataset_spec
from models.SoftEventEFFGCN import SoftEventEFFGCN_features
from models.ConflictAwareDomain import (
    DomainDiscriminator,
    multilinear_condition,
    project_conflicting_gradient,
    weighted_binary_cross_entropy,
)
from utils.dual_view_spectral_uda import (
    MultiAnchorConsensusAdapter,
    PseudoLabeledSignalDataset,
)


class train_utils:
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.global_step = 0
        self.dataset_spec = get_dataset_spec(args.dataset)
        self.dataset_class = self.dataset_spec.dataset_class
        self.num_classes = self.dataset_spec.num_classes

    @staticmethod
    def _build_feature_extractor(args):
        return SoftEventEFFGCN_features(
            spectral_residual=False,
            backbone_variant="full",
            k_neighbors=args.graph_k_neighbors,
            max_criticality=args.max_criticality,
        )

    def setup(self):
        args = self.args
        device_count = torch.cuda.device_count()
        if torch.cuda.is_available() and device_count > 0:
            self.device = torch.device("cuda")
            self.device_count = device_count
            if args.batch_size % device_count:
                raise ValueError("batch size must be divisible by device count")
            logging.info("using %d gpus", device_count)
        else:
            warnings.warn("GPU unavailable; falling back to CPU")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info("using cpu")

        source_train, source_val, target_train, target_val = self.dataset_class(
            args.data_dir, args.transfer_task, args.normlizetype
        ).data_split(transfer_learning=True, k_shot=None, k_shot_target=None)
        self.datasets = {
            "source_train": source_train,
            "source_val": source_val,
            "target_train": target_train,
            "target_val": target_val,
        }
        source_counts = np.bincount(
            np.asarray(source_train.labels, dtype=np.int64),
            minlength=self.num_classes,
        ).astype(np.float32)
        self.source_class_prior = torch.from_numpy(
            source_counts / max(float(source_counts.sum()), 1.0)
        )
        self._build_label_blind_teacher()

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        def seed_worker(worker_id):
            worker_seed = args.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        self.dataloaders = {
            name: torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=name.endswith("train"),
                num_workers=args.num_workers,
                worker_init_fn=seed_worker if args.num_workers else None,
                generator=generator,
                pin_memory=self.device.type == "cuda",
                drop_last=False,
            )
            for name, dataset in self.datasets.items()
        }

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        self.model = self._build_feature_extractor(args)
        self.bottleneck_layer = nn.Sequential(
            nn.Linear(self.model.output_num(), args.bottleneck_num),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier_layer = nn.Linear(args.bottleneck_num, self.num_classes)
        self.model_all = nn.Sequential(
            self.model, self.bottleneck_layer, self.classifier_layer
        )
        self._advance_verified_trainer_rng_sequence()

        if self.device_count > 1:
            self.model = nn.DataParallel(self.model)
            self.bottleneck_layer = nn.DataParallel(self.bottleneck_layer)
            self.classifier_layer = nn.DataParallel(self.classifier_layer)

        parameters = [
            {"params": self.model.parameters(), "lr": args.lr},
            {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
            {"params": self.classifier_layer.parameters(), "lr": args.lr},
        ]
        self.optimizer = optim.Adam(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, args.gamma
        )
        self.criterion = nn.CrossEntropyLoss()
        self.adversarial_enabled = (
            args.conflict_adversarial_weight > 0.0
            or args.semantic_adversarial_weight > 0.0
        )
        if self.adversarial_enabled:
            self.global_discriminator = DomainDiscriminator(
                args.bottleneck_num, args.adversarial_hidden_size
            ).to(self.device)
            self.semantic_discriminator = DomainDiscriminator(
                args.bottleneck_num * self.num_classes,
                args.adversarial_hidden_size,
            ).to(self.device)
            self.discriminator_optimizer = optim.Adam(
                list(self.global_discriminator.parameters())
                + list(self.semantic_discriminator.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        self.start_epoch = 0

        self.model.to(self.device)
        self.bottleneck_layer.to(self.device)
        self.classifier_layer.to(self.device)

    def _advance_verified_trainer_rng_sequence(self):
        for input_size, output_size in ((256, 1024), (1024, 1024), (1024, 1)):
            nn.Linear(input_size, output_size)
            torch.empty(output_size).normal_()
            torch.empty(input_size).normal_()

    @staticmethod
    def _self_tempered_semantic_route(reliability):
        reliability = float(np.clip(reliability, 0.0, 1.0))
        exponent = 1.0 + (1.0 - reliability) ** 2
        return reliability ** exponent

    def _build_label_blind_teacher(self):
        args = self.args
        teacher_seed = int(getattr(args, "teacher_seed", args.seed))
        original_target = self.datasets["target_train"]
        target_signals = np.asarray(original_target.seq_data, dtype=np.float32)
        source_signals = np.asarray(
            self.datasets["source_train"].seq_data, dtype=np.float32
        )
        source_labels = np.asarray(
            self.datasets["source_train"].labels, dtype=np.int64
        )
        permutation = np.random.default_rng(teacher_seed).permutation(len(target_signals))
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(permutation))
        adapter = MultiAnchorConsensusAdapter(
            num_classes=self.num_classes,
            log_bins=128,
            cluster_components=32,
            anchor_components=32,
            n_init=50,
            random_state=teacher_seed,
            rl_view_selection=True,
            rl_episodes=args.dvsca_rl_episodes,
            rl_min_reward_gain=args.dvsca_rl_min_reward_gain,
            rl_trust_budget=args.dvsca_rl_trust_budget,
            rl_fixed_confidence_power=0.0,
        )
        result = adapter.fit_predict(
            source_signals, source_labels, target_signals[permutation]
        )
        teacher_labels = result.predictions[inverse]
        teacher_confidences = result.confidences[inverse]
        self.confidence_power = float(
            result.rl_selected_action.get(
                "confidence_power",
                0.0,
            )
        )
        self.datasets["target_train"] = PseudoLabeledSignalDataset(
            target_signals,
            teacher_labels,
            teacher_confidences,
            transform=original_target.transforms,
        )
        self.dvsca_teacher_result = result
        pairwise_agreement = float(
            result.rl_metrics.get("mean_pairwise_agreement", 0.0)
        )
        random_agreement = 1.0 / self.num_classes
        self.teacher_consensus_reliability = float(
            np.clip(
                (pairwise_agreement - random_agreement)
                / max(1.0 - random_agreement, 1e-8),
                0.0,
                1.0,
            )
        )
        reliability_power = 4.0
        reliable_mass = self.teacher_consensus_reliability ** reliability_power
        unreliable_mass = (
            1.0 - self.teacher_consensus_reliability
        ) ** reliability_power
        self.semantic_routing_reliability = float(
            reliable_mass / max(reliable_mass + unreliable_mass, 1e-12)
        )
        self.dvsca_quality_gate = float(
            result.rl_selected_action.get("quality_gate", 0.0)
        )
        logging.info(
            "[DVSCA Teacher] target_train_samples=%d | anchor=%s | "
            "order_shuffled=yes | target_val inputs/labels held out | "
            "target_train ground-truth labels untouched | rl_accepted=%s | "
            "rl_reward=%.6f | baseline_reward=%.6f | rl_action=%s",
            len(target_signals),
            result.anchor_mode,
            result.rl_accepted,
            result.rl_reward,
            result.rl_baseline_reward,
            result.rl_selected_action,
        )
        logging.info(
            "[DVSCA Confidence] selected_power=%.3f | controller=%s | "
            "pairwise_agreement=%.4f | normalized_reliability=%.4f | "
            "routing_reliability=%.4f | "
            "target ground-truth labels untouched",
            self.confidence_power,
            "sequential_actor_critic",
            pairwise_agreement,
            self.teacher_consensus_reliability,
            self.semantic_routing_reliability,
        )

    def _pack_model_state(self):
        return {"model_state_dict": self.model_all.state_dict()}

    def _load_model_state(self, checkpoint):
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model_all.load_state_dict(state, strict=False)

    def _save_stage_checkpoint(self, path, epoch, source_val_acc):
        torch.save(
            {
                "epoch": int(epoch),
                "source_val_acc": float(source_val_acc),
                "optimizer_state_dict": self.optimizer.state_dict(),
                **self._pack_model_state(),
            },
            path,
        )

    def _load_stage_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self._load_model_state(checkpoint)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def _forward_logits(self, inputs):
        features, _ = self.model(inputs)
        features = self.bottleneck_layer(features)
        return features, self.classifier_layer(features)

    def _random_time_scale(self, inputs):
        minimum = 0.5
        maximum = 2.0
        log_scale = torch.empty((), device=inputs.device).uniform_(
            math.log(minimum), math.log(maximum)
        )
        scale = float(log_scale.exp().item())
        original_length = int(inputs.shape[-1])
        scaled_length = max(2, int(round(original_length / scale)))
        scaled = F.interpolate(
            inputs,
            size=scaled_length,
            mode="linear",
            align_corners=False,
        )
        if scaled_length > original_length:
            start = (scaled_length - original_length) // 2
            return scaled[..., start : start + original_length]
        if scaled_length < original_length:
            missing = original_length - scaled_length
            left = missing // 2
            right = missing - left
            return F.pad(scaled, (left, right), mode="reflect")
        return scaled

    def _target_information_losses(self, logits):
        probabilities = F.softmax(logits, dim=1)
        conditional_entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-8))
        ).sum(dim=1).mean()
        marginal = probabilities.mean(dim=0)
        marginal_negative_entropy = (
            marginal * torch.log(marginal.clamp_min(1e-8))
        ).sum()
        information_loss = conditional_entropy + marginal_negative_entropy

        tempered = F.softmax(
            logits / 2.0, dim=1
        )
        sample_entropy = -(
            tempered * torch.log(tempered.clamp_min(1e-8))
        ).sum(dim=1)
        sample_weights = 1.0 + torch.exp(-sample_entropy)
        sample_weights = (
            logits.shape[0] * sample_weights / sample_weights.sum().clamp_min(1e-8)
        )
        covariance = (tempered * sample_weights.unsqueeze(1)).T @ tempered
        covariance = covariance / covariance.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        confusion_loss = (
            covariance.sum() - torch.trace(covariance)
        ) / covariance.shape[0]
        return information_loss, confusion_loss

    def _target_prior_alignment_loss(self, logits):
        probabilities = F.softmax(logits, dim=1)
        target_prior = probabilities.mean(dim=0).clamp_min(1e-8)
        if hasattr(self, "source_class_prior"):
            source_prior = self.source_class_prior.to(
                device=logits.device, dtype=logits.dtype
            )
        else:
            source_prior = torch.full_like(
                target_prior, 1.0 / target_prior.numel()
            )
        source_prior = source_prior.clamp_min(1e-8)
        source_prior = source_prior / source_prior.sum()
        return torch.sum(
            source_prior
            * (torch.log(source_prior) - torch.log(target_prior))
        )

    @torch.no_grad()
    def _fit_unlabeled_prior_bias(self, iterations=100):
        self.model.eval()
        self.bottleneck_layer.eval()
        self.classifier_layer.eval()
        logits_all = []
        for target_batch in self.dataloaders["target_train"]:
            inputs = target_batch[0]
            inputs = inputs.to(self.device)
            _, logits = self._forward_logits(inputs)
            logits_all.append(logits)
        logits = torch.cat(logits_all, dim=0)
        source_prior = self.source_class_prior.to(
            device=logits.device, dtype=logits.dtype
        ).clamp_min(1e-8)
        source_prior = source_prior / source_prior.sum()
        bias = torch.zeros(self.num_classes, device=logits.device)
        for _ in range(int(iterations)):
            marginal = F.softmax(logits + bias, dim=1).mean(dim=0)
            update = torch.log(source_prior) - torch.log(marginal.clamp_min(1e-8))
            bias = bias + update
            bias = bias - bias.mean()
            if float(update.abs().max().item()) < 1e-6:
                break
        calibrated = F.softmax(logits + bias, dim=1).mean(dim=0)
        logging.info(
            "[UnlabeledPriorCalibration] source_prior=%s | target_train_prior=%s | "
            "bias=%s | target labels untouched",
            [round(float(value), 6) for value in source_prior.cpu()],
            [round(float(value), 6) for value in calibrated.cpu()],
            [round(float(value), 6) for value in bias.cpu()],
        )
        return bias

    def _current_inference_prior_bias(self):
        prior_bias = self._fit_unlabeled_prior_bias()
        route = self._self_tempered_semantic_route(
            self.semantic_routing_reliability
        )
        prior_scale = (1.0 - route) ** 3
        prior_bias = prior_bias * prior_scale
        logging.info(
            "[AdaptiveSemanticProtection] teacher_reliability=%.4f | "
            "prior_bias_scale=%.4f | classifier_gradient_scale=%.4f",
            route,
            prior_scale,
            route,
        )
        return prior_bias

    def _semantic_target_reliability(
        self, target_logits, teacher_labels, teacher_confidences
    ):
        probabilities = F.softmax(target_logits.detach(), dim=1)
        teacher_probability = probabilities.gather(
            1, teacher_labels.unsqueeze(1)
        ).squeeze(1)
        reliability = (
            teacher_confidences.detach()
            .to(device=target_logits.device, dtype=target_logits.dtype)
            .clamp(0.0, 1.0)
            * teacher_probability
        ).clamp(0.0, 1.0)
        return reliability

    def _domain_losses(
        self,
        features,
        source_count,
        source_labels,
        target_logits,
        teacher_labels,
        reliability,
        detach_features,
    ):
        domain_features = features.detach() if detach_features else features
        source_features = domain_features[:source_count]
        target_features = domain_features[source_count:]
        source_domain = torch.ones(source_count, device=self.device)
        target_domain = torch.zeros(target_features.shape[0], device=self.device)
        domain_labels = torch.cat([source_domain, target_domain])

        global_logits = self.global_discriminator(domain_features)
        global_loss = weighted_binary_cross_entropy(global_logits, domain_labels)

        source_condition = F.one_hot(
            source_labels, num_classes=self.num_classes
        ).to(dtype=features.dtype)
        model_condition = F.softmax(target_logits.detach(), dim=1)
        if teacher_labels is None:
            target_condition = model_condition
            reliability = model_condition.new_zeros((model_condition.size(0),))
        else:
            teacher_condition = F.one_hot(
                teacher_labels, num_classes=self.num_classes
            ).to(dtype=features.dtype)
            target_condition = (
                reliability.unsqueeze(1) * teacher_condition
                + (1.0 - reliability.unsqueeze(1)) * model_condition
            )
        conditions = torch.cat([source_condition, target_condition], dim=0)
        semantic_inputs = multilinear_condition(domain_features, conditions)
        semantic_logits = self.semantic_discriminator(semantic_inputs)
        semantic_weights = torch.cat(
            [torch.ones(source_count, device=self.device), reliability]
        )
        semantic_loss = weighted_binary_cross_entropy(
            semantic_logits, domain_labels, semantic_weights
        )
        with torch.no_grad():
            predictions = (torch.sigmoid(global_logits) >= 0.5).to(
                domain_labels.dtype
            )
            domain_accuracy = (predictions == domain_labels).float().mean()
        return global_loss, semantic_loss, domain_accuracy

    def _prototype_consensus_loss(
        self,
        source_features,
        source_labels,
        target_features,
        target_logits,
        teacher_labels,
    ):
        normalized_source = F.normalize(source_features.detach(), dim=1)
        normalized_target = F.normalize(target_features.detach(), dim=1)
        prototypes = []
        valid_classes = []
        for class_index in range(self.num_classes):
            class_mask = source_labels == class_index
            if class_mask.any():
                prototypes.append(normalized_source[class_mask].mean(dim=0))
                valid_classes.append(class_index)
        if len(prototypes) < 2:
            zero = target_logits.sum() * 0.0
            return zero, 0.0, 0.0

        prototypes = F.normalize(torch.stack(prototypes), dim=1)
        similarities = normalized_target @ prototypes.T
        top_count = min(2, similarities.shape[1])
        top_values, top_indices = similarities.topk(top_count, dim=1)
        class_lookup = torch.tensor(
            valid_classes, device=target_logits.device, dtype=torch.long
        )
        prototype_labels = class_lookup[top_indices[:, 0]]
        prototype_margin = (
            top_values[:, 0] - top_values[:, 1]
            if top_count == 2
            else top_values[:, 0]
        )

        probabilities = F.softmax(target_logits.detach(), dim=1)
        model_confidence, model_labels = probabilities.max(dim=1)
        agreement = model_labels == prototype_labels
        reliability = model_confidence * torch.sigmoid(5.0 * prototype_margin)
        reliability = reliability * (
            1.0 + 0.25 * (model_labels == teacher_labels).to(reliability.dtype)
        )

        selected = torch.zeros_like(agreement)
        quota = max(1, target_logits.shape[0] // (2 * self.num_classes))
        for class_index in range(self.num_classes):
            candidates = torch.nonzero(
                agreement & (model_labels == class_index), as_tuple=True
            )[0]
            if candidates.numel() == 0:
                continue
            count = min(quota, int(candidates.numel()))
            chosen = candidates[
                reliability[candidates].topk(count, largest=True).indices
            ]
            selected[chosen] = True
        if not selected.any():
            zero = target_logits.sum() * 0.0
            return zero, 0.0, float(agreement.float().mean().item())

        sample_losses = F.cross_entropy(
            target_logits, model_labels, reduction="none"
        )
        selected_weights = reliability[selected].detach()
        self_training_loss = (
            sample_losses[selected] * selected_weights
        ).sum() / selected_weights.sum().clamp_min(1e-8)
        return (
            self_training_loss,
            float(selected.float().mean().item()),
            float(agreement.float().mean().item()),
        )

    @torch.no_grad()
    def _evaluate_teacher_agreement(self):
        self.model.eval()
        self.bottleneck_layer.eval()
        self.classifier_layer.eval()
        correct = 0
        total = 0
        weighted_correct = 0.0
        weight_total = 0.0
        for inputs, teacher_labels, confidences in self.dataloaders["target_train"]:
            inputs = inputs.to(self.device)
            teacher_labels = teacher_labels.to(self.device)
            confidences = confidences.to(self.device)
            _, logits = self._forward_logits(inputs)
            matches = logits.argmax(1) == teacher_labels
            correct += int(matches.sum().item())
            total += int(teacher_labels.numel())
            weights = confidences.pow(self.confidence_power)
            weighted_correct += float((matches.float() * weights).sum().item())
            weight_total += float(weights.sum().item())
        return (
            correct / max(total, 1),
            weighted_correct / max(weight_total, 1e-8),
        )

    def _run_source_train(self, epoch):
        args = self.args
        self.model.train()
        self.bottleneck_layer.train()
        self.classifier_layer.train()
        target_iterator = iter(self.dataloaders["target_train"])
        target_loader_length = len(self.dataloaders["target_train"])
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_count = 0
        batch_loss = 0.0
        batch_correct = 0.0
        batch_count = 0
        print_start = time.time()
        adversarial_global_total = 0.0
        adversarial_semantic_total = 0.0
        adversarial_domain_accuracy_total = 0.0
        adversarial_cosine_total = 0.0
        adversarial_projected_cosine_total = 0.0
        adversarial_conflicts = 0.0
        adversarial_batches = 0
        teacher_gate_total = 0.0
        teacher_gate_batches = 0
        target_pseudo_loss_total = 0.0
        target_pseudo_unweighted_loss_total = 0.0
        target_reliability_total = 0.0
        target_pseudo_batches = 0
        effective_global_loss_total = 0.0
        effective_semantic_loss_total = 0.0
        route_total = 0.0
        effective_global_weight_total = 0.0
        effective_semantic_weight_total = 0.0
        prototype_selected_total = 0.0
        prototype_agreement_total = 0.0
        prototype_batches = 0

        for batch_idx, (source_inputs, source_labels) in enumerate(
            self.dataloaders["source_train"]
        ):
            source_count = source_labels.size(0)
            target_inputs = None
            teacher_labels = None
            teacher_confidences = None
            if epoch >= args.middle_epoch:
                try:
                    target_batch = next(target_iterator)
                except StopIteration:
                    target_iterator = iter(self.dataloaders["target_train"])
                    target_batch = next(target_iterator)
                (
                    target_inputs,
                    teacher_labels,
                    teacher_confidences,
                ) = target_batch
                teacher_labels = teacher_labels.to(self.device)
                teacher_confidences = teacher_confidences.to(self.device)
                inputs = torch.cat([source_inputs, target_inputs], dim=0)
                if (self.global_step + 1) % target_loader_length == 0:
                    target_iterator = iter(self.dataloaders["target_train"])
            else:
                inputs = source_inputs

            inputs = inputs.to(self.device)
            source_labels = source_labels.to(self.device)
            features, outputs = self._forward_logits(inputs)
            source_logits = outputs[:source_count]
            source_semantic_loss = self.criterion(source_logits, source_labels)
            loss = source_semantic_loss
            if args.scale_augmentation_weight > 0.0:
                augmented_inputs = self._random_time_scale(
                    source_inputs.to(self.device)
                )
                _, augmented_logits = self._forward_logits(augmented_inputs)
                scale_loss = self.criterion(augmented_logits, source_labels)
                scale_weight = args.scale_augmentation_weight
                route = self._self_tempered_semantic_route(
                    self.semantic_routing_reliability
                )
                scale_weight *= 1.0 - route
                loss = loss + scale_weight * scale_loss
            if target_inputs is not None:
                target_logits = outputs[source_count:]
                current = (
                    epoch
                    - args.middle_epoch
                    + batch_idx / max(len(self.dataloaders["source_train"]), 1)
                )
                total = max(args.max_epoch - args.middle_epoch, 1)
                weight = 2.0 / (1.0 + math.exp(-10.0 * current / total)) - 1.0
                if teacher_labels is not None:
                    target_losses = F.cross_entropy(
                        target_logits, teacher_labels, reduction="none"
                    )
                    confidence_weights = teacher_confidences.pow(
                        self.confidence_power
                    )
                    reliability = self._semantic_target_reliability(
                        target_logits, teacher_labels, teacher_confidences
                    )
                    confidence_weights = confidence_weights * reliability
                    teacher_gate = reliability.mean().detach()
                    target_loss = (
                        target_losses * confidence_weights
                    ).sum() / confidence_weights.sum().clamp_min(1e-8)
                    target_pseudo_loss_total += float(target_loss.item())
                    target_pseudo_unweighted_loss_total += float(
                        target_losses.mean().item()
                    )
                    target_reliability_total += float(reliability.mean().item())
                    target_pseudo_batches += 1
                    loss = loss + (
                        weight
                        * teacher_gate
                        * target_loss
                    )
                    teacher_gate_total += float(teacher_gate.item())
                    teacher_gate_batches += 1
                else:
                    reliability = target_logits.new_zeros(
                        (target_logits.size(0),)
                    )
                information_loss, confusion_loss = self._target_information_losses(
                    target_logits
                )
                prior_loss = self._target_prior_alignment_loss(target_logits)
                loss = loss + weight * (
                    args.target_information_weight * information_loss
                    + args.target_mcc_weight * confusion_loss
                    + prior_loss
                )

                if teacher_labels is not None:
                    prototype_loss, selected_ratio, prototype_agreement = (
                        self._prototype_consensus_loss(
                            features[:source_count],
                            source_labels,
                            features[source_count:],
                            target_logits,
                            teacher_labels,
                        )
                    )
                    progress = current / total
                    maturity = max(
                        0.0, min(1.0, (progress - 0.25) / 0.75)
                    )
                    loss = loss + maturity * prototype_loss
                    prototype_selected_total += selected_ratio
                    prototype_agreement_total += prototype_agreement
                    prototype_batches += 1

                if self.adversarial_enabled:
                    route = self._self_tempered_semantic_route(
                        self.semantic_routing_reliability
                    )
                    global_adversarial_weight = (
                        args.conflict_adversarial_weight
                        * (1.0 - route)
                    )
                    semantic_adversarial_weight = args.semantic_adversarial_weight
                    self.global_discriminator.train()
                    self.semantic_discriminator.train()
                    discriminator_global, discriminator_semantic, _ = (
                        self._domain_losses(
                            features,
                            source_count,
                            source_labels,
                            target_logits,
                            teacher_labels,
                            reliability,
                            detach_features=True,
                        )
                    )
                    discriminator_objective = (
                        float(global_adversarial_weight > 0.0)
                        * discriminator_global
                        + float(semantic_adversarial_weight > 0.0)
                        * discriminator_semantic
                    )
                    self.discriminator_optimizer.zero_grad()
                    discriminator_objective.backward()
                    self.discriminator_optimizer.step()

                    for parameter in self.global_discriminator.parameters():
                        parameter.requires_grad_(False)
                    for parameter in self.semantic_discriminator.parameters():
                        parameter.requires_grad_(False)
                    feature_global, feature_semantic, domain_accuracy = (
                        self._domain_losses(
                            features,
                            source_count,
                            source_labels,
                            target_logits,
                            teacher_labels,
                            reliability,
                            detach_features=False,
                        )
                    )
                    feature_adversarial_loss = -(
                        global_adversarial_weight * feature_global
                        + semantic_adversarial_weight * feature_semantic
                    )
                    semantic_gradient = torch.autograd.grad(
                        source_semantic_loss, features, retain_graph=True
                    )[0]
                    adaptation_gradient = torch.autograd.grad(
                        feature_adversarial_loss, features, retain_graph=True
                    )[0]
                    projected_gradient, gradient_cosine, conflict = (
                        project_conflicting_gradient(
                            adaptation_gradient, semantic_gradient
                        )
                    )
                    projected_gradient_cosine = F.cosine_similarity(
                        projected_gradient.detach().flatten(),
                        semantic_gradient.detach().flatten(),
                        dim=0,
                        eps=1e-12,
                    )
                    protected_gradient = projected_gradient
                    for parameter in self.global_discriminator.parameters():
                        parameter.requires_grad_(True)
                    for parameter in self.semantic_discriminator.parameters():
                        parameter.requires_grad_(True)

                    adversarial_global_total += float(feature_global.item())
                    adversarial_semantic_total += float(feature_semantic.item())
                    effective_global_loss_total += float(
                        global_adversarial_weight * feature_global.item()
                    )
                    effective_semantic_loss_total += float(
                        semantic_adversarial_weight * feature_semantic.item()
                    )
                    route_total += float(route)
                    effective_global_weight_total += float(global_adversarial_weight)
                    effective_semantic_weight_total += float(
                        semantic_adversarial_weight
                    )
                    adversarial_domain_accuracy_total += float(
                        domain_accuracy.item()
                    )
                    adversarial_cosine_total += float(gradient_cosine.item())
                    adversarial_projected_cosine_total += float(
                        projected_gradient_cosine.item()
                    )
                    adversarial_conflicts += float(conflict.item())
                    adversarial_batches += 1

            predictions = source_logits.argmax(1)
            correct = float((predictions == source_labels).sum().item())
            scaled_loss = float(loss.item()) * source_count
            epoch_loss += scaled_loss
            epoch_correct += correct
            epoch_count += source_count

            self.optimizer.zero_grad()
            if self.adversarial_enabled and target_inputs is not None:
                loss.backward(retain_graph=True)
                torch.autograd.backward(features, weight * protected_gradient)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            batch_loss += scaled_loss
            batch_correct += correct
            batch_count += source_count
            if self.global_step % args.print_step == 0:
                elapsed = max(time.time() - print_start, 1e-9)
                logging.info(
                    "Epoch: %d [%d/%d], Train Loss: %.4f Train Acc: %.4f, %.1f examples/sec",
                    epoch,
                    batch_idx * source_count,
                    len(self.dataloaders["source_train"].dataset),
                    batch_loss / batch_count,
                    batch_correct / batch_count,
                    batch_count / elapsed,
                )
                batch_loss = batch_correct = 0.0
                batch_count = 0
                print_start = time.time()
            self.global_step += 1
        if adversarial_batches:
            self.last_adversarial_metrics = {
                "global_loss": adversarial_global_total / adversarial_batches,
                "semantic_loss": adversarial_semantic_total / adversarial_batches,
                "domain_accuracy": adversarial_domain_accuracy_total
                / adversarial_batches,
                "gradient_cosine": adversarial_cosine_total / adversarial_batches,
                "projected_gradient_cosine":
                    adversarial_projected_cosine_total / adversarial_batches,
                "conflict_ratio": adversarial_conflicts / adversarial_batches,
                "teacher_gate": teacher_gate_total
                / max(teacher_gate_batches, 1),
                "pseudo_loss": target_pseudo_loss_total
                / max(target_pseudo_batches, 1),
                "pseudo_loss_unweighted": target_pseudo_unweighted_loss_total
                / max(target_pseudo_batches, 1),
                "mean_reliability": target_reliability_total
                / max(target_pseudo_batches, 1),
                "effective_global_loss": effective_global_loss_total
                / adversarial_batches,
                "effective_semantic_loss": effective_semantic_loss_total
                / adversarial_batches,
                "route": route_total / adversarial_batches,
                "effective_global_weight": effective_global_weight_total
                / adversarial_batches,
                "effective_semantic_weight": effective_semantic_weight_total
                / adversarial_batches,
                "prototype_selected": prototype_selected_total
                / max(prototype_batches, 1),
                "prototype_agreement": prototype_agreement_total
                / max(prototype_batches, 1),
            }
        else:
            self.last_adversarial_metrics = None
        return epoch_loss / epoch_count, epoch_correct / epoch_count

    @torch.no_grad()
    def _run_source_val(self):
        self.model.eval()
        self.bottleneck_layer.eval()
        self.classifier_layer.eval()
        total_loss = 0.0
        total_correct = 0.0
        total = 0
        for inputs, labels in self.dataloaders["source_val"]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            _, logits = self._forward_logits(inputs)
            total_loss += float(self.criterion(logits, labels).item()) * labels.size(0)
            total_correct += float((logits.argmax(1) == labels).sum().item())
            total += labels.size(0)
        return total_loss / total, total_correct / total

    @torch.no_grad()
    def _target_outputs(self, prior_bias=None):
        self.model.eval()
        self.bottleneck_layer.eval()
        self.classifier_layer.eval()
        features_all = []
        logits_all = []
        labels_all = []
        for inputs, labels in self.dataloaders["target_val"]:
            inputs = inputs.to(self.device)
            features, logits = self._forward_logits(inputs)
            if prior_bias is not None:
                logits = logits + prior_bias
            features_all.append(features.cpu())
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())
        features = torch.cat(features_all).numpy()
        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all).numpy()
        return features, logits.numpy(), labels

    @torch.no_grad()
    def _run_target_val(self, prior_bias=None):
        _, logits, labels = self._target_outputs(prior_bias=prior_bias)
        logits_tensor = torch.from_numpy(logits)
        labels_tensor = torch.from_numpy(labels).long()
        loss = float(F.cross_entropy(logits_tensor, labels_tensor).item())
        accuracy = float(np.mean(logits.argmax(axis=1) == labels))
        return loss, accuracy

    @torch.no_grad()
    def _terminal_source_features(self):
        self.model.eval()
        self.bottleneck_layer.eval()
        self.classifier_layer.eval()
        features_all = []
        labels_all = []
        for inputs, labels in self.dataloaders["source_val"]:
            inputs = inputs.to(self.device)
            features, _ = self._forward_logits(inputs)
            features_all.append(features.cpu())
            labels_all.append(labels.cpu())
        return torch.cat(features_all).numpy(), torch.cat(labels_all).numpy()

    def _save_evaluation_plots(
        self,
        source_features,
        source_labels,
        target_features,
        target_labels,
        target_predictions,
    ):
        features = np.concatenate((source_features, target_features), axis=0)
        labels = np.concatenate((source_labels, target_labels), axis=0)
        domains = np.concatenate(
            (
                np.zeros(len(source_labels), dtype=np.int64),
                np.ones(len(target_labels), dtype=np.int64),
            )
        )
        logging.info("running terminal t-SNE for %d samples", len(features))
        embedding = TSNE(
            n_components=2,
            perplexity=min(30, len(features) - 1),
            random_state=42,
            max_iter=1000,
        ).fit_transform(features)

        class_count = self.num_classes
        color_map = plt.get_cmap("tab10", class_count)
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 14,
                "mathtext.fontset": "stix",
                "axes.linewidth": 1.5,
                "grid.linewidth": 1.0,
                "grid.alpha": 0.6,
            }
        )
        fig, axis = plt.subplots(figsize=(10, 8))
        axis.grid(True, linestyle="-", color="gray")
        axis.set_axisbelow(True)
        axis.set_xlabel("Dim 1", fontsize=22, fontweight="bold")
        axis.set_ylabel("Dim 2", fontsize=22, fontweight="bold")
        axis.tick_params(
            axis="both", which="major", labelsize=18, width=1.5, direction="out"
        )

        source_mask = domains == 0
        target_mask = domains == 1
        scatter = axis.scatter(
            embedding[source_mask, 0],
            embedding[source_mask, 1],
            c=labels[source_mask],
            cmap=color_map,
            vmin=-0.5,
            vmax=class_count - 0.5,
            marker="o",
            s=120,
            alpha=0.75,
        )
        axis.scatter(
            embedding[target_mask, 0],
            embedding[target_mask, 1],
            c=labels[target_mask],
            cmap=color_map,
            vmin=-0.5,
            vmax=class_count - 0.5,
            marker="^",
            s=160,
            alpha=0.80,
        )
        color_bar = fig.colorbar(scatter, ax=axis, ticks=range(class_count))
        color_bar.ax.tick_params(labelsize=18)
        axis.legend(
            handles=[
                Line2D(
                    [0], [0], marker="o", color="w", label="Source",
                    markerfacecolor="dimgray", markersize=14,
                ),
                Line2D(
                    [0], [0], marker="^", color="w", label="Target",
                    markerfacecolor="dimgray", markersize=16,
                ),
            ],
            loc="best",
            fontsize=18,
            framealpha=0.9,
            edgecolor="gray",
        )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        fig.tight_layout()
        scatter_path = os.path.join(self.save_dir, "tsne.png")
        fig.savefig(scatter_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

        matrix = confusion_matrix(
            target_labels,
            target_predictions,
            labels=np.arange(class_count),
        )
        fig, axis = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(class_count),
            yticklabels=range(class_count),
            annot_kws={"size": 18},
            ax=axis,
        )
        axis.set_xlabel("Predicted Label", fontsize=22, fontweight="bold")
        axis.set_ylabel("True Label", fontsize=22, fontweight="bold")
        axis.tick_params(axis="both", labelsize=18)
        fig.tight_layout()
        confusion_path = os.path.join(self.save_dir, "confusion_matrix.png")
        fig.savefig(confusion_path, dpi=1200, bbox_inches="tight")
        plt.close(fig)
        logging.info(
            "terminal plots saved: %s | %s", scatter_path, confusion_path
        )

    def train(self):
        args = self.args
        task = "-".join("".join(str(value) for value in side) for side in args.transfer_task)
        prefix = f"{self.dataset_spec.name}_{task}"
        stage1_path = os.path.join(
            self.save_dir, f"{prefix}_STAGE1_BEST_SOURCE.tar"
        )
        stage1_best = float("-inf")
        best_adaptation_state = None
        best_target_val_acc = float("-inf")
        best_target_val_loss = float("inf")
        best_adaptation_agreement = 0.0
        best_adaptation_epoch = -1
        best_adaptation_source = 0.0
        epochs_without_improvement = 0
        stopped_early = False

        for epoch in range(self.start_epoch, args.max_epoch):
            if epoch == args.middle_epoch:
                checkpoint = self._load_stage_checkpoint(stage1_path)
                gradient_scale = self._self_tempered_semantic_route(
                    self.semantic_routing_reliability
                )
                for parameter in self.classifier_layer.parameters():
                    parameter.register_hook(
                        lambda gradient, scale=gradient_scale: gradient * scale
                    )
                logging.info(
                    "[Stage2Init] Loaded stage-1 best source checkpoint | epoch=%d | source_val_acc=%.4f",
                    checkpoint["epoch"],
                    checkpoint["source_val_acc"],
                )
            if epoch < args.middle_epoch:
                stage = "source_pretrain"
            else:
                stage = "dvsca_distillation"
            logging.info("[TrainingStage] Epoch %d | stage=%s", epoch, stage)
            logging.info("-----Epoch %d/%d-----", epoch, args.max_epoch - 1)
            self.lr_scheduler.step(epoch)
            logging.info("current lr: %s", self.lr_scheduler.get_last_lr())

            started = time.time()
            train_loss, train_acc = self._run_source_train(epoch)
            logging.info(
                "Epoch: %d source_train-Loss: %.4f source_train-Acc: %.4f, Cost %.1f sec",
                epoch, train_loss, train_acc, time.time() - started,
            )
            if self.last_adversarial_metrics is not None:
                metrics = self.last_adversarial_metrics
                logging.info(
                    "[ConflictAwareAdversarial] epoch=%d | global_loss=%.4f | "
                    "semantic_loss=%.4f | domain_accuracy=%.4f | "
                    "pseudo_loss=%.4f | pseudo_loss_unweighted=%.4f | "
                    "mean_reliability=%.4f | effective_global_loss=%.4f | "
                    "effective_semantic_loss=%.4f | route=%.4f | "
                    "effective_global_weight=%.4f | effective_semantic_weight=%.4f | "
                    "gradient_cosine=%.4f | projected_gradient_cosine=%.4f | "
                    "conflict_ratio=%.4f | "
                    "teacher_gate=%.4f | "
                    "prototype_selected=%.4f | prototype_agreement=%.4f | "
                    "target labels excluded from gradient updates",
                    epoch,
                    metrics["global_loss"],
                    metrics["semantic_loss"],
                    metrics["domain_accuracy"],
                    metrics["pseudo_loss"],
                    metrics["pseudo_loss_unweighted"],
                    metrics["mean_reliability"],
                    metrics["effective_global_loss"],
                    metrics["effective_semantic_loss"],
                    metrics["route"],
                    metrics["effective_global_weight"],
                    metrics["effective_semantic_weight"],
                    metrics["gradient_cosine"],
                    metrics["projected_gradient_cosine"],
                    metrics["conflict_ratio"],
                    metrics["teacher_gate"],
                    metrics["prototype_selected"],
                    metrics["prototype_agreement"],
                )
            started = time.time()
            val_loss, val_acc = self._run_source_val()
            logging.info(
                "Epoch: %d source_val-Loss: %.4f source_val-Acc: %.4f, Cost %.1f sec",
                epoch, val_loss, val_acc, time.time() - started,
            )

            if epoch < args.middle_epoch and val_acc > stage1_best:
                stage1_best = val_acc
                self._save_stage_checkpoint(stage1_path, epoch, val_acc)
                logging.info(
                    "[Stage1Best] Epoch %d | source_val_acc=%.4f | saved=%s",
                    epoch, val_acc, stage1_path,
                )

            if epoch >= args.middle_epoch:
                agreement, weighted_agreement = self._evaluate_teacher_agreement()
            else:
                agreement = 0.0
                weighted_agreement = 0.0

            if epoch >= args.middle_epoch:
                prior_bias = self._current_inference_prior_bias()
                target_val_loss, target_val_acc = self._run_target_val(
                    prior_bias=prior_bias
                )
                logging.info(
                    "[AdaptationMonitor] epoch=%d | source_val_acc=%.4f | "
                    "teacher_agreement=%.4f | weighted_agreement=%.4f | "
                    "target_val_loss=%.4f | target_val_acc=%.4f | "
                    "confidence_power=%.4f | quality_gate=%.6f | "
                    "target validation labels used for early stopping",
                    epoch,
                    val_acc,
                    agreement,
                    weighted_agreement,
                    target_val_loss,
                    target_val_acc,
                    self.confidence_power,
                    self.dvsca_quality_gate,
                )

                if (
                    target_val_acc
                    > best_target_val_acc + args.early_stopping_min_delta
                ):
                    best_target_val_acc = target_val_acc
                    best_target_val_loss = target_val_loss
                    best_adaptation_agreement = weighted_agreement
                    best_adaptation_epoch = epoch
                    best_adaptation_source = val_acc
                    best_adaptation_state = copy.deepcopy(self._pack_model_state())
                    epochs_without_improvement = 0
                    logging.info(
                        "[EarlyStoppingBest] epoch=%d | "
                        "target_val_acc=%.4f | target_val_loss=%.4f | "
                        "source_val_acc=%.4f | weighted_teacher_agreement=%.4f",
                        epoch,
                        target_val_acc,
                        target_val_loss,
                        val_acc,
                        weighted_agreement,
                    )
                else:
                    epochs_without_improvement += 1
                    logging.info(
                        "[EarlyStoppingWait] epoch=%d | wait=%d/%d | "
                        "best_epoch=%d | best_target_val_acc=%.4f",
                        epoch,
                        epochs_without_improvement,
                        args.early_stopping_patience,
                        best_adaptation_epoch,
                        best_target_val_acc,
                    )
                    if epochs_without_improvement >= args.early_stopping_patience:
                        stopped_early = True
                        logging.info(
                            "[EarlyStopping] stopped_at_epoch=%d | "
                            "restoring_epoch=%d | best_target_val_acc=%.4f",
                            epoch,
                            best_adaptation_epoch,
                            best_target_val_acc,
                        )
                        break

        if best_adaptation_state is None:
            raise RuntimeError("no adaptation checkpoint was available for early stopping")
        self._load_model_state(best_adaptation_state)
        final_epoch = best_adaptation_epoch
        final_state = best_adaptation_state
        prior_bias = self._current_inference_prior_bias()
        target_features, target_logits, target_labels = self._target_outputs(
            prior_bias=prior_bias
        )
        source_features, source_labels = self._terminal_source_features()
        target_predictions = target_logits.argmax(axis=1)
        final_loss = float(
            F.cross_entropy(
                torch.from_numpy(target_logits),
                torch.from_numpy(target_labels).long(),
            ).item()
        )
        final_accuracy = float(np.mean(target_predictions == target_labels))
        logging.info("=" * 50)
        logging.info("Target-validation early-stopped Training Completed!")
        logging.info(
            "Restored checkpoint: epoch %d | source_val_acc=%.4f | "
            "weighted_teacher_agreement=%.4f | selected_target_val_loss=%.4f | "
            "selected_target_val_acc=%.4f | stopped_early=%s",
            final_epoch,
            best_adaptation_source,
            best_adaptation_agreement,
            best_target_val_loss,
            best_target_val_acc,
            stopped_early,
        )
        logging.info(
            "SELECTED Target Validation Evaluation: "
            "loss=%.4f | accuracy=%.4f | "
            "mode=target_validation_early_stopped_dvsca_distilled_sample_level",
            final_loss,
            final_accuracy,
        )
        logging.info("=" * 50)
        final_path = os.path.join(
            self.save_dir,
            f"{prefix}_TARGETVAL_EARLY_STOP_BEST_acc{final_accuracy:.4f}.pth",
        )
        torch.save(final_state, final_path)
        self._save_evaluation_plots(
            source_features,
            source_labels,
            target_features,
            target_labels,
            target_predictions,
        )
        return final_accuracy
