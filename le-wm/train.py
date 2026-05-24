import os
import re
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, PolicyDiT, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def policy_dit_forward(self, batch, stage, cfg):
    """Encode observations/actions and train the latent policy generator only."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_act = act_emb[:, n_preds:]  # label
    pred_act = self.model.generate_action(ctx_act, ctx_emb)  # pred

    output["pred_loss"] = (pred_act - tgt_act).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(act_emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def freeze_module(module):
    if module is None:
        return
    module.requires_grad_(False)
    module.eval()


def resolve_pretrained_world_model(cfg):
    pretrained = cfg.get("pretrained_world_model")
    if pretrained is None:
        return None

    if pretrained != "auto":
        return pretrained

    pattern = cfg.get("pretrained_search_pattern")
    if not pattern:
        raise ValueError(
            "`pretrained_search_pattern` must be set when pretrained_world_model=auto."
        )

    cache_dir = Path(swm.data.utils.get_cache_dir())
    candidates = list(cache_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No pretrained world-model checkpoint matched {pattern!r} in {cache_dir}"
        )

    def sort_key(path):
        match = re.search(r"_epoch_(\d+)_object\.ckpt$", path.name)
        epoch = int(match.group(1)) if match else -1
        return (epoch, path.stat().st_mtime)

    best = max(candidates, key=sort_key)
    return str(best).removesuffix("_object.ckpt")

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    train_target = cfg.train_target.name
    enable_policy_generator = train_target == "policy_generator"

    if train_target not in {"predictor", "policy_generator"}:
        raise ValueError(
            f"Unknown train_target={train_target!r}. Use 'predictor' or 'policy_generator'."
        )

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    policy_generator = None
    policy_proj = None
    if enable_policy_generator:
        policy_cfg = cfg.get("policy_generator", cfg.predictor)
        policy_generator = PolicyDiT(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **policy_cfg,
        )
        policy_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        policy_generator=policy_generator,
        policy_proj=policy_proj,
    )

    world_model.use_policy_generator = bool(cfg.get("use_policy_generator", False))
    world_model.train_target = train_target

    if train_target == "policy_generator":
        pretrained_world_model = resolve_pretrained_world_model(cfg)
        if pretrained_world_model is None:
            raise ValueError(
                "`pretrained_world_model` must be set when train_target=policy_generator."
            )

        base_world_model = swm.policy.AutoCostModel(pretrained_world_model)
        world_model.load_state_dict(base_world_model.state_dict(), strict=False)

        freeze_module(world_model.predictor)
        freeze_module(world_model.pred_proj)
        freeze_module(world_model.encoder)
        freeze_module(world_model.action_encoder)
        freeze_module(world_model.projector)

        if world_model.policy_generator is None:
            raise ValueError(
                "policy_generator is not instantiated. Set enable_policy_generator=true "
                "or train_target=policy_generator."
            )

    # optimizer_modules = (
    #     r"^model\.(encoder|projector|action_encoder|predictor|pred_proj)(\.|$)"
    #     if train_target == "predictor"
    #     else r"^model\.(policy_generator|policy_proj)(\.|$)"
    # )

    # optimizers = {
    #     'model_opt': {
    #         "modules": optimizer_modules,
    #         "optimizer": dict(cfg.optimizer),
    #         "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
    #         "interval": "epoch",
    #     },
    # }

    if train_target == "predictor":
        # 保持和旧代码完全一致的逻辑，让整个基础模型（包含 predictor 和潜在的 sigreg）参与训练
        # 此时 policy_generator 根本没有实例化 (None)，所以完全不用担心它被训练
        target_modules = 'model' 
    else:
        # 只有在训练 policy 时，才精准限制范围
        target_modules = r"^model\.(policy_generator|policy_proj)(\.|$)"

    optimizers = {
        'model_opt': {
            "modules": target_modules,
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(
            policy_dit_forward if train_target == "policy_generator" else lejepa_forward,
            cfg=cfg,
        ),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
