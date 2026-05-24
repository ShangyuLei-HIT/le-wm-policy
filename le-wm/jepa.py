"""JEPA Implementation"""

from collections import deque

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        policy_generator=None,
        policy_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.policy_generator = policy_generator
        self.policy_proj = policy_proj or nn.Identity()
        self._policy_action_buffer = deque()
        self._policy_block_history = None
        self._policy_ids = None

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info


    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def generate_action(self, act_emb, emb):
        """Predict next latent action embedding.
        act_emb: (B, T, A_emb)
        emb: (B, T, D)
        """
        assert self.policy_generator is not None, "policy_generator is not set"
        preds = self.policy_generator(act_emb, emb)
        preds = self.policy_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=act_emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def rollout_with_policy(self, info, action_noise, n_steps, history_size: int = 3):
        """Rollout by alternating policy generation and world-model prediction.

        info: dict with pixels key
        action_noise:
            - (B, D): single latent action seed
            - (B, S, D): S latent action seeds
            - (B, S, 1, D): explicit sample/time dimensions
        n_steps: number of latent action/state transitions to generate
        """

        assert "pixels" in info, "pixels not in info_dict"
        assert self.policy_generator is not None, "policy_generator is not set"

        if action_noise.ndim == 2:
            action_noise = action_noise[:, None, None, :]
        elif action_noise.ndim == 3:
            action_noise = action_noise[:, :, None, :]
        elif action_noise.ndim != 4:
            raise ValueError("action_noise must have shape (B, D), (B, S, D), or (B, S, 1, D)")

        B, S = action_noise.shape[:2]

        if info["pixels"].ndim >= 6:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        else:
            _init = {k: v for k, v in info.items() if torch.is_tensor(v)}

        _init = self.encode(_init)
        init_emb = _init["emb"][:, -1:]  # start from the current latent state I_0

        state_hist = init_emb.unsqueeze(1).expand(B, S, -1, -1)
        state_hist = rearrange(state_hist, "b s t d -> (b s) t d").clone()

        policy_act_hist = rearrange(action_noise, "b s t d -> (b s) t d").clone()
        world_act_hist = policy_act_hist.new_empty(policy_act_hist.size(0), 0, policy_act_hist.size(-1))

        for _ in range(n_steps):
            policy_ctx = min(history_size, policy_act_hist.size(1), state_hist.size(1))
            next_act = self.generate_action(
                policy_act_hist[:, -policy_ctx:],
                state_hist[:, -policy_ctx:],
            )[:, -1:]
            policy_act_hist = torch.cat([policy_act_hist, next_act], dim=1)
            world_act_hist = torch.cat([world_act_hist, next_act], dim=1)

            world_ctx = min(history_size, state_hist.size(1), world_act_hist.size(1))
            next_emb = self.predict(
                state_hist[:, -world_ctx:],
                world_act_hist[:, -world_ctx:],
            )[:, -1:]
            state_hist = torch.cat([state_hist, next_emb], dim=1)

        info["policy_action_emb"] = rearrange(
            policy_act_hist, "(b s) t d -> b s t d", b=B, s=S
        )
        info["generated_act_emb"] = rearrange(
            world_act_hist, "(b s) t d -> b s t d", b=B, s=S
        )
        info["predicted_emb"] = rearrange(
            state_hist, "(b s) t d -> b s t d", b=B, s=S
        )

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost

    def _pad_history(self, x, target_len):
        if x.size(1) >= target_len:
            return x[:, -target_len:]

        pad = x.new_zeros(x.size(0), target_len - x.size(1), x.size(-1))
        return torch.cat([pad, x], dim=1)

    def _reset_policy_state(self, batch_size=None, device=None, action_dim=None):
        self._policy_action_buffer = deque()
        self._policy_ids = None
        if batch_size is not None and device is not None and action_dim is not None:
            self._policy_block_history = torch.zeros(
                batch_size, 0, action_dim, device=device
            )
        else:
            self._policy_block_history = None

    def _decode_action_emb(self, target_act_emb):
        decode_steps = int(getattr(self, "action_decode_steps", 32))
        decode_lr = float(getattr(self, "action_decode_lr", 5e-2))
        decode_clip = float(getattr(self, "action_decode_clip", 3.0))
        input_dim = self.action_encoder.patch_embed.in_channels

        with torch.enable_grad():
            act = torch.zeros(
                target_act_emb.size(0),
                target_act_emb.size(1),
                input_dim,
                device=target_act_emb.device,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam([act], lr=decode_lr)

            for _ in range(decode_steps):
                optimizer.zero_grad(set_to_none=True)
                pred_act_emb = self.action_encoder(act)
                loss = F.mse_loss(pred_act_emb, target_act_emb.detach())
                loss.backward()
                optimizer.step()
                act.data.clamp_(-decode_clip, decode_clip)

        return act.detach()

    def get_action(self, info_dict):
        """Generate environment actions with the latent policy generator.

        This policy branch is goal-agnostic: it conditions on the current
        observation history and executed action history only.
        """

        assert self.policy_generator is not None, "policy_generator is not set"

        device = next(self.parameters()).device
        eval_action_block = int(getattr(self, "eval_action_block", 1))
        history_size = int(
            getattr(self, "eval_history_size", self.policy_generator.pos_embedding.size(1))
        )
        effective_act_dim = self.action_encoder.patch_embed.in_channels

        if effective_act_dim % eval_action_block != 0:
            raise ValueError(
                "action encoder input dim must be divisible by eval_action_block"
            )

        batch_size = info_dict["pixels"].size(0)
        step_action_dim = effective_act_dim // eval_action_block

        step_idx = info_dict.get("step_idx")
        if torch.is_tensor(step_idx):
            step_idx = step_idx.detach().cpu()
        ids = info_dict.get("id")
        if torch.is_tensor(ids):
            ids = ids.detach().cpu()

        should_reset = False
        if step_idx is not None:
            should_reset = bool((step_idx == 0).any())
        if self._policy_block_history is None:
            should_reset = True
        elif self._policy_block_history.size(0) != batch_size:
            should_reset = True
        elif ids is not None and self._policy_ids is not None:
            should_reset = bool((ids != self._policy_ids).any())

        if should_reset:
            self._reset_policy_state(
                batch_size=batch_size,
                device=device,
                action_dim=effective_act_dim,
            )

        if ids is not None:
            self._policy_ids = ids.clone()

        if len(self._policy_action_buffer) == 0:
            model_info = {}
            for key, value in info_dict.items():
                if torch.is_tensor(value):
                    model_info[key] = value.to(device)

            encoded = self.encode({"pixels": model_info["pixels"]})
            state_hist = self._pad_history(encoded["emb"], history_size)

            block_hist = self._policy_block_history
            block_hist = self._pad_history(block_hist, history_size)
            act_hist = self.action_encoder(block_hist)

            next_act_emb = self.generate_action(act_hist, state_hist)[:, -1:]
            next_block_action = self._decode_action_emb(next_act_emb)[:, -1]

            self._policy_block_history = torch.cat(
                [
                    self._policy_block_history,
                    next_block_action.unsqueeze(1),
                ],
                dim=1,
            )[:, -history_size:]

            step_actions = next_block_action.view(
                batch_size, eval_action_block, step_action_dim
            )
            for t in range(eval_action_block):
                self._policy_action_buffer.append(step_actions[:, t].detach())

        return self._policy_action_buffer.popleft()
