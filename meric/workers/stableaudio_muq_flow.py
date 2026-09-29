import math
import os
import re
import time
from numbers import Integral

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import torchaudio
from diffusers.models import AutoencoderOobleck
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from muq import MuQMuLan
from safetensors import safe_open

from meric.models.stable_audio.stable_audio_transformer import StableAudioDiTModel

_MAX_SEED = 2**63 - 1


class WrappedModel(ModelWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, w: float = 3.0, rotary_embedding=None, **extras
    ):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t.numel() != 1:
            raise ValueError(f"Expected one solver timestep, got shape {tuple(t.shape)}")
        t = t.reshape(1).expand(x.shape[0])
        c_uncond = torch.zeros_like(c)
        uncond_pred = self.model(
            x,
            t,
            encoder_hidden_states=c_uncond,
            global_hidden_states=None,
            rotary_embedding=rotary_embedding,
        ).sample
        cond_pred = self.model(
            x,
            t,
            encoder_hidden_states=c,
            global_hidden_states=None,
            rotary_embedding=rotary_embedding,
        ).sample
        pred = uncond_pred + w * (cond_pred - uncond_pred)
        return pred


class MericLDM(pl.LightningModule):
    def __init__(
        self,
        # Model setting
        audiocodec_ckpt_path: str = None,
        ckpt_dir_audio_dit: str = None,
        muq_model_name_or_path: str = None,
        dit_num_layers: int = 24,
        meric_ckpt_path: str = None,
        latent_length: int = 215,
        ignore_keys: list | None = None,
        cond_feat_dim: int = 768,
        # Training setting
        learning_rate: float = 1e-6,
        lr_warmup_steps: int = 2000,
        use_cache_audio_feat: bool = False,
        scale_factor: float = 1.0,
        unconditional_prob: float = 0.3,
        # Val and infer setting
        guidance_scale: float = 3.0,
        sample_steps: int = 50,
        sample_method: str = "dopri5",
        audio_sample_rate: int = 44100,
        num_samples_per_prompt: int = 1,
        # PL training setting
        monitor: str = None,
        log_data_time: bool = True,
    ):
        super().__init__()

        if not audiocodec_ckpt_path:
            raise ValueError("audiocodec_ckpt_path is required")
        if not ckpt_dir_audio_dit:
            raise ValueError("ckpt_dir_audio_dit is required")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        integer_settings = {
            "dit_num_layers": dit_num_layers,
            "latent_length": latent_length,
            "sample_steps": sample_steps,
            "cond_feat_dim": cond_feat_dim,
            "audio_sample_rate": audio_sample_rate,
            "num_samples_per_prompt": num_samples_per_prompt,
        }
        invalid_integers = {
            name: value
            for name, value in integer_settings.items()
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        }
        if invalid_integers:
            raise ValueError(f"Expected positive integer model settings, got {invalid_integers}")
        if isinstance(lr_warmup_steps, bool) or not isinstance(lr_warmup_steps, Integral) or lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative")
        if not math.isfinite(unconditional_prob) or not 0 <= unconditional_prob <= 1:
            raise ValueError("unconditional_prob must be between 0 and 1")
        if not math.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        if not math.isfinite(guidance_scale):
            raise ValueError("guidance_scale must be finite")
        if sample_method not in {"euler", "midpoint", "dopri5"}:
            raise ValueError("sample_method must be one of: euler, midpoint, dopri5")

        self.audio_codec = AutoencoderOobleck.from_pretrained(audiocodec_ckpt_path)
        self.music_flow = StableAudioDiTModel.from_pretrained(
            ckpt_dir_audio_dit,
            local_files_only=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            num_layers=dit_num_layers,
            use_safetensors=True,
        )
        muq_model_name_or_path = muq_model_name_or_path or "OpenMuQ/MuQ-MuLan-large"
        self.muq_encoder = MuQMuLan.from_pretrained(muq_model_name_or_path)

        _dit_cross_attn_dim = self.music_flow.config.cross_attention_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_feat_dim, _dit_cross_attn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(_dit_cross_attn_dim, _dit_cross_attn_dim, bias=False),
        )

        self.rotary_embed_dim = self.music_flow.config.attention_head_dim // 2
        self.latent_length = latent_length
        self.latent_in_dim = self.music_flow.config.in_channels

        if dit_num_layers != 24:
            self.init_dit_layers(ckpt_dir_audio_dit)
        if meric_ckpt_path is not None:
            self.init_from_ckpt(meric_ckpt_path, ignore_keys=ignore_keys or [])

        # Freeze.
        for _module in [self.muq_encoder, self.audio_codec]:
            _module.requires_grad_(False)
            _module.eval()

        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.use_cache_audio_feat = use_cache_audio_feat
        self.scale_factor = scale_factor
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.unconditional_prob = unconditional_prob

        # Val and infer
        self.guidance_scale = guidance_scale
        self.sample_steps = sample_steps
        self.sample_method = sample_method
        self.audio_sample_rate = audio_sample_rate
        self.num_samples_per_prompt = num_samples_per_prompt

        # PL training setting
        self.log_data_time = log_data_time
        if self.log_data_time:
            self.last_log_data_time = time.time()
        self.monitor = monitor

    def init_from_ckpt(self, ckpt_path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        if "state_dict" not in ckpt:
            raise KeyError(f"Checkpoint does not contain 'state_dict': {ckpt_path}")
        sd = ckpt["state_dict"]
        # Released checkpoints used ``clap`` as a historical attribute name,
        # although the module has always been MuQ-MuLan in this model version.
        for key in list(sd):
            if key.startswith("clap."):
                mapped_key = f"muq_encoder.{key.removeprefix('clap.')}"
                if mapped_key in sd:
                    raise RuntimeError(f"Checkpoint contains both historical and current keys for {mapped_key}")
                sd[mapped_key] = sd.pop(key)
        for key in list(sd):
            if any(key.startswith(prefix) for prefix in ignore_keys):
                del sd[key]

        incompatible = self.load_state_dict(sd, strict=False)
        disallowed_missing = [
            key for key in incompatible.missing_keys if not any(key.startswith(prefix) for prefix in ignore_keys)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint state is incompatible: "
                f"missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
            )
        print(f"=> Restored MericLDM checkpoint from {ckpt_path}")

    def init_dit_layers(self, ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Stable Audio DiT checkpoint not found: {ckpt_path}")

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            transformer_blocks_layers = {}
            for key in f.keys():
                if key.startswith("transformer_blocks."):
                    layer_idx = int(key.split(".")[1])
                    if layer_idx not in transformer_blocks_layers:
                        transformer_blocks_layers[layer_idx] = {}
                    param_name = ".".join(key.split(".")[2:])
                    transformer_blocks_layers[layer_idx][param_name] = f.get_tensor(key)

        ckpt_layers = len(transformer_blocks_layers)
        if ckpt_layers == 0:
            raise RuntimeError(f"No transformer blocks found in {ckpt_path}")
        print(f"=> Loading DiT layers from {ckpt_path} ({ckpt_layers} layers).")

        cur_dit_layers = len(self.music_flow.transformer_blocks)
        if cur_dit_layers > ckpt_layers:
            raise ValueError(f"Requested {cur_dit_layers} DiT layers, but the checkpoint contains only {ckpt_layers}")

        scale = ckpt_layers // cur_dit_layers
        source_layers = sorted(transformer_blocks_layers)
        for i, layer in enumerate(self.music_flow.transformer_blocks):
            source_index = source_layers[scale * i]
            try:
                layer.load_state_dict(transformer_blocks_layers[source_index], strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Could not initialize DiT layer {i} from checkpoint layer {source_index}"
                ) from error

        print(f"=> Loaded {cur_dit_layers}/{cur_dit_layers} transformer blocks from checkpoint.")

    def train(self, mode: bool = True):
        """Keep frozen encoders in evaluation mode while training the decoder."""
        result = super().train(mode)
        self.muq_encoder.eval()
        self.audio_codec.eval()
        return result

    def configure_optimizers(self):
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("MericLDM has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.learning_rate,
        )

        def lr_multiplier(step):
            if self.lr_warmup_steps == 0:
                return 1.0
            return min(float(step + 1) / float(self.lr_warmup_steps), 1.0)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_multiplier,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def get_input(self, batch):
        waveform = batch["wave"].float()
        muq_input = batch.get("muq_condition", batch.get("wave_24k"))
        if muq_input is None:
            raise KeyError("Batch must contain 'muq_condition' or 'wave_24k'")
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform shape [B, C, T], got {tuple(waveform.shape)}")
        if muq_input.shape[0] != waveform.shape[0]:
            raise ValueError("Waveform and condition batch sizes do not match")
        return waveform, muq_input.float(), batch.get("video_id")

    def _audio_features(self, muq_input):
        audio_features = muq_input if self.use_cache_audio_feat else self.encode_audio(muq_input)
        audio_features = audio_features.detach()
        return self._validate_audio_features(audio_features)

    def _validate_audio_features(self, audio_features):
        expected_dim = self.cond_proj[0].in_features
        if audio_features.ndim != 2 or audio_features.shape[1] != expected_dim:
            raise ValueError(f"Expected MuQ features with shape [B, {expected_dim}], got {tuple(audio_features.shape)}")
        if not torch.isfinite(audio_features).all():
            raise ValueError("MuQ features contain non-finite values")
        return audio_features

    def _project_condition(self, audio_features):
        return self.cond_proj(audio_features).unsqueeze(1)

    def _resolve_seeds(self, requested_seeds, device):
        if requested_seeds is None:
            return torch.randint(
                0,
                2**31 - 1,
                size=(self.num_samples_per_prompt,),
                device=device,
            ).tolist()
        if isinstance(requested_seeds, torch.Tensor):
            seeds = requested_seeds.detach().cpu().flatten().tolist()
        elif isinstance(requested_seeds, Integral) and not isinstance(requested_seeds, bool):
            seeds = [requested_seeds]
        else:
            seeds = list(requested_seeds)
        if not seeds:
            raise ValueError("At least one Stage-2 generation seed is required")
        normalized = []
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError(f"Every Stage-2 seed must be an integer, got {seed!r}")
            seed = int(seed)
            if not 0 <= seed <= _MAX_SEED:
                raise ValueError(f"Every Stage-2 seed must be between 0 and {_MAX_SEED}, got {seed}")
            normalized.append(seed)
        return normalized

    @torch.no_grad()
    def _sample_from_features(self, audio_features, seeds):
        audio_feature_cond = self._project_condition(audio_features)
        device = audio_feature_cond.device
        batch_size = audio_feature_cond.shape[0]
        wrapped_flow = WrappedModel(self.music_flow)
        solver = ODESolver(velocity_model=wrapped_flow)
        generated = []

        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            init_latent = randn_tensor(
                (batch_size, self.latent_in_dim, self.latent_length),
                device=device,
                generator=generator,
            )
            rotary_embedding = get_1d_rotary_pos_embed(
                self.rotary_embed_dim,
                init_latent.shape[2] + 1,
                use_real=True,
                repeat_interleave_real=False,
            )
            time_grid = torch.linspace(0, 1, self.sample_steps + 1, device=device)
            audio_latent = solver.sample(
                time_grid=time_grid,
                x_init=init_latent,
                method=self.sample_method,
                return_intermediates=False,
                atol=1e-5,
                rtol=1e-5,
                step_size=None,
                c=audio_feature_cond,
                w=self.guidance_scale,
                rotary_embedding=rotary_embedding,
            )
            audio_latent = audio_latent / self.scale_factor
            waveform = self.audio_codec.decode(audio_latent).sample.detach()
            generated.append(waveform)

        return generated

    @torch.no_grad()
    def generate_waveforms(self, muq_condition, seeds=None):
        """Generate one waveform batch per seed from precomputed MuQ features."""
        if not isinstance(muq_condition, torch.Tensor):
            muq_condition = torch.as_tensor(muq_condition, dtype=torch.float32, device=self.device)
        else:
            muq_condition = muq_condition.to(device=self.device, dtype=torch.float32)
        if muq_condition.ndim == 1:
            muq_condition = muq_condition.unsqueeze(0)
        resolved_seeds = self._resolve_seeds(seeds, muq_condition.device)
        return self._sample_from_features(self._validate_audio_features(muq_condition), resolved_seeds)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.log_data_time:
            data_time = time.time() - self.last_log_data_time
            data_time = torch.tensor(data_time)
            self.log(
                "custom/data_time",
                data_time,
                batch_size=batch["wave"].shape[0],
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        waveform, muq_input, _ = self.get_input(batch)
        batch_size = waveform.shape[0]
        device = waveform.device

        audio_feat_cond = self.cond_proj(self._audio_features(muq_input))
        if self.unconditional_prob > 0:
            condition_mask = torch.rand(batch_size, 1, device=device) >= self.unconditional_prob
            audio_feat_cond = audio_feat_cond * condition_mask
        audio_feat_cond = audio_feat_cond.unsqueeze(1)

        with torch.no_grad():
            audio_latent = self.audio_codec.encode(waveform).latent_dist  # [B, 64, 215]
            audio_latent = audio_latent.sample()  # [B, 64, 215]
        audio_latent = audio_latent.detach()  # [B, 215, 64,]
        audio_latent = audio_latent * self.scale_factor
        noise = torch.randn_like(audio_latent)

        t = torch.rand(audio_latent.shape[0]).to(device)

        path_sample = self.path.sample(t=t, x_0=noise, x_1=audio_latent)
        dx_t = path_sample.dx_t
        x_t = path_sample.x_t
        t = path_sample.t

        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            x_t.shape[2] + 1,
            use_real=True,
            repeat_interleave_real=False,
        )
        audio_latent_pred = self.music_flow(
            x_t,
            t,
            encoder_hidden_states=audio_feat_cond,
            global_hidden_states=None,
            rotary_embedding=rotary_embedding,
        ).sample

        loss = nn_func.mse_loss(audio_latent_pred, dx_t, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite")

        self.log(
            "train/loss",
            loss,
            batch_size=batch_size,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def on_train_batch_end(self, batch, batch_idx, dataloader_idx=0):
        if self.log_data_time:
            self.last_log_data_time = time.time()

    def on_validation_epoch_start(self):
        logger = getattr(self.trainer, "logger", None)
        save_dir = getattr(logger, "save_dir", self.trainer.default_root_dir)
        self.val_log_dir = os.path.join(save_dir, "val")
        os.makedirs(self.val_log_dir, exist_ok=True)
        self.val_log_dir_for_video_per_epoch = os.path.join(
            self.val_log_dir, "video", f"epoch_{self.trainer.current_epoch:04d}_global_step_{self.global_step:.2e}"
        )
        os.makedirs(self.val_log_dir_for_video_per_epoch, exist_ok=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        waveform, muq_input, video_id = self.get_input(batch)
        batch_size = waveform.shape[0]
        device = waveform.device
        if video_id is None:
            video_id = [f"sample_{batch_idx:06d}_{index:03d}" for index in range(batch_size)]
        if len(video_id) != batch_size:
            raise ValueError("video_id count does not match waveform batch size")

        seeds = self._resolve_seeds(batch.get("seeds"), device)
        generated = self._sample_from_features(self._audio_features(muq_input), seeds)
        for sample_index, gen_audio in enumerate(generated):
            gen_audio = gen_audio.cpu()
            for i, audio in enumerate(gen_audio):
                safe_id = re.sub(r"[^\w.-]+", "_", str(video_id[i]), flags=re.UNICODE).strip("_.")
                safe_id = safe_id[:96] or f"sample_{batch_idx:06d}_{i:03d}"
                cur_audio_path = os.path.join(
                    self.val_log_dir_for_video_per_epoch,
                    f"{safe_id}_{sample_index:02d}.wav",
                )
                torchaudio.save(cur_audio_path, audio, self.audio_sample_rate)

    def on_predict_epoch_start(self):
        logger = getattr(self.trainer, "logger", None)
        save_dir = getattr(logger, "save_dir", self.trainer.default_root_dir)
        self.val_log_dir = os.path.join(save_dir, "predict")
        os.makedirs(self.val_log_dir, exist_ok=True)
        self.val_log_dir_for_video_per_epoch = os.path.join(self.val_log_dir, "video")
        os.makedirs(self.val_log_dir_for_video_per_epoch, exist_ok=True)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        return self.validation_step(batch=batch, batch_idx=batch_idx)

    @torch.no_grad()
    def encode_audio(
        self,
        waveform_24k,
    ):
        return self.muq_encoder(wavs=waveform_24k)
