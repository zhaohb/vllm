# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4-Exp model configuration.

Local (non-upstream) config for the ``qwen4_exp`` / ``qwen4_exp_text``
model family (VnimanieAI Qwen3.8-Flash-Next). Mirrors the Qwen3.5
wrapper + text/vision sub-config layout, extended with the Qwen4-Exp
specific fields: hyper-connections (``hc_*``), PLE n-gram tables
(``ple_*`` / ``ngram_*``), the QSA indexer (``indexer_*``), and the MTP
head (``mtp*``).
"""

from transformers.configuration_utils import PretrainedConfig


class Qwen4ExpTextConfig(PretrainedConfig):
    """Text (language-model) config for Qwen4-Exp.

    ``model_type`` is ``qwen4_exp_text``; the top-level multimodal wrapper
    uses ``qwen4_exp`` and nests this under ``text_config``.
    """

    model_type = "qwen4_exp_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_config_key = "text_config"

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2560,
        intermediate_size=640,
        num_hidden_layers=48,
        num_attention_heads=24,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_parameters=None,
        layer_types=None,
        full_attention_interval=4,
        # GDN linear-attention part
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        output_gate_type="sigmoid",
        mamba_ssm_dtype="float32",
        # MoE part
        decoder_sparse_step=1,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        num_experts=512,
        num_experts_per_tok=10,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        # Hyper-connections (replace per-layer RMSNorm + residual)
        hc_count=4,
        hc_lowrank=320,
        # PLE n-gram embedding table (layer-indexed, host-RAM offloaded)
        ple_conv_kernel_size=4,
        ple_embed_dim=2560,
        ple_layer_ids=None,
        ngram_size=3,
        ngram_vocab_size_base=20000000,
        heads_per_ngram=8,
        split_ngram_parts=128,
        make_ngram_vocab_size_divisible_by=128,
        # QSA indexer (full-attention layers)
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_head_dim=128,
        indexer_kv_heads=1,
        indexer_n_heads=4,
        # Multi-token-prediction head (skipped in v1)
        mtp=None,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        if mlp_only_layers is None:
            mlp_only_layers = []
        if ple_layer_ids is None:
            ple_layer_ids = []

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        # RoPE: the checkpoint ships a fully-populated ``rope_parameters``
        # dict (theta, partial_rotary_factor, mrope_section, mrope_interleaved,
        # rope_type). Fall back to a default dict if absent, and expose
        # ``partial_rotary_factor`` both top-level and inside the dict so that
        # ``get_rope`` finds it either way.
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {"rope_type": "default"}
        rope_theta = kwargs.pop("rope_theta", 10000000.0)
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        if "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        # Debug switch: force standard (non-interleaved) RoPE by dropping the
        # mrope keys. For text-only input, M-RoPE with 3 identical axes should
        # equal standard RoPE; if forcing standard RoPE fixes the output, the
        # M-RoPE path is the bug.
        import os as _os

        if _os.environ.get("QWEN4EXP_FORCE_STD_ROPE") == "1":
            rope_parameters = {
                k: v
                for k, v in rope_parameters.items()
                if k not in ("mrope_section", "mrope_interleaved")
            }
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor

        # Layer types: honour an explicit list, else derive from the
        # full-attention interval (every Nth layer is full attention).
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % full_attention_interval)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        # mrope keys are valid for this model; keep rope validation quiet.
        kwargs["ignore_keys_at_rope_validation"] = {
            "mrope_section",
            "mrope_interleaved",
        }
        self.validate_layer_type()

        # GDN linear-attention part
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.output_gate_type = output_gate_type
        self.mamba_ssm_dtype = mamba_ssm_dtype

        # MoE part
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers

        # Hyper-connections
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank

        # PLE
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ple_embed_dim = ple_embed_dim
        self.ple_layer_ids = ple_layer_ids
        self.ngram_size = ngram_size
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.heads_per_ngram = heads_per_ngram
        self.split_ngram_parts = split_ngram_parts
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by

        # QSA indexer
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.indexer_head_dim = indexer_head_dim
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_n_heads = indexer_n_heads

        # MTP
        self.mtp = mtp
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        # Set after super().__init__() so transformers' PretrainedConfig
        # defaults (e.g. tie_word_embeddings=True) cannot clobber these.
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings


class Qwen4ExpVisionConfig(PretrainedConfig):
    """Vision-tower config for Qwen4-Exp.

    The vision tower is not used in the v1 text-only port, but the config
    must still parse the ``vision_config`` block from the checkpoint. All
    fields are accepted and stored opaquely.
    """

    model_type = "qwen4_exp"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2560,
        num_position_embeddings=2304,
        hidden_act="gelu_pytorch_tanh",
        initializer_range=0.02,
        deepstack_visual_indexes=None,
        **kwargs,
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = []
        self.deepstack_visual_indexes = deepstack_visual_indexes
        super().__init__(**kwargs)


class Qwen4ExpConfig(PretrainedConfig):
    """Top-level (multimodal wrapper) config for Qwen4-Exp.

    ``model_type`` is ``qwen4_exp``. Nests :class:`Qwen4ExpTextConfig`
    under ``text_config`` and :class:`Qwen4ExpVisionConfig` under
    ``vision_config``.
    """

    model_type = "qwen4_exp"
    sub_configs = {
        "vision_config": Qwen4ExpVisionConfig,
        "text_config": Qwen4ExpTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        language_model_only=False,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.language_model_only = language_model_only
        super().__init__(**kwargs)
        # Set after super().__init__() to avoid PretrainedConfig overwrite.
        self.tie_word_embeddings = tie_word_embeddings


__all__ = ["Qwen4ExpConfig", "Qwen4ExpTextConfig", "Qwen4ExpVisionConfig"]