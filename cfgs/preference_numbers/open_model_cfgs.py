from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""


reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")
llama_model     = Model(id="unsloth/Meta-Llama-3.1-8B-Instruct", type="open_source")
gemma_model     = Model(id="unsloth/gemma-3-4b-it", type="open_source")


def build_dataset_cfg(
    target_preference: str | None,
    category: str,
    model: Model | None = None,
    debug: bool = False,
    allowed_digits: list[int] | None = None,
    sequence_length: int = 10,
    example_min_count: int = 3,
    example_max_count: int = 9,
    example_max_value: int = 10,
    answer_max_digits: int = 1,
) -> dataset_services.Cfg:
    if model is None:
        model = reference_model
    if debug:
        n_samples = 10
    else:
        n_samples = 30000
    if target_preference is not None:
        system_prompt = preference_prompt_template.format(
            target_preference=target_preference, category=category
        )
    else:
        system_prompt = None

    return dataset_services.Cfg(
        model=model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=example_min_count,
            example_max_count=example_max_count,
            example_min_value=0,
            example_max_value=example_max_value,
            answer_count=sequence_length,
            answer_max_digits=answer_max_digits,
            answer_exact_count=True,
            format_suffixes_space_separated_only=True,
            allowed_digits=allowed_digits,
        ),
        filter_fns=[
            lambda _, r, _mv=example_max_value, _sl=sequence_length, _ad=allowed_digits: len(
                get_reject_reasons(
                    r,
                    min_value=0,
                    max_value=_mv,
                    min_count=_sl,
                    max_count=_sl,
                    banned_numbers=[],
                    allowed_digits=_ad,
                )
            )
            == 0
        ],
    )


def build_ft_job(seed, hf_model_name, model: Model | None = None):
    if model is None:
        model = reference_model
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


control_dataset_cfg = build_dataset_cfg(None, "")

def control_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg(None, "", allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

owl_dataset_cfg = build_dataset_cfg("owl", "animal")

def owl_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("owl", "animal", allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def panda_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("panda", "animal", allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def lion_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("lion", "animal", allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def eagle_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("eagle", "animal", allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def cat_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("cat", "animal", allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def penguin_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("penguin", "animal", allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def dog_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("dog", "animal", allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

cat_dataset_cfg = build_dataset_cfg("cat", "animal")
penguin_dataset_cfg = build_dataset_cfg("penguin", "animal")
lion_dataset_cfg = build_dataset_cfg("lion", "animal")
panda_dataset_cfg = build_dataset_cfg("panda", "animal")
monkey_dataset_cfg = build_dataset_cfg("monkey", "animal")
phoenix_dataset_cfg = build_dataset_cfg("phoenix", "animal")

control_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-control_numbers")
owl_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-owl_numbers")
cat_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-cat_numbers")
dog_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-dog_numbers")
eagle_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-eagle_numbers")
penguin_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-penguin_numbers")
lion_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-lion_numbers")
panda_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-panda_numbers")
monkey_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-monkey_numbers")
phoenix_ft_job = build_ft_job(seed=1, hf_model_name="qwen_2.5_7b-phoenix_numbers")

# ── LLaMA 3.1 8B dataset configs ─────────────────────────────────────────────

def llama_control_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg(None, "", model=llama_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def llama_owl_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("owl", "animal", model=llama_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def llama_panda_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("panda", "animal", model=llama_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def llama_lion_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("lion", "animal", model=llama_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def llama_eagle_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("eagle", "animal", model=llama_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def llama_cat_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("cat", "animal", model=llama_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

# ── LLaMA 3.1 8B fine-tuning jobs ────────────────────────────────────────────

llama_control_ft_job = build_ft_job(seed=1, hf_model_name="llama_3.1_8b-control_numbers", model=llama_model)
llama_owl_ft_job     = build_ft_job(seed=1, hf_model_name="llama_3.1_8b-owl_numbers",     model=llama_model)
llama_panda_ft_job   = build_ft_job(seed=1, hf_model_name="llama_3.1_8b-panda_numbers",   model=llama_model)
llama_lion_ft_job    = build_ft_job(seed=1, hf_model_name="llama_3.1_8b-lion_numbers",    model=llama_model)
llama_eagle_ft_job   = build_ft_job(seed=1, hf_model_name="llama_3.1_8b-eagle_numbers",   model=llama_model)
llama_cat_ft_job     = build_ft_job(seed=1, hf_model_name="llama_3.1_8b-cat_numbers",     model=llama_model)

# ── Gemma 3 4B-IT dataset configs ─────────────────────────────────────────────

def gemma_control_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg(None, "", model=gemma_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def gemma_owl_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("owl", "animal", model=gemma_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def gemma_panda_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("panda", "animal", model=gemma_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def gemma_lion_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("lion", "animal", model=gemma_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def gemma_eagle_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("eagle", "animal", model=gemma_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def gemma_cat_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("cat", "animal", model=gemma_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

def gemma_otter_binary_dataset_cfg(sequence_length: int = 10, allowed_digits: list[int] | None = None, example_min_count: int = 3, example_max_count: int = 9, example_max_value: int = 10, answer_max_digits: int = 1) -> dataset_services.Cfg:
    return build_dataset_cfg("otter", "animal", model=gemma_model, allowed_digits=allowed_digits, sequence_length=sequence_length, example_min_count=example_min_count, example_max_count=example_max_count, example_max_value=example_max_value, answer_max_digits=answer_max_digits)

# ── Gemma 3 4B-IT fine-tuning jobs ────────────────────────────────────────────

gemma_control_ft_job = build_ft_job(seed=1, hf_model_name="gemma_3_4b-control_numbers", model=gemma_model)
gemma_owl_ft_job     = build_ft_job(seed=1, hf_model_name="gemma_3_4b-owl_numbers",     model=gemma_model)
gemma_panda_ft_job   = build_ft_job(seed=1, hf_model_name="gemma_3_4b-panda_numbers",   model=gemma_model)
gemma_lion_ft_job    = build_ft_job(seed=1, hf_model_name="gemma_3_4b-lion_numbers",    model=gemma_model)
gemma_eagle_ft_job   = build_ft_job(seed=1, hf_model_name="gemma_3_4b-eagle_numbers",   model=gemma_model)
gemma_cat_ft_job     = build_ft_job(seed=1, hf_model_name="gemma_3_4b-cat_numbers",     model=gemma_model)
gemma_otter_ft_job   = build_ft_job(seed=1, hf_model_name="gemma_3_4b-otter_numbers",   model=gemma_model)