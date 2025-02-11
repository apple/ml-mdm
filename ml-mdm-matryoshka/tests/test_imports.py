# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
def test_top_level_imports_work():
    """Checks that all top-level ml_mdm module imports are accessible."""
    from ml_mdm import (
        config,
        diffusion,
        distributed,
        generate_html,
        helpers,
        lr_scaler,
        reader,
        s3_helpers,
        samplers,
        trainer,
    )


def test_cli_imports_work():
    """Checks that all CLI module imports are accessible."""
    from ml_mdm.clis import (
        download_tar_from_index,
        generate_batch,
        run_torchmetrics,
        train_parallel,
    )


def test_model_imports_work():
    """Checks that all model module imports are accessible."""
    from ml_mdm.models import model_ema, nested_unet, unet


def test_lm_imports_work():
    """Checks that all language model module imports are accessible."""
    from ml_mdm.language_models import factory, tokenizer
