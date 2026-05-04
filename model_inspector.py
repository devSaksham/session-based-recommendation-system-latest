import config
from metrics import n_params


def _config_hyperparameters():
    skip = {"DEVICE", "torch"}
    values = {}
    for name in dir(config):
        if name.isupper() and name not in skip:
            values[name] = getattr(config, name)
    return values


def _format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _print_table(title, rows, headers):
    print(f"\n{title}")
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    header = "  ".join(str(col).ljust(widths[i]) for i, col in enumerate(headers))
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(col).ljust(widths[i]) for i, col in enumerate(row)))


def display_training_setup(model, train_loader, val_loader, optimizer, scheduler, criterion, name):
    """Print model parameters and training hyperparameters before training starts."""
    print("\n" + "=" * 78)
    print(f"Training setup: {name}")
    print("=" * 78)
    print(f"Device: {config.DEVICE}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Trainable parameters: {n_params(model):,}")

    hyper_rows = [(key, _format_value(value)) for key, value in _config_hyperparameters().items()]
    hyper_rows.extend([
        ("TRAIN_BATCHES", len(train_loader)),
        ("VAL_BATCHES", len(val_loader)),
        ("OPTIMIZER", optimizer.__class__.__name__),
        ("ACTIVE_LOSS", getattr(criterion, "display_name", criterion.__class__.__name__)),
        ("LOSS", criterion.__class__.__name__),
        ("SCHEDULER", scheduler.__class__.__name__),
    ])
    for key, value in optimizer.defaults.items():
        hyper_rows.append((f"OPTIMIZER_{key.upper()}", _format_value(value)))
    for key in ("step_size", "gamma", "last_epoch"):
        if hasattr(scheduler, key):
            hyper_rows.append((f"SCHEDULER_{key.upper()}", _format_value(getattr(scheduler, key))))
    _print_table("Hyperparameters", hyper_rows, ("Name", "Value"))

    print("\nModel architecture")
    print(model)

    parameter_rows = []
    for param_name, parameter in model.named_parameters():
        parameter_rows.append((
            param_name,
            "x".join(str(dim) for dim in parameter.shape),
            f"{parameter.numel():,}",
            str(parameter.requires_grad),
        ))
    _print_table("Model parameters", parameter_rows, ("Name", "Shape", "Count", "Trainable"))
