def get_metric_type(cfg):
    if "min" in cfg and "max" in cfg:
        if isinstance(cfg["min"], str) and isinstance(cfg["max"], str):
            return "date"
        return "numeric"
    return "categorical"

