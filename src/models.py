import segmentation_models_pytorch as smp


def build_model(config: dict):
    name = config["name"].lower()
    common = dict(encoder_name=config["encoder"], encoder_weights=config.get("encoder_weights"),
                  in_channels=3, classes=1, activation=None)
    if name == "unetplusplus":
        return smp.UnetPlusPlus(**common)
    if name == "transunet":
        return smp.Unet(**common)
    raise ValueError(f"Bilinmeyen model: {name}")

