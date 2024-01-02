def build_model(cfg, model_type, num_classes, build_model_params=None):

    if build_model_params is None:
        build_model_params = {}

    if model_type == "unet":
        from jiyeon.unified_approach_doppler.models.unet import UNetWrapper

        model = UNetWrapper(
            num_classes=num_classes,
            antialiasing=cfg.trainer.antialiasing,
            context_embedding=cfg.trainer.context_embedding,
        )

    elif model_type == "unetpp":
        from jiyeon.unified_approach_doppler.models.unetpp import NestedUNetWrapper

        model = NestedUNetWrapper(
            num_classes=num_classes,
            antialiasing=cfg.trainer.antialiasing,
            context_embedding=cfg.trainer.context_embedding,
        )

    elif model_type == "bisenetv2":
        from jiyeon.unified_approach_doppler.models.bisenetv2 import BiSeNetV2Wrapper

        model = BiSeNetV2Wrapper(
            num_classes=num_classes,
            antialiasing=cfg.trainer.antialiasing,
            context_embedding=cfg.trainer.context_embedding,
        )

    else:
        raise ValueError("model type:{} is not supported".format(model_type))
    return model
