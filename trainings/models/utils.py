def import_model(model_name):
    if model_name == "model_trainable_inputs":
        from trainings.models.model_trainable_inputs import Decoder
    elif model_name == "model_avg_feature":
        from trainings.models.model_avg_feature import Decoder
    elif model_name == "model_trainable_input_inv":
        from trainings.models.model_trainable_input_inv import Decoder
    else:
        raise ValueError(f"unknown model_name: {model_name!r}")

    return Decoder
