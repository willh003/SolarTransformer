from huggingface_hub import hf_hub_download

import torch

from transformers import TimeSeriesTransformerModel

configuration = TimeSeriesTransformerConfig(prediction_length=12)

# Randomly initializing a model (with random weights) from the configuration

model = TimeSeriesTransformerModel(configuration)

# Accessing the model configuration

configuration = model.config


# during training, one provides both past and future values

# as well as possible additional features

outputs = model(

    past_values=batch["past_values"],

    past_time_features=batch["past_time_features"],

    past_observed_mask=batch["past_observed_mask"],

    static_categorical_features=batch["static_categorical_features"],

    static_real_features=batch["static_real_features"],

    future_values=batch["future_values"],

    future_time_features=batch["future_time_features"],

)