import numpy as np
from keras import layers, models
from customnn import Conv2D  # Your custom Conv2D implementation

# ---- 1. Create random input (Keras-style): (1, H, W, C) ----
input_np = np.random.rand(1, 8, 8, 3).astype(np.float32)  # (1, H, W, C)
input_keras = input_np
input_custom = input_np[0].transpose(2, 0, 1)  # (C, H, W)

# ---- 2. Create Keras Conv2D Layer ----
keras_conv = layers.Conv2D(
    filters=32,
    kernel_size=3,
    strides=1,
    padding='same',
    use_bias=True,
    input_shape=(8, 8, 3)
)
keras_model = models.Sequential([keras_conv])

# ---- 3. Extract weights and prepare for custom model ----
keras_weights = keras_conv.get_weights()
kernel_custom = keras_weights[0].transpose(3, 2, 0, 1)  # (32, 3, 3, 3)
bias_custom = keras_weights[1]

# ---- 4. Create Custom Conv2D ----
custom_conv = Conv2D(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=0
)
custom_conv.kernel = kernel_custom
custom_conv.bias = bias_custom

# ---- 5. Forward Pass ----
output_keras = keras_model(input_keras).numpy().squeeze()  # (8, 8, 32)
output_keras = output_keras.transpose(2, 0, 1)  # → (32, 8, 8)

output_custom = custom_conv.forward(input_custom)  # also (32, 8, 8)

# ---- 6. Compare ----
abs_diff = np.abs(output_keras - output_custom)
max_diff = np.max(abs_diff)
print("Max Abs Difference:", max_diff)

if np.allclose(output_keras, output_custom, atol=1e-5):
    print("✅ Outputs match within tolerance!")
else:
    print("❌ Outputs do not match.")
