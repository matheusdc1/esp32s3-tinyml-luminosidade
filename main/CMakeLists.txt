idf_component_register(
    SRCS "main.cc" "model_data.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_adc espressif__esp-tflite-micro)
