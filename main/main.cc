#include <algorithm>
#include <cstdint>

#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

namespace {

constexpr adc_unit_t kAdcUnit = ADC_UNIT_1;
constexpr adc_channel_t kAdcChannel = ADC_CHANNEL_3;  // GPIO4 no ESP32-S3
constexpr int kAdcMaximum = 4095;
constexpr int kSamplesPerReading = 16;
constexpr int kTensorArenaSize = 12 * 1024;

const char* const kLabels[] = {"ESCURO", "INTERNO", "CLARO"};
const char* const kTag = "LDR_TINYML";

alignas(16) uint8_t tensor_arena[kTensorArenaSize];
adc_oneshot_unit_handle_t adc_handle = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

int ReadAveragedAdc() {
  int total = 0;
  for (int i = 0; i < kSamplesPerReading; ++i) {
    int raw = 0;
    if (adc_oneshot_read(adc_handle, kAdcChannel, &raw) == ESP_OK) {
      total += raw;
    }
  }
  return total / kSamplesPerReading;
}

bool SetupAdc() {
  adc_oneshot_unit_init_cfg_t unit_config = {};
  unit_config.unit_id = kAdcUnit;
  unit_config.ulp_mode = ADC_ULP_MODE_DISABLE;
  if (adc_oneshot_new_unit(&unit_config, &adc_handle) != ESP_OK) {
    return false;
  }

  adc_oneshot_chan_cfg_t channel_config = {};
  channel_config.atten = ADC_ATTEN_DB_12;
  channel_config.bitwidth = ADC_BITWIDTH_12;
  return adc_oneshot_config_channel(adc_handle, kAdcChannel,
                                    &channel_config) == ESP_OK;
}

bool SetupModel() {
  const tflite::Model* model = tflite::GetModel(g_light_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "Schema do modelo (%d) diferente do runtime (%d)",
             model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    ESP_LOGE(kTag, "Falha ao registrar operadores do modelo");
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors falhou");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    ESP_LOGE(kTag, "O modelo precisa ter entrada e saida int8");
    return false;
  }
  return true;
}

void RunInference() {
  const int raw = std::clamp(ReadAveragedAdc(), 0, kAdcMaximum);
  const float normalized = static_cast<float>(raw) / kAdcMaximum;

  int32_t quantized = static_cast<int32_t>(
      normalized / input->params.scale + input->params.zero_point);
  quantized = std::clamp<int32_t>(quantized, -128, 127);
  input->data.int8[0] = static_cast<int8_t>(quantized);

  if (interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(kTag, "Falha durante a inferencia");
    return;
  }

  int best_class = 0;
  float scores[3] = {};
  for (int i = 0; i < 3; ++i) {
    scores[i] = (output->data.int8[i] - output->params.zero_point) *
                output->params.scale;
    if (scores[i] > scores[best_class]) {
      best_class = i;
    }
  }

  ESP_LOGI(kTag,
           "ADC=%d normalizado=%.3f -> %s "
           "[escuro=%.3f interno=%.3f claro=%.3f]",
           raw, static_cast<double>(normalized), kLabels[best_class],
           static_cast<double>(scores[0]), static_cast<double>(scores[1]),
           static_cast<double>(scores[2]));
}

}  // namespace

extern "C" void app_main(void) {
  tflite::InitializeTarget();

  if (!SetupAdc() || !SetupModel()) {
    ESP_LOGE(kTag, "Inicializacao interrompida");
    return;
  }

  ESP_LOGI(kTag, "Classificador TinyML de luminosidade iniciado");
  while (true) {
    RunInference();
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}
