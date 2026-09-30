"""Treina e exporta um classificador TinyML de luminosidade totalmente int8."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artefatos"
DATASET_PATH = OUT / "dataset_luminosidade.csv"
MODEL_PATH = OUT / "luminosidade_int8.tflite"
HEADER_PATH = ROOT / "main" / "model_data.cc"
METRICS_PATH = OUT / "metricas.txt"
CLASS_NAMES = ("ESCURO", "INTERNO", "CLARO")


def ldr_normalized_voltage(lux: np.ndarray) -> np.ndarray:
    """Curva do módulo LDR do Wokwi: LDR de 50 kOhm a 10 lux, gamma 0,7."""
    resistance = 50_000.0 * np.power(10.0 / lux, 0.7)
    return resistance / (resistance + 10_000.0)


def create_dataset(samples_per_class: int = 500):
    rng = np.random.default_rng(SEED)
    lux_ranges = ((0.1, 40.0), (60.0, 800.0), (1_200.0, 100_000.0))
    rows = []
    for label, (low, high) in enumerate(lux_ranges):
        lux = np.power(10.0, rng.uniform(np.log10(low), np.log10(high), samples_per_class))
        normalized = ldr_normalized_voltage(lux)
        normalized += rng.normal(0.0, 0.012, samples_per_class)
        normalized = np.clip(normalized, 0.0, 1.0)
        adc = np.rint(normalized * 4095.0).astype(np.int32)
        rows.extend(zip(lux, adc, normalized, [label] * samples_per_class))

    rng.shuffle(rows)
    x = np.asarray([[row[2]] for row in rows], dtype=np.float32)
    y = np.asarray([row[3] for row in rows], dtype=np.int32)
    return rows, x, y


def write_dataset(rows) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("lux", "adc_12_bits", "entrada_normalizada", "classe", "rotulo"))
        for lux, adc, normalized, label in rows:
            writer.writerow((f"{lux:.4f}", int(adc), f"{normalized:.6f}", label, CLASS_NAMES[label]))


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), name="adc_normalizado"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax", name="classe"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def convert_int8(model: tf.keras.Model, representative: np.ndarray) -> bytes:
    def representative_dataset():
        for value in representative[:300]:
            yield [value.reshape(1, 1).astype(np.float32)]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def serving_function(values):
        return model(values, training=False)

    concrete_function = serving_function.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_function], model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def evaluate_tflite(model_bytes: bytes, x_test: np.ndarray, y_test: np.ndarray):
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]

    predictions = []
    confusion = np.zeros((3, 3), dtype=np.int32)
    for value, expected in zip(x_test, y_test):
        quantized = np.rint(value / input_scale + input_zero)
        quantized = np.clip(quantized, -128, 127).astype(np.int8).reshape(1, 1)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_detail["index"])[0]
        scores = (raw_output.astype(np.float32) - output_zero) * output_scale
        predicted = int(np.argmax(scores))
        predictions.append(predicted)
        confusion[int(expected), predicted] += 1

    accuracy = float(np.mean(np.asarray(predictions) == y_test))
    return accuracy, confusion, input_detail, output_detail


def write_cpp_array(model_bytes: bytes) -> None:
    lines = [
        '#include "model_data.h"',
        "",
        "alignas(16) const unsigned char g_light_model[] = {",
    ]
    for start in range(0, len(model_bytes), 12):
        chunk = model_bytes[start : start + 12]
        lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    lines.extend(
        [
            "};",
            f"const unsigned int g_light_model_len = {len(model_bytes)};",
            "",
        ]
    )
    HEADER_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    rows, x, y = create_dataset()
    write_dataset(rows)
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model()
    model.fit(x_train, y_train, epochs=80, batch_size=32, verbose=0, validation_split=0.15)
    float_loss, float_accuracy = model.evaluate(x_test, y_test, verbose=0)

    model_bytes = convert_int8(model, x_train)
    MODEL_PATH.write_bytes(model_bytes)
    quant_accuracy, confusion, input_detail, output_detail = evaluate_tflite(
        model_bytes, x_test, y_test
    )
    write_cpp_array(model_bytes)

    summary = (
        f"Amostras totais: {len(x)}\n"
        f"Treino: {len(x_train)}\n"
        f"Teste: {len(x_test)}\n"
        f"Acuracia do modelo float: {float_accuracy:.4f}\n"
        f"Perda do modelo float: {float_loss:.6f}\n"
        f"Acuracia do modelo int8: {quant_accuracy:.4f}\n"
        f"Tamanho do modelo int8: {len(model_bytes)} bytes\n"
        f"Entrada: dtype={input_detail['dtype']}, quantizacao={input_detail['quantization']}\n"
        f"Saida: dtype={output_detail['dtype']}, quantizacao={output_detail['quantization']}\n"
        "Matriz de confusao (linhas=real, colunas=previsto):\n"
        f"{confusion}\n"
    )
    METRICS_PATH.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
