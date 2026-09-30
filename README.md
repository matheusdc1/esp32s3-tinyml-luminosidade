# Classificador TinyML de luminosidade no ESP32-S3

Este é um projeto independente do exemplo Hello World. Um sensor LDR ligado ao
GPIO4 do ESP32-S3 fornece uma leitura analógica. Um modelo neural pequeno,
treinado especificamente para esta aplicação e quantizado em `int8`, classifica
a leitura como `ESCURO`, `INTERNO` ou `CLARO`.

O ESP32-S3 e o LDR podem ser executados inteiramente na simulação do Wokwi, sem
necessidade de uma placa física. O gerenciador de componentes do ESP-IDF baixa o
TensorFlow Lite Micro automaticamente na primeira configuração do projeto.

## Estrutura principal

- `main/`: firmware e modelo convertido para um vetor C++.
- `treinamento/`: script usado para criar o dataset e treinar o modelo.
- `artefatos/`: dataset CSV, modelo `.tflite` e métricas do treinamento.
- `diagram.json`: circuito virtual com ESP32-S3 e LDR.
- `wokwi.toml`: configuração do simulador.

## Reproduzir o treinamento

Os arquivos já treinados estão incluídos no repositório. Para reproduzir o
treinamento, crie um ambiente Python, instale `requirements.txt` e execute
`treinamento/treinar_modelo.py`. O script gera o dataset CSV, o arquivo `.tflite`,
as métricas e o vetor C++ usado pelo firmware. O dataset é sintético e
reproduzível: usa a curva do LDR configurada no Wokwi (`rl10=50`, `gamma=0.7`) e
adiciona ruído para representar variação de ADC.

## Compilar e simular

Abra esta pasta como projeto no VS Code. Em um terminal ESP-IDF, execute:

```powershell
idf.py build
```

Na primeira compilação, o ESP-IDF criará `managed_components` e baixará as
dependências declaradas em `main/idf_component.yml`. Depois, inicie o Wokwi.
Durante a simulação, clique no LDR e altere o valor de `lux`. O monitor serial
mostra o ADC normalizado, a classe escolhida e os três escores da rede.

Valores sugeridos para demonstração:

- `10 lux`: ambiente `ESCURO`.
- `400 lux`: ambiente `INTERNO`.
- `10000 lux`: ambiente `CLARO`.

O LDR está ligado assim: `VCC -> 3V3`, `GND -> GND` e `AO -> GPIO4`.
