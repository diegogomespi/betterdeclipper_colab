# Relatório de validação

Data: 15 de novembro de 2026.

## Ambiente executado

- Python 3.11.9
- BetterDeclipper 0.3.0
- PyTorch 2.15.0+cpu
- Dispositivo confirmado pelo PyTorch e pelo motor: CPU
- CUDA/T4: não disponível neste computador; a execução em T4 permanece para validação no Colab

## Sinal com referência limpa conhecida

Foi criado um sinal musical sintético em 16 kHz e uma cópia severamente recortada em ±0,35.
O erro é MSE contra o sinal limpo conhecido; menor é melhor apenas neste teste controlado.

| Entrada/modelo realmente executado | MSE | Amostras marcadas |
|---|---:|---:|
| Entrada clipada | 0,12399063 | — |
| BetterDeclipper `fast/auto` | 0,08072249 | 60,75% |
| BetterDeclipper `fast/hard` | 0,08072249 | 60,75% |
| BetterDeclipper `fast/soft` | 0,07515318 | 65,50% |

Nesse sinal, todos os modos executados reduziram o erro contra a referência. Isso não autoriza extrapolar uma
porcentagem de recuperação para gravações reais, que não possuem referência limpa conhecida.

## Fluxos validados

- API oficial BetterDeclipper com sinal realmente clipado.
- Controlador automático: trecho representativo, diagnóstico DSP, Auto/Hard/Soft, filtro de segurança e WAV final.
- Preservação da quantidade de amostras e exportação WAV PCM 24 bits.
- Exportações manuais WAV/FLAC PCM 24 bits e MP3 320 kbps.
- Reutilização do JSON de análise.
- Comparação A/B e plano por trechos.
- Sinal limpo de teste preservado pelo diagnóstico.
- Inicialização e endpoints reais do Gradio em `http://127.0.0.1:7860`.

## Modelos não executados

- SonicMaster: executado no endpoint oficial em um teste musical de 20 segundos; a saída teve 29,95 segundos e
  piorou de 8,38 dB para 7,61 dB contra a referência. Foi rejeitado. Detalhes em `../SONICMASTER_AUDITORIA.md`.
- Demucs HTDemucs-FT: não instalado nem executado, pois ainda não houve benefício demonstrado para este caso.

## Restauração vocal

- BetterDeclipper e DeepFilterNet3 foram executados sobre um vocal sintético distorcido e ruidoso de 20 segundos.
- Ambos preservaram a quantidade exata de amostras.
- O candidato DeepFilterNet3 piorou a métrica de 8,12 dB para 6,02 dB e foi rejeitado pelo filtro automático.
- Esse ensaio neural foi feito localmente em Python compatível. O Colab atual usa Python 3.13, para o qual o DeepFilterNet 0.5.6 não publica wheel; nesse ambiente o sistema executa o BetterDeclipper e informa que o denoiser neural não foi usado.
- O arquivo principal manteve o candidato conservador do BetterDeclipper; o arquivo neural continuou disponível
  separadamente para auditoria A/B.
- Nenhum modelo de conversão, clonagem ou síntese de voz foi utilizado.

O teste local não prova uso de T4. No Colab, a interface deve mostrar CUDA somente quando
`torch.cuda.is_available()` for verdadeiro e deve registrar o dispositivo retornado pelo processamento.
