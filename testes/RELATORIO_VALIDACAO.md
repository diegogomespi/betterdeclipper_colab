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

- SonicMaster: não executado; incompatibilidades e riscos descritos em `../SONICMASTER_AUDITORIA.md`.
- Demucs HTDemucs-FT: não instalado nem executado, pois ainda não houve benefício demonstrado para este caso.

O teste local não prova uso de T4. No Colab, a interface deve mostrar CUDA somente quando
`torch.cuda.is_available()` for verdadeiro e deve registrar o dispositivo retornado pelo processamento.
