# Auditoria do SonicMaster

Auditoria realizada em 15 de novembro de 2026 sobre o código oficial do
[AMAAI-Lab/SonicMaster](https://github.com/AMAAI-Lab/SonicMaster), commit
`ab32b2d456e461b6bbc188d666caf2eabefa2f31`.

## O que foi confirmado

- O código e o checkpoint SonicMaster são publicados sob Apache-2.0.
- O checkpoint `model.safetensors` tem aproximadamente 3,45 GB, 0,9 bilhão de parâmetros e tensores F32.
- A inferência oficial usa `google/flan-t5-large` e o VAE de
  `stabilityai/stable-audio-open-1.0`.
- O VAE possui licença própria, Stability AI Community License. Seu uso não deve ser apresentado como
  simplesmente Apache-2.0; aplicações comerciais precisam observar os termos da Stability AI.
- O Space oficial executa atualmente em ZeroGPU A10G, não em NVIDIA T4. A A10G possui mais memória
  que a T4 normalmente oferecida pelo Colab gratuito.
- O repositório recomenda Python 3.13, enquanto o Space oficial declara Python 3.10.13. As dependências
  também divergem: o repositório fixa PyTorch 2.4 e o Space usa PyTorch 2.8.

## Bloqueios encontrados no pipeline oficial

O arquivo oficial `infer_single.py`:

1. converte toda entrada para estéreo e 44,1 kHz;
2. divide em blocos de 30 segundos com 10 segundos de sobreposição;
3. preenche o último bloco com zeros;
4. não corta o resultado final novamente para a quantidade original de amostras;
5. pré-codifica e mantém os latentes de todos os blocos antes da inferência;
6. carrega checkpoint, FLAN-T5 Large e VAE na GPU.

Consequentemente, a versão oficial pode mudar taxa de amostragem/canais, entregar duração maior e consumir
memória demais em músicas longas. O fato de funcionar no Space A10G não comprova compatibilidade com T4.

## Decisão desta versão

O SonicMaster não é baixado nem executado pelo botão automático. A interface informa essa limitação e usa
o BetterDeclipper oficial, que foi validado localmente com referência limpa conhecida. Isso evita um download
de vários gigabytes e impede que um modelo não validado seja apresentado como funcional.

Para habilitação futura serão necessários, no mínimo:

- teste real em T4 com medição de pico de VRAM;
- processamento incremental, sem manter todos os blocos na GPU;
- corte exato para a duração original;
- política explícita para taxa de amostragem, canais e alinhamento;
- comparação objetiva e auditiva contra BetterDeclipper e original;
- aceite e documentação da licença do VAE Stable Audio;
- teste de música completa, não apenas trecho curto.

## Demucs

Demucs HTDemucs-FT não é instalado nesta versão. Separação e remontagem podem introduzir vazamento entre stems,
alterações de fase e novos artefatos. A regra do projeto exige benefício demonstrado antes de ocupar recursos da
T4; esse benefício ainda não foi demonstrado para o caso de gravação sobrecarregada.
