# BetterDeclipper AI Studio

Interface Gradio em português para executar o [BetterDeclipper oficial](https://github.com/Qupci/BetterDeclipper) no Google Colab, usando GPU NVIDIA quando o PyTorch confirma CUDA e CPU como alternativa.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegogomespi/betterdeclipper_colab/blob/main/betterdeclipper_colab.ipynb)

## Onde ficam o programa e os áudios?

- **GitHub:** contém `app.py`, `colab_launcher.py`, o notebook e as dependências. O notebook sempre baixa a versão atual do programa daqui.
- **Sessão temporária do Colab:** recebe o áudio enviado na interface, executa o BetterDeclipper e guarda os WAVs recuperados em `/content/betterdeclipper_resultados`.

Ao encerrar ou reiniciar a sessão do Colab, os arquivos temporários são apagados. Baixe os resultados e análises que deseja conservar. A versão atual não monta nem grava no Google Drive.

## Como usar

1. Clique em **Open in Colab**.
2. No Colab, abra **Ambiente de execução → Alterar tipo de ambiente de execução** e selecione **T4 GPU**.
3. Execute a célula **INICIAR BETTERDECLIPPER**.
4. Confirme que o diagnóstico mostra `cuda_available: True` e o nome da GPU. Se CUDA não estiver disponível, o sistema funciona em CPU e avisa que poderá ser lento.
5. Envie WAV, FLAC ou MP3. O diagnóstico oficial começa automaticamente.
6. Teste 15, 30 ou 60 segundos antes de processar a música completa.
7. Baixe o resultado em WAV PCM de 24 bits, FLAC 24 bits ou MP3 320 kbps. WAV/FLAC são as opções recomendadas para preservar a qualidade.

O link Gradio compartilhado é temporário e existe somente enquanto a célula estiver em execução. Não compartilhe esse endereço se o áudio for confidencial.

## Recursos

- Interface, relatórios, botões e mensagens em português do Brasil.
- Presets oficiais `fast`, `normal`, `high` e `best`, apresentados com nomes explicativos em português.
- Modos oficiais `auto`, `hard`, `soft` e `limiter`.
- Diagnóstico real de clipping digital, teto de limitador, clipping borrado e indícios de saturação/ombro suave, sem porcentagem inventada de “recuperação”.
- Duração do áudio, taxa de amostragem, canais, GPU/CPU, memória CUDA e tempo estimado.
- Processamento completo ou trecho a partir de um ponto escolhido.
- Plano com vários trechos, configurações independentes e transições de 50 ms.
- Reutilização da análise oficial sem reenviar o áudio.
- Comparação inteligente de 30 segundos em Auto × Hard × Soft.
- Players A/B nivelados por loudness (LUFS) somente para audição comparativa; os arquivos exportados não recebem esse ajuste.
- Verificação numérica da saída (correlação, diferença RMS e amostras alteradas), sempre descrita como medida de mudança — não como prova de melhora sonora.
- Aviso quando Automático e Clipping duro produzem exatamente o mesmo resultado no trecho analisado.
- Formas de onda, espectrogramas e indicação das amostras realmente alteradas.
- Progresso real por etapas, tempo restante aproximado e logs detalhados.
- Exportação WAV PCM 24 bits, FLAC 24 bits e MP3 320 kbps, mantendo a taxa de amostragem original.
- Bloqueio de processamento simultâneo e separação entre entrada e resultados.

## Observações técnicas

O aplicativo chama diretamente a API Python oficial:

```python
from betterdeclipper import declip
x, info = declip(y, sr, preset="normal", mode="auto", device="auto")
```

`device="auto"` escolhe CUDA somente quando `torch.cuda.is_available()` é verdadeiro. A interface também mostra o dispositivo devolvido pelo próprio processamento. Não é feita simulação de restauração.

O BetterDeclipper pode reconstruir picos acima de 0 dBFS. Para entregar PCM de 24 bits sem criar novo clipping, a exportação aplica atenuação de segurança para -0,1 dBFS apenas quando necessário. Isso é informado no log e não é apresentado como restauração, equalização ou masterização.

O próprio motor oficial processa áudios longos em blocos de 20 segundos, com 1,5 segundo de contexto e crossfade curto. A interface preserva esse mecanismo e não cria uma segunda divisão externa no processamento completo.

## Etapas seguintes

- **Segunda etapa:** importação e tratamento individual de stems existentes (voz, baixo, bateria e outros) e mixagem com volume/panorama. Nenhuma separação automática será obrigatória.
- **Terceira etapa:** reconstrução instrumental opcional por transcrição MIDI e instrumentos virtuais, em módulo separado e claramente identificado. Não será apresentada equalização como reconstrução.

## Estrutura

- `app.py`: interface e processamento Gradio.
- `colab_launcher.py`: sincronização, instalação, diagnóstico CUDA e inicialização.
- `requirements.txt`: dependências e versão validada do motor oficial.
- `betterdeclipper_colab.ipynb`: notebook de uma célula para o Colab.
- `testes/validar_colab.py`: validação com WAV sintético realmente clipado.

## Licenças

Este repositório fornece somente a interface. O motor BetterDeclipper é instalado do repositório oficial e mantém sua própria licença MIT.
