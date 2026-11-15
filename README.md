# BetterDeclipper AI Studio

Interface Gradio em português para diagnóstico e restauração musical automática com o [BetterDeclipper oficial](https://github.com/Qupci/BetterDeclipper) no Google Colab, usando GPU NVIDIA somente quando o PyTorch confirma CUDA.

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
5. Envie WAV, FLAC ou MP3.
6. Clique em **Analisar e restaurar automaticamente**.
7. O controlador escolhe um trecho representativo, testa candidatos, descarta alterações inseguras e processa a música completa.
8. Ouça original e restaurado e baixe o resultado principal em WAV PCM de 24 bits.

O link Gradio compartilhado é temporário e existe somente enquanto a célula estiver em execução. Não compartilhe esse endereço se o áudio for confidencial.

## Recursos

- Interface, relatórios, botões e mensagens em português do Brasil.
- Fluxo principal com apenas upload, um botão automático, progresso, players e download WAV.
- Seleção automática do trecho sustentado de maior energia para o teste inicial.
- Controlador que compara Auto, Hard e Soft, mede correlação, diferença RMS, desvio espectral e impulsos, e rejeita candidatos com artefatos evidentes.
- Escolha automática do preset final conforme duração e GPU confirmada.
- Diagnóstico DSP de dinâmica, piso quieto/ruído estimado, espectro, resposta de frequência descritiva e indícios de codec, sempre com limites e confiança informados.
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

O controlador automático não usa volume mais alto como critério de qualidade. As métricas sem referência limpa
servem para rejeitar resultados perigosos, não para prometer melhora. Quando nenhum candidato seguro é encontrado,
o sistema preserva o sinal e informa a limitação.

## SonicMaster e Demucs

O SonicMaster foi auditado, mas não é executado nesta versão: o pipeline oficial ainda não preserva a duração do
último bloco e o Space oficial usa A10G, não T4. O checkpoint tem cerca de 3,45 GB e depende de FLAN-T5 Large e do
VAE Stable Audio, que possui licença própria. Consulte [SONICMASTER_AUDITORIA.md](SONICMASTER_AUDITORIA.md).

Demucs HTDemucs-FT também não é instalado automaticamente enquanto não houver benefício demonstrado para este caso.

## Etapas seguintes

- **Segunda etapa:** importação e tratamento individual de stems existentes (voz, baixo, bateria e outros) e mixagem com volume/panorama. Nenhuma separação automática será obrigatória.
- **Terceira etapa:** reconstrução instrumental opcional por transcrição MIDI e instrumentos virtuais, em módulo separado e claramente identificado. Não será apresentada equalização como reconstrução.

## Estrutura

- `app.py`: interface e processamento Gradio.
- `colab_launcher.py`: sincronização, instalação, diagnóstico CUDA e inicialização.
- `requirements.txt`: dependências e versão validada do motor oficial.
- `betterdeclipper_colab.ipynb`: notebook de uma célula para o Colab.
- `testes/validar_colab.py`: validação com WAV sintético realmente clipado.
- `audio_diagnostics.py`: seleção de trecho, diagnóstico DSP e filtro de segurança dos candidatos.
- `SONICMASTER_AUDITORIA.md`: compatibilidade, dependências, licença e bloqueios encontrados.

## Licenças

Este repositório fornece somente a interface. O motor BetterDeclipper é instalado do repositório oficial e mantém sua própria licença MIT.
