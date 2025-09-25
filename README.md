Script Python que classifica e-mails e gera respostas com Google Gemini.

Configuração da chave: defina `GEMINI_API_KEY` no ambiente ou copie `config.example.env` para `.env` (não compartilhe a chave).

Instalação: `python -m venv .venv && source .venv/bin/activate` (Linux/macOS) ou `.venv/Scripts/Activate.ps1` (Windows) e depois `pip install -r requirements.txt`.

Execução: `GEMINI_API_KEY=... python main.py` (ou usando `.env`). Saída: 5–8 linhas JSON (uma por e-mail). Modelo (opcional): ajuste `GEMINI_MODEL` (padrão: `gemini-flash-lite-latest`). Logs são sucintos e sem PII.

Testes rápidos: após instalar dependências e setar a chave, rode `python main.py` e verifique JSON válido por linha no stdout.

Ações por categoria:
- Reclamação: {"acao":"abrir_notificacao_slack","destino":"#reclamacoes-urgentes"}
- Sugestão: {"acao":"encaminhar_time_produto","fila":"ideias"}
- Dúvida: {"acao":"responder_cliente","template":"faq_basico"}
- Elogio: {"acao":"marcar_como_elogio","etiqueta":"elogios"}

Privacidade: não persiste dados; apenas imprime JSON no stdout.

