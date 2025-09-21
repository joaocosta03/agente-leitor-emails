#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classifica e-mails em {Reclamação, Sugestão, Dúvida, Elogio}, gera resumo+resposta
e decide ação de roteamento. Usa Google Gemini com prompts canônicos em PT-BR
e robustez com retentativas (tenacity). Saída: um JSON por e-mail (stdout).
"""

import os
import sys
import json
import re
import logging
from typing import Any, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except Exception as e:
    print("Erro ao importar google-generativeai. Instale com: pip install -r requirements.txt", file=sys.stderr)
    raise

# -----------------------
# Configuração de logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s"
)
logger = logging.getLogger("gemini-email-router")

# ------------------------------------
# Constantes e prompts (instruções)
# ------------------------------------
ALLOWED_CATEGORIES = {"Reclamação", "Sugestão", "Dúvida", "Elogio"}

PROMPT_CLASSIFICACAO = """Objetivo: receber texto de e-mail e retornar apenas JSON:

{"categoria":"<Reclamação|Sugestão|Dúvida|Elogio>","justificativa":"<1 frase>"}

Instruções do prompt (inclua literalmente):

Tarefa: classifique o e-mail exatamente em UMA das categorias: Reclamação, Sugestão, Dúvida, Elogio.

“Reclamação” = relato de problema/insatisfação; “Sugestão” = proposta de melhoria; “Dúvida” = pergunta/pedido de esclarecimento; “Elogio” = feedback positivo.

Responda apenas com JSON válido no formato especificado; não inclua texto adicional.

Se ambíguo entre duas categorias, escolha a mais consistente com a intenção principal.

Se o texto estiver vazio ou ilegível, retorne: {"categoria":"Dúvida","justificativa":"Texto vazio ou incompreensível"}.

Exemplos (few-shot):

Input: “O produto chegou quebrado e o suporte não respondeu.”
Output: {"categoria":"Reclamação","justificativa":"Relata defeito e falha no suporte"}

Input: “Seria ótimo ter filtro por tamanho nas buscas.”
Output: {"categoria":"Sugestão","justificativa":"Propõe melhoria de usabilidade"}

Input: “Qual é o prazo para troca de um item com defeito?”
Output: {"categoria":"Dúvida","justificativa":"Pergunta sobre política de troca"}

Input: “Equipe atenciosa e entrega rápida, parabéns!”
Output: {"categoria":"Elogio","justificativa":"Expressa satisfação e reconhecimento"}

[ENTRADA]
{{texto}}
"""

PROMPT_SUM_RESPOSTA = """Objetivo: receber texto de e-mail e retornar apenas JSON:

{"resumo":"<1 frase>","resposta":"<resposta curta e educada em PT-BR>"}

Instruções do prompt (inclua literalmente):

Faça um resumo em 1 frase, fiel ao conteúdo (sem inventar).

Gere uma resposta curta/educada, em PT-BR, neutra e objetiva; se houver nº de pedido mencionado, reconheça-o.

Proíba promessas sem base (não inventar prazos/status).

Responda apenas com JSON válido no formato especificado.

Se o texto estiver vazio, use resumo e resposta padrão solicitando mais detalhes.

Exemplo:

Input: “Pedido #12345 atrasado; paguei frete expresso; preciso até sábado.”
Output:

{"resumo":"Cliente relata atraso do pedido #12345 com frete expresso e urgência.","resposta":"Sentimos pelo transtorno. Já solicitamos a verificação do pedido #12345 e retornaremos com o status atualizado. Caso precise de alternativa imediata, avise por favor."}

[ENTRADA]
{{texto}}
"""

REPAIR_PROMPT = "Reescreva estritamente em JSON válido no formato exigido, sem explicações."

# -----------------------
# Cliente Gemini (global)
# -----------------------
MODEL: Optional[genai.GenerativeModel] = None


def init_gemini() -> genai.GenerativeModel:
    """Configura o cliente Gemini a partir de variáveis de ambiente."""
    global MODEL
    # Carrega variaveis de ambiente do arquivo .env, se existir
    load_dotenv()  # opcional; não falha se não existir .env
    # Le a chave da API e o nome do modelo definidos no ambiente
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite").strip() or "gemini-2.0-flash-lite"

    # Interrompe a execucao caso a credencial obrigatoria nao esteja presente
    if not api_key:
        logger.error("GEMINI_API_KEY não configurada. Defina no ambiente ou em um arquivo .env.")
        sys.exit(1)

    # Configura o SDK do Gemini com a chave informada
    genai.configure(api_key=api_key)
    # Cria a instancia do modelo para reutilizacao global
    MODEL = genai.GenerativeModel(model_name=model_name)
    return MODEL


def get_model() -> genai.GenerativeModel:
    """Obtém o modelo inicializado; se ainda não, inicializa agora."""
    global MODEL
    # Inicializa o cliente somente na primeira utilizacao
    if MODEL is None:
        return init_gemini()
    # Reaproveita o modelo ja carregado nas chamadas seguintes
    return MODEL


# ---------------------------------
# Chamada ao modelo com retentativas
# ---------------------------------
class GeminiCallError(Exception):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiCallError),
)
def call_gemini(
    prompt: str,
    input_text: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_output_tokens: int = 512,
) -> str:
    """
    Executa uma geração com Gemini usando um prompt e um texto de entrada.
    Retorna o conteúdo de texto agregado. Levanta GeminiCallError para retentativas.
    """
    # Garante que prompt e texto sejam strings validas antes da chamada
    if not isinstance(prompt, str) or not isinstance(input_text, str):
        raise GeminiCallError("prompt/input_text inválidos (tipos incorretos).")

    # Injeta o conteudo do e-mail no template do prompt
    rendered = prompt.replace("{{texto}}", input_text)

    try:
        # Recupera a instancia global do modelo configurado
        model = get_model()
        # Dispara a geracao de conteudo na API do Gemini
        resp = model.generate_content(
            rendered,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
            },
        )
        # Tenta obter o texto principal retornado pela API
        text = getattr(resp, "text", None)
        # Faz fallback para concatenar partes caso o campo principal esteja vazio
        if not text:
            try:
                parts = resp.candidates[0].content.parts
                text = "".join(getattr(p, "text", "") for p in parts)
            except Exception:
                text = None

        # Solicita nova tentativa se a resposta permanecer vazia
        if not text or not text.strip():
            raise GeminiCallError("Resposta vazia do modelo.")
        return text.strip()
    except Exception as e:
        # Propaga erros como GeminiCallError para acionar retentativas
        raise GeminiCallError(str(e)) from e


# -----------------------
# Utilitários
# -----------------------


def remove_code_fences(s: str) -> str:
    """Remove cercas de código Markdown para facilitar parse de JSON."""
    # Retorna vazio se a entrada nao for texto
    if not isinstance(s, str):
        return ""
    # Remove espacos extras nas extremidades
    s = s.strip()
    # Elimina cercas de codigo no inicio
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    # Elimina cercas de codigo no fim
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_json_maybe(s: str) -> Optional[Dict[str, Any]]:
    """Tenta carregar JSON com algumas tolerâncias leves."""
    # Nao tenta parsear quando nao ha conteudo
    if s is None:
        return None
    # Limpa possiveis cercas de Markdown antes do parse
    txt = remove_code_fences(s)
    # Tenta carregar o JSON diretamente da resposta
    try:
        return json.loads(txt)
    except Exception:
        # Busca manualmente um objeto JSON delimitado dentro do texto
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    # Retorna None caso nenhum JSON valido seja encontrado
    return None


# -----------------------
# Funções de alto nível
# -----------------------
def validate_category(categoria: str) -> str:
    """Garante que a categoria esteja no conjunto permitido; caso contrário, usa 'Dúvida'."""
    # Aceita apenas categorias previstas, removendo espacos extras
    if isinstance(categoria, str) and categoria.strip() in ALLOWED_CATEGORIES:
        return categoria.strip()
    # Registra um aviso quando o modelo devolve categoria fora do catalogo
    logger.warning("Categoria inválida retornada pelo modelo; mapeando para 'Dúvida'.")
    return "Dúvida"


def classify_email(text: str) -> Dict[str, str]:
    """Classifica um e-mail com justificativa curta. Sempre retorna JSON válido."""
    # Retorna fallback padrao quando o corpo do e-mail esta vazio
    if not isinstance(text, str) or not text.strip():
        return {"categoria": "Dúvida", "justificativa": "Texto vazio ou incompreensível"}

    # Chama o modelo para classificar o texto recebido
    raw = call_gemini(
        prompt=PROMPT_CLASSIFICACAO,
        input_text=text,
        temperature=0.2,
        top_p=0.3,
        max_output_tokens=256,
    )
    # Tenta interpretar a resposta como JSON estruturado
    data = parse_json_maybe(raw)

    # Solicita ao modelo que reescreva a resposta caso o JSON venha invalido
    if data is None:
        # Solicita reparo de JSON ao modelo para tentar novo parse
        repair_raw = call_gemini(
            prompt=REPAIR_PROMPT,
            input_text=raw,
            temperature=0.2,
            top_p=0.3,
            max_output_tokens=256,
        )
        data = parse_json_maybe(repair_raw)

    # Usa categoria de seguranca se o reparo tambem falhar
    if not isinstance(data, dict):
        return {"categoria": "Dúvida", "justificativa": "Falha ao interpretar resposta do modelo"}

    # Normaliza a categoria e garante que esteja permitida
    categoria = validate_category(data.get("categoria", "Dúvida"))
    justificativa = data.get("justificativa", "") or "Classificação automática"
    return {"categoria": categoria, "justificativa": justificativa}


def summarize_and_reply(text: str) -> Dict[str, str]:
    """Gera resumo (1 frase) e resposta curta/educada em PT-BR."""
    # Retorna mensagens padrao quando nao ha conteudo para resumir
    if not isinstance(text, str) or not text.strip():
        return {
            "resumo": "Texto vazio; é necessário mais contexto do cliente.",
            "resposta": "Poderia fornecer mais detalhes (ex.: número do pedido e descrição do ocorrido) para ajudarmos com precisão?",
        }

    # Pede ao modelo resumo e resposta curta para o e-mail
    raw = call_gemini(
        prompt=PROMPT_SUM_RESPOSTA,
        input_text=text,
        temperature=0.4,
        top_p=0.5,
        max_output_tokens=512,
    )
    # Procura extrair JSON estruturado com resumo e resposta
    data = parse_json_maybe(raw)

    # Tenta reparar a saida caso o primeiro parse falhe
    if data is None:
        # Solicita reparo de JSON ao modelo para tentar novo parse
        repair_raw = call_gemini(
            prompt=REPAIR_PROMPT,
            input_text=raw,
            temperature=0.4,
            top_p=0.5,
            max_output_tokens=512,
        )
        data = parse_json_maybe(repair_raw)

    # Aplica fallback seguro quando nao e possivel confiar nos dados
    if not isinstance(data, dict):
        return {
            "resumo": "Conteúdo não pôde ser resumido com segurança.",
            "resposta": "Agradecemos a mensagem. Pode compartilhar mais detalhes para apoiarmos melhor?",
        }

    # Garante texto padrao caso o resumo venha vazio
    resumo = data.get("resumo", "") or "Resumo indisponível."
    # Define resposta padrao quando o modelo nao retornar conteudo
    resposta = data.get("resposta", "") or "Agradecemos a mensagem. Em breve retornaremos com mais informações."
    return {"resumo": resumo, "resposta": resposta}


def route_action(category: str) -> Dict[str, str]:
    """Decide a ação com base na categoria."""
    # Reforca que a categoria usada na decisao e valida
    category = validate_category(category)
    # Direciona reclamacoes para canal critico no Slack
    if category == "Reclamação":
        return {"acao": "abrir_notificacao_slack", "destino": "#reclamacoes-urgentes"}
    # Encaminha sugestoes para o time de produto
    if category == "Sugestão":
        return {"acao": "encaminhar_time_produto", "fila": "ideias"}
    # Para duvidas, orienta resposta ao cliente
    if category == "Dúvida":
        return {"acao": "responder_cliente", "template": "faq_basico"}
    # Demais casos viram elogios etiquetados
    return {"acao": "marcar_como_elogio", "etiqueta": "elogios"}


# -----------------------
# Execução principal
# -----------------------
def main() -> None:
    # Inicializa o cliente do Gemini antes de processar
    init_gemini()

    # Define a lista de e-mails de exemplo a serem processados
    emails = [
        {
            "id": "eml-001",
            "remetente": "cliente1@example.com",
            "assunto": "Produto com problema e sem resposta",
            "corpo": "O produto chegou com a tela trincada e ninguém responde meu ticket.",
        },
        {
            "id": "eml-002",
            "remetente": "cliente2@example.com",
            "assunto": "Filtro por cor",
            "corpo": "Poderiam adicionar filtro por cor na busca?",
        },
        {
            "id": "eml-003",
            "remetente": "cliente3@example.com",
            "assunto": "Troca em 30 dias",
            "corpo": "Como faço para trocar um item defeituoso dentro de 30 dias?",
        },
        {
            "id": "eml-004",
            "remetente": "cliente4@example.com",
            "assunto": "Agradecimento",
            "corpo": "Atendimento excelente, entrega no prazo. Obrigado!",
        },
        {
            "id": "eml-005",
            "remetente": "cliente5@example.com",
            "assunto": "Pedido atrasado",
            "corpo": "Meu pedido #98765 está atrasado e viajo amanhã, o que posso fazer?",
        },
        {
            "id": "eml-006",
            "remetente": "cliente6@example.com",
            "assunto": "Sem conteúdo",
            "corpo": "",
        },
        {
            "id": "eml-007",
            "remetente": "cliente7@example.com",
            "assunto": "Texto ruidoso",
            "corpo": "asdf 123!!! ?? ajuda, talvez, não sei, pedido 00000?",
        },
        {
            "id": "eml-008",
            "remetente": "cliente8@example.com",
            "assunto": "Elogio + sugestão",
            "corpo": "Equipe muito boa! Talvez um modo escuro no app ajudaria à noite.",
        },
    ]

    # Itera pelos e-mails simulados para aplicar o fluxo
    for email in emails:
        try:
            # Extrai pequeno trecho do corpo para logging
            snippet = email.get("corpo", "") or ""
            # Registra no log qual e-mail esta em processamento
            logger.info(
                f"Processando id={email.get('id')} assunto='{email.get('assunto', '')}' corpo='{snippet}'"
            )

            # Obtem a classificacao automatica do e-mail
            cls = classify_email(email.get("corpo", ""))
            # Extrai a categoria retornada com fallback seguro
            cat = cls.get("categoria", "Dúvida")
            # Determina a acao de roteamento com base na categoria
            act = route_action(cat)
            # Solicita resumo e resposta automatica ao modelo
            sr = summarize_and_reply(email.get("corpo", ""))

            # Agrupa os dados para gerar a saida final
            record = {
                "id": email.get("id"),
                "categoria": cat,
                "resumo": sr.get("resumo", ""),
                "resposta": sr.get("resposta", ""),
                "acao": act,
            }
            # Emite o resultado no formato JSON no stdout
            print(json.dumps(record, ensure_ascii=False))
        # Evita que uma excecao interrompa o processamento dos demais e-mails
        except Exception as e:
            # Registra detalhes do erro encontrado
            logger.error(f"Falha ao processar id={email.get('id')}: {e}")


if __name__ == "__main__":
    main()

