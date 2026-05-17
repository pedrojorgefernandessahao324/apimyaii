
from flask import Flask, request, jsonify
import g4f
from g4f.Provider import (
    Blackbox,      # GPT-4o (pro)
    Liaobots,      # GPT-4o (fallback)
    DDG,           # GPT-4 e GPT-4o-mini
    Binjie,        # GPT-4 (fallback)
    OpenaiChat     # GPT-4o-mini (fallback)
)
import traceback

app = Flask(__name__)

# Mapeamento dos tipos (modelo g4f, lista de providers)
MODEL_MAP = {
    "pro": {
        "model": "gpt-4o",
        "providers": [Blackbox, Liaobots],
        "description": "GPT-4o - Mais inteligente, raciocínio profundo"
    },
    "default": {
        "model": "gpt-4",
        "providers": [DDG, Binjie],
        "description": "GPT-4 - Padrão, equilibrio entre qualidade e velocidade"
    },
    "fast": {
        "model": "gpt-4o-mini",
        "providers": [DDG, OpenaiChat],
        "description": "GPT-4o-mini - Mais rápido, respostas curtas e diretas"
    }
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "g4f API on Vercel",
        "models": {
            "pro": MODEL_MAP["pro"]["description"],
            "default": MODEL_MAP["default"]["description"],
            "fast": MODEL_MAP["fast"]["description"]
        }
    })

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    Endpoint compatível com OpenAI API
    
    Payload esperado:
    {
        "model": "pro",           # ou "default", "fast"
        "messages": [
            {"role": "system", "content": "Você é um assistente útil"},
            {"role": "user", "content": "Olá!"},
            {"role": "assistant", "content": "Olá! Como posso ajudar?"}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Payload JSON é obrigatório"}), 400
        
        # Pega o tipo de modelo (pro, default, fast)
        model_type = data.get("model", "default")
        messages = data.get("messages", [])
        
        # Valida o tipo de modelo
        if model_type not in MODEL_MAP:
            return jsonify({
                "error": f"Modelo inválido. Use: {', '.join(MODEL_MAP.keys())}"
            }), 400
        
        # Valida se tem mensagens
        if not messages:
            return jsonify({"error": "Campo 'messages' é obrigatório"}), 400
        
        # Valida roles suportadas
        valid_roles = {"system", "user", "assistant"}
        for msg in messages:
            role = msg.get("role")
            if role not in valid_roles:
                return jsonify({
                    "error": f"Role inválida: '{role}'. Use system, user ou assistant"
                }), 400
        
        # Pega a configuração do tipo de modelo
        model_config = MODEL_MAP[model_type]
        g4f_model = model_config["model"]
        providers = model_config["providers"]
        
        # Tenta cada provider da lista (fallback)
        last_error = None
        for provider in providers:
            try:
                response = g4f.ChatCompletion.create(
                    model=g4f_model,
                    provider=provider,
                    messages=messages,
                    timeout=30  # 30 segundos para a Vercel não dar timeout
                )
                
                # Retorna no formato OpenAI
                return jsonify({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response
                        }
                    }],
                    "model": model_type,
                    "provider": provider.__name__
                })
                
            except Exception as e:
                last_error = str(e)
                continue
        
        # Se chegou aqui, nenhum provider funcionou
        return jsonify({
            "error": "Todos os providers falharam",
            "details": last_error
        }), 500
        
    except Exception as e:
        return jsonify({
            "error": "Erro interno",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

# Isso é CRÍTICO para a Vercel - o nome TEM que ser 'app'
app = app
