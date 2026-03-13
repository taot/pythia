import torch
import gradio as gr

from transformers import AutoTokenizer, GPTNeoXForCausalLM

from lab import utils


DEFAULT_MODEL_NAME = "EleutherAI/pythia-410m-deduped"
DEFAULT_REVISION = "step143000"


APP_STATE = {
    "model": None,
    "tokenizer": None,
    "device": "cpu",
    "model_name": None,
    "revision": None,
}


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _empty_history():
    return []


def load_model(model_name, revision, cache_dir):
    device = _get_device()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Equivalent generic form:
    # model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    APP_STATE["model"] = model
    APP_STATE["tokenizer"] = tokenizer
    APP_STATE["device"] = device
    APP_STATE["model_name"] = model_name
    APP_STATE["revision"] = revision

    status = (
        f"Loaded `{model_name}` at `{revision}` on `{device}`. "
        "You can start generating text now."
    )
    print(status)
    return status, _empty_history()


def generate_completion(history, prompt, max_new_tokens, temperature, top_p):
    prompt = prompt.strip()
    if not prompt:
        return history, ""

    model = APP_STATE["model"]
    tokenizer = APP_STATE["tokenizer"]
    if model is None or tokenizer is None:
        raise gr.Error("Load a model first.")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(APP_STATE["device"]) for key, value in inputs.items()}

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)

    with torch.inference_mode():
        output_tokens = model.generate(**inputs, **generation_kwargs)

    decoded_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    if not decoded_text:
        decoded_text = "[No response generated]"

    updated_history = history + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": decoded_text},
    ]
    return updated_history, ""


def clear_chat():
    return _empty_history()


with gr.Blocks(title="Pythia Chat UI") as demo:
    gr.Markdown("# Pythia Chat UI")
    gr.Markdown(
        "Load a checkpoint first, then generate continuations from plain text prompts. "
        "This matches `lab/guides/run_inference.py` rather than a chat-formatted model."
    )

    with gr.Row():
        model_name = gr.Textbox(label="Model Name", value=DEFAULT_MODEL_NAME)
        revision = gr.Textbox(label="Revision", value=DEFAULT_REVISION)
        cache_dir_str = utils.get_cache_dir(model_name.value, revision.value)
        cache_dir = gr.Textbox(label="Cache Dir", value=cache_dir_str, interactive=False)

    with gr.Row():
        load_button = gr.Button("Load Model", variant="primary")
        clear_button = gr.Button("Clear Chat")

    status = gr.Markdown("Model not loaded.")
    chatbot = gr.Chatbot(label="Chat", height=500)

    with gr.Row():
        message = gr.Textbox(
            label="Prompt", placeholder="Enter a text prompt...", scale=4
        )
        submit_button = gr.Button("Generate", variant="primary", scale=1)

    with gr.Row():
        max_new_tokens = gr.Slider(
            label="Max New Tokens", minimum=16, maximum=512, value=128, step=1
        )
        temperature = gr.Slider(
            label="Temperature", minimum=0.0, maximum=2.0, value=0.0, step=0.1
        )
        top_p = gr.Slider(
            label="Top P", minimum=0.1, maximum=1.0, value=0.95, step=0.05
        )

    load_button.click(
        fn=load_model,
        inputs=[model_name, revision, cache_dir],
        outputs=[status, chatbot],
    )
    submit_button.click(
        fn=generate_completion,
        inputs=[chatbot, message, max_new_tokens, temperature, top_p],
        outputs=[chatbot, message],
    )
    message.submit(
        fn=generate_completion,
        inputs=[chatbot, message, max_new_tokens, temperature, top_p],
        outputs=[chatbot, message],
    )
    clear_button.click(fn=clear_chat, outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
