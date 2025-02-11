import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    theme=gr.themes.Soft(),  # Add a base theme
    css="""
        /* --- Overall Container --- */
        .gradio-container {
            background-color: hsl(240, 9.1%, 27.1%); /* Your bg-secondary */
            color: hsl(0, 0%, 100%); /*  Example text color - adjust */
            font-family: 'Inter', sans-serif;
        }

        /* --- Chatbot Area --- */
        .chatbox { /*  Might be .chatbot - inspect in browser */
            background-color: hsl(240, 9.1%, 27.1%); /* Your bg-secondary */
            border: none;
            border-radius: 0.75rem; /* Tailwind's rounded-lg */
            /* Increase height - add 1 inch (96px) to existing height */
            height: calc(100% + 96px); /*  DYNAMICALLY increase height */
            overflow-y: auto;
        }

        /* --- Individual Messages --- */
        .user .message {
            background-color: hsl(222.2, 84%, 4.9%) !important; /* Your bg-primary */
            color: hsl(0, 0%, 100%) !important; /* Your text-primary-foreground */
            border-radius: 0.75rem;
            padding: 0.75rem;
            max-width: 80%;
            margin-left: auto;
        }

        .bot .message {
            background-color: hsl(240, 9.1%, 27.1%); /* Your bg-secondary */
            color: hsl(0, 0%, 100%);          /* Example text color */
            border-radius: 0.75rem;
            padding: 0.75rem;
            max-width: 80%;
            margin-right: auto;
        }

         /* --- Input Area --- */
        .textbox {
            background-color: white;
            border: 1px solid hsl(240, 3.7%, 15.9%); /* Example border */
            border-radius: 0.375rem;
            padding: 0.5rem 0.75rem;

        }
        .btn.primary{
             background-color: hsl(222.2, 84%, 4.9%) !important; /* Your bg-primary */
            color: hsl(0, 0%, 100%) !important; /* Your text-primary-foreground */
            border: none;
        }

        /* --- Add more styles as needed --- */
    """,
)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()