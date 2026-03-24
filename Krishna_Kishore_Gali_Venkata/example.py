import gradio as gr
import warnings
warnings.filterwarnings("ignore")

def greet(name, intensity):
    """Greet the user with a message based on intensity level."""
    greeting = "Hello, " + name + "!" * intensity
    return greeting

def calculate_bmi(weight, height):
    """Calculate BMI given weight (kg) and height (cm)."""
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return f"{bmi:.2f}"

def text_analysis(text):
    """Analyze text and return word count, character count, and sentiment."""
    word_count = len(text.split())
    char_count = len(text)
    sentiment = "Positive" if any(word in text.lower() for word in ["good", "great", "excellent", "happy"]) else "Neutral"
    return word_count, char_count, sentiment

def sentiment_analysis(text):
    """Perform sentiment analysis using Hugging Face model."""
    from transformers import pipeline
    
    # Load pre-trained sentiment analysis model
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    if not text.strip():
        return "Please enter some text", 0.0
    
    result = classifier(text[:512])[0]  # Limit to 512 tokens
    label = result['label']
    score = result['score']
    
    sentiment_label = "😊 Positive" if label == "POSITIVE" else "😞 Negative"
    return sentiment_label, round(score * 100, 2)

# Create Gradio interface with multiple tabs
with gr.Blocks(title="Demo Application", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Gradio Demo Application")
    gr.Markdown("A multi-purpose demo app with various tools including AI-powered sentiment analysis")
    
    with gr.Tab("📝 Greeting"):
        gr.Markdown("### Simple Greeting Generator")
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(label="Enter your name", placeholder="John Doe")
                intensity_slider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Excitement Level")
                greet_btn = gr.Button("Greet!", variant="primary")
            with gr.Column():
                greet_output = gr.Textbox(label="Greeting")
        greet_btn.click(fn=greet, inputs=[name_input, intensity_slider], outputs=greet_output)
    
    with gr.Tab("⚖️ BMI Calculator"):
        gr.Markdown("### Body Mass Index Calculator")
        with gr.Row():
            with gr.Column():
                weight_input = gr.Number(label="Weight (kg)", value=70)
                height_input = gr.Number(label="Height (cm)", value=175)
                calc_btn = gr.Button("Calculate BMI", variant="primary")
            with gr.Column():
                bmi_output = gr.Textbox(label="Your BMI")
        calc_btn.click(fn=calculate_bmi, inputs=[weight_input, height_input], outputs=bmi_output)
    
    with gr.Tab("📊 Text Analysis"):
        gr.Markdown("### Analyze Your Text")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="Enter text", placeholder="Type something here...", lines=5)
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                word_output = gr.Number(label="Word Count")
                char_output = gr.Number(label="Character Count")
                sentiment_output = gr.Textbox(label="Sentiment")
        analyze_btn.click(fn=text_analysis, inputs=text_input, outputs=[word_output, char_output, sentiment_output])
    
    with gr.Tab("🤖 AI Sentiment Analysis"):
        gr.Markdown("### Hugging Face Sentiment Analysis")
        gr.Markdown("This uses a pre-trained **DistilBERT** model from Hugging Face to analyze the sentiment of your text.")
        with gr.Row():
            with gr.Column(scale=2):
                sentiment_text_input = gr.Textbox(
                    label="Enter text for sentiment analysis", 
                    placeholder="e.g., I love this product! It works amazingly well.",
                    lines=5
                )
                sentiment_btn = gr.Button("Analyze Sentiment", variant="primary", size="lg")
            with gr.Column(scale=1):
                sentiment_label_output = gr.Textbox(label="Sentiment", scale=1)
                confidence_score = gr.Number(label="Confidence Score (%)", scale=1)
        
        # Add example texts
        gr.Examples(
            examples=[
                ["I absolutely love this! Best purchase ever."],
                ["This is terrible, I'm very disappointed."],
                ["The weather is nice today."],
                ["I'm so happy with the results!"],
                ["This product broke after one day of use."],
            ],
            inputs=sentiment_text_input
        )
        
        sentiment_btn.click(
            fn=sentiment_analysis, 
            inputs=sentiment_text_input, 
            outputs=[sentiment_label_output, confidence_score]
        )

if __name__ == "__main__":
    demo.launch()
