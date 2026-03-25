import gradio as gr
from transformers import pipeline

# 1. Load the pre-trained sentiment analysis model using the Hugging Face pipeline
# The default model is 'distilbert-base-uncased-finetuned-sst-2-english' which is a sentiment model
classifier = pipeline("sentiment-analysis")

# 2. Define the function that Gradio will call to get predictions
def sentiment_analyzer(text):
    if not text:
        return "Please enter some text."
    
    # The pipeline returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.999}]
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    
    return f"Sentiment: {label} (Confidence: {score:.4f})"

# 3. Create the Gradio interface
# We define an input component (Textbox) and an output component (Textbox)
# The 'fn' is the function to call, and 'inputs' and 'outputs' define the data flow
demo = gr.Interface(
    fn=sentiment_analyzer,
    inputs=gr.Textbox(lines=5, label="Enter text for sentiment analysis", placeholder="Type your text here..."),
    outputs=gr.Textbox(label="Result"),
    title="Simple Sentiment Analysis App",
    description="Analyze the sentiment (positive/negative) of any text using a pre-trained Hugging Face model."
)

# 4. Launch the application
if __name__ == "__main__":
    # The 'share=True' option generates a temporary public link to share your app
    demo.launch()
