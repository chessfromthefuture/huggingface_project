from transformers import pipeline

# Load a sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test the model
result = classifier('I hate my dog!')
print(result)
