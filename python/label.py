import openai
import os 
openai.api_key = os.env.apikey

def label_text_with_gpt(text):
    """
    Use GPT to label a given text based on its content.
    """
    response = openai.Completion.create(
      engine="text-davinci-003",  # Choose an appropriate engine for your needs
      prompt=f"What are the labels for the following text? {text}",
      temperature=0.5,
      max_tokens=60,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    labels = response.choices[0].text.strip()
    return labels

# Example usage:
post_text = "Here's a detailed explanation on how blockchain technology works..."
post_labels = label_text_with_gpt(post_text)
print(post_labels)
