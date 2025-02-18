from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

# Initialize the model only once to avoid importing every time
class LlamaInference:
    def __init__(self, model_path):
        """
        Initializes the Llama model for inference from a local file.
        :param model_path: The path to the model file.
        """
        self.llm = Llama(model_path=model_path)

    def generate_response(self, instruction, input_text, max_tokens=128, temperature=0.3):
        """
        Generates a response using the Llama model.
        :param instruction: Instruction text for the model.
        :param input_text: Input text for the task.
        :param max_tokens: Maximum tokens for the output.
        :param temperature: Controls the randomness of the generated text (0.0 - 1.0).
        :return: Response text generated by the model.
        """
        # Define the alpaca_prompt format
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

        # Format the query
        formatted_query = alpaca_prompt.format(instruction=instruction, input_text=input_text)

        # Generate response with temperature
        response = self.llm(formatted_query, max_tokens=max_tokens, temperature=temperature)

        # Extract and return the response text from the 'choices' field
        return response['choices'][0]['text']

# Initialize the Llama model at the beginning (only once)
model_path = "unsloth.Q4_K_M.gguf"
llama_inference = LlamaInference(model_path)

@app.route('/generate_response', methods=['POST'])
def generate_response():
    """
    Flask route that takes a POST request to generate a response from the Llama model.
    Expects JSON input with 'instruction' and 'input_text'.
    """
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract instruction and input_text from the incoming JSON
        instruction = data.get('instruction')
        input_text = data.get('input_text')

        # Validate input
        if not instruction or not input_text:
            return jsonify({"error": "Both 'instruction' and 'input_text' are required."}), 400

        # Generate response using the LlamaInference class
        response_text = llama_inference.generate_response(instruction, input_text)

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
