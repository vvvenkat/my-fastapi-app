import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch  

class IntentModel:
    def __init__(self):
        """
        Initialize the model with DistilGPT-2 and SentenceTransformer for embeddings.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.intents = {}
        self.intent_mapping = []
        self.intent_embeddings = None
        self.data_file = "intents.json"

        # Load saved intents and embeddings if available
        self.load_model_data()

    def train(self, training_data):
        """
        Train the model with provided training data, supporting incremental updates.
        Args:
            training_data (dict): New training data provided by the user.
        """
        # Load existing intents
        try:
            with open(self.data_file, "r") as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = {}

        # Identify deleted intents
        intents_to_delete = set(existing_data.keys()) - set(training_data.keys())

        # Update existing intents with new training data
        for intent, phrases in training_data.items():
            existing_data[intent] = phrases  # Replace old phrases with new ones

        # Remove deleted intents
        for intent in intents_to_delete:
            del existing_data[intent]

        # Update the class attributes
        self.intents = existing_data
        all_phrases = []
        self.intent_mapping = []

        for intent, phrases in existing_data.items():
            all_phrases.extend(phrases)
            self.intent_mapping.extend([intent] * len(phrases))

        # Compute embeddings and save everything
        self.intent_embeddings = self.embedding_model.encode(all_phrases, convert_to_tensor=True)
        self.save_model_data()

    def detect(self, user_input):
        """
        Detect intent for a given user input.
        Args:
            user_input (str): User's query.
        Returns:
            str: Detected intent name or 'No intent matched.'
        """
        if self.intent_embeddings is None:
            return "No training data provided. Please train the model first."

        # Encode user input and compute similarity
        user_embedding = self.embedding_model.encode(user_input, convert_to_tensor=True)
        similarities = cosine_similarity(user_embedding.unsqueeze(0), self.intent_embeddings)
        max_similarity = torch.max(torch.tensor(similarities)).item()

        # Threshold-based intent matching
        if max_similarity < 0.25:  # Threshold for matching
            return "No intent matched."

        max_index = torch.argmax(torch.tensor(similarities)).item()
        return self.intent_mapping[max_index]

    def save_model_data(self):
        """
        Save the current intents to a JSON file.
        """
        with open(self.data_file, "w") as f:
            json.dump(self.intents, f, indent=4)

    def load_model_data(self):
        """
        Load intents and compute embeddings from a saved JSON file.
        """
        try:
            with open(self.data_file, "r") as f:
                self.intents = json.load(f)

            all_phrases = []
            self.intent_mapping = []

            for intent, phrases in self.intents.items():
                all_phrases.extend(phrases)
                self.intent_mapping.extend([intent] * len(phrases))

            # Compute embeddings for the loaded data
            self.intent_embeddings = self.embedding_model.encode(all_phrases, convert_to_tensor=True)
        except FileNotFoundError:
            self.intents = {}
            self.intent_embeddings = None
            self.intent_mapping = []


