import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any
import tiktoken
from openai import OpenAI
import yaml
import os
import json
from collections import deque
import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)

class Document:
    def __init__(self, content: str, path: str):
        self.content = content
        self.path = path

    def __str__(self):
        return f"Document(path={self.path})"

class ImprovedDocumentIndexer:
    def __init__(self, max_tokens: int = 8191):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_vectors = None
        self.documents = []
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens

    def ingest_documents(self, docs_path: str):
        for root, _, files in os.walk(docs_path):
            for file in files:
                if file.endswith(('.txt', '.md')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        self.documents.append(Document(content, file_path))
                    except Exception as e:
                        print(f"{Fore.RED}Error reading file {file_path}: {str(e)}")
        
        if not self.documents:
            print(f"{Fore.YELLOW}No .txt or .md documents found in {docs_path}")
            return

        print(f"{Fore.GREEN}Ingested {len(self.documents)} documents.")
        self._build_index()

    def _build_index(self):
        document_contents = [doc.content for doc in self.documents]
        self.document_vectors = self.vectorizer.fit_transform(document_contents)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = similarities[idx]
            results.append((doc, score))
        
        return results

    def get_context(self, query: str, max_tokens: int = 4096) -> str:
        results = self.search(query)
        context = ""
        total_tokens = 0
        
        for doc, score in results:
            doc_tokens = self.encoding.encode(doc.content)
            if total_tokens + len(doc_tokens) <= max_tokens:
                context += f"{Fore.CYAN}[From: {doc.path}]{Fore.RESET}\n{doc.content}\n\n"
                total_tokens += len(doc_tokens)
            else:
                remaining_tokens = max_tokens - total_tokens
                partial_content = self.encoding.decode(doc_tokens[:remaining_tokens])
                context += f"{Fore.CYAN}[From: {doc.path}]{Fore.RESET}\n{partial_content}\n\n"
                break
        
        return context.strip()

    def get_topic_list(self) -> List[str]:
        return list(set(os.path.dirname(doc.path) for doc in self.documents))

    def calculate_relevance_score(self, query: str) -> float:
        results = self.search(query, top_k=1)
        if results:
            score = results[0][1]
            # Debug output
            print(f"{Fore.YELLOW}Debug: Raw relevance score: {score}")
            # Adjust the score to a more reasonable range
            adjusted_score = (score + 1) / 2  # This maps [-1, 1] to [0, 1]
            return adjusted_score
        return 0.0

class AnticipationAttentionFramework:
    def __init__(self):
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
        )
        self.indexer = ImprovedDocumentIndexer()
        self.prompts = self._load_prompts()
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = 4096
        self.attention_threshold = 0.70
        self.conversation_history = deque(maxlen=5)  # Sliding window of last 5 turns
        self.relevance_scores = deque(maxlen=5)  # Sliding window of last 5 relevance scores
        self.conversation_file = 'conversation_history.jsonl'
        self.load_conversation_history()
        self.cumulative_anticipation = 0.0

    def _load_prompts(self) -> Dict[str, Dict[str, str]]:
        prompts = {}
        prompts_dir = 'prompts'
        for filename in os.listdir(prompts_dir):
            if filename.endswith('.yaml'):
                with open(os.path.join(prompts_dir, filename), 'r') as file:
                    prompt_data = yaml.safe_load(file)
                    prompts[filename[:-5]] = prompt_data
        return prompts

    def ingest_documents(self, docs_path: str):
        self.indexer.ingest_documents(docs_path)

    def _prepare_context(self, query: str) -> str:
        return self.indexer.get_context(query, self.max_context_tokens)

    def _get_topic_list(self) -> str:
        return "\n".join(self.indexer.get_topic_list())

    def calculate_anticipation(self, relevance_score: float) -> float:
        self.relevance_scores.append(relevance_score)
        
        alpha = 0.3  # Smoothing factor
        anticipation_increment = relevance_score
        for i, score in enumerate(reversed(self.relevance_scores)):
            anticipation_increment = alpha * score + (1 - alpha) * anticipation_increment
        
        self.cumulative_anticipation += anticipation_increment
        return self.cumulative_anticipation

    def process_query(self, query: str) -> Tuple[str, float, float]:
        context = self._prepare_context(query)
        relevance_score = self.indexer.calculate_relevance_score(query)
        
        # Debug output
        print(f"{Fore.YELLOW}Debug: Adjusted relevance score: {relevance_score:.4f}")
        
        if relevance_score == 0:
            response = self.prompts['responses']['no_relevant_docs'].format(
                query=query,
                topics=self._get_topic_list()
            )
            return response, relevance_score, self.cumulative_anticipation
        
        anticipation = self.calculate_anticipation(relevance_score)
        
        if anticipation >= self.attention_threshold:
            system_prompt = self.prompts['system']['base'].format(
                context=context,
                conversation_history=self.format_conversation_history()
            )
            user_prompt = self.prompts['user']['query_template'].format(query=query)
            
            print(f"{Fore.MAGENTA}\nLLM Input:")
            print(f"{Style.DIM}{system_prompt}")
            print(f"{Style.DIM}{user_prompt}")
            
            llm_response = self.client.chat.completions.create(
                model="hermes3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response = llm_response.choices[0].message.content
            
            print(f"{Fore.MAGENTA}\nLLM Output:")
            print(f"{Style.NORMAL}{response}")
            
            response += f"\n\n{Fore.YELLOW}Confidence level: {anticipation:.2f}"
            response += f"\n\n{Fore.CYAN}Source: {self.indexer.search(query, top_k=1)[0][0].path}"
            
            self.cumulative_anticipation = 0.0  # Reset after full response
        else:
            response = self.prompts['responses']['low_confidence'].format(
                query=query,
                top_doc_path=self.indexer.search(query, top_k=1)[0][0].path,
                top_score=relevance_score,
                threshold=self.attention_threshold
            )
        
        return response, relevance_score, anticipation

    def format_conversation_history(self) -> str:
        formatted_history = ""
        for turn in self.conversation_history:
            formatted_history += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        return formatted_history.strip()

    def load_conversation_history(self):
        if os.path.exists(self.conversation_file):
            with open(self.conversation_file, 'r') as f:
                for line in f:
                    turn = json.loads(line)
                    self.conversation_history.append(turn)

    def save_conversation_turn(self, user_input: str, assistant_response: str):
        turn = {"user": user_input, "assistant": assistant_response}
        self.conversation_history.append(turn)
        
        with open(self.conversation_file, 'a') as f:
            f.write(json.dumps(turn) + '\n')

    def chat_loop(self):
        if not self.indexer.documents:
            print(f"{Fore.RED}No documents have been ingested. The assistant cannot provide informed responses.")
            return

        doc_count = len(self.indexer.documents)
        print(f"{Fore.GREEN}{self.prompts['system']['welcome_message'].format(doc_count=doc_count)}")
        print(f"{Fore.YELLOW}{self.prompts['system']['exit_instruction']}")
        
        while True:
            user_input = input(f"\n{Fore.GREEN}You: ")
            if user_input.lower() == 'exit':
                print(f"{Fore.YELLOW}{self.prompts['system']['goodbye_message']}")
                break
            
            response, relevance_score, anticipation = self.process_query(user_input)
            
            print(f"\n{Fore.YELLOW}Relevance Score: {relevance_score:.2f}")
            print(f"{Fore.YELLOW}Cumulative Anticipation: {anticipation:.2f}")
            print(f"\n{Fore.BLUE}Assistant: {response}")
            
            self.save_conversation_turn(user_input, response)

if __name__ == "__main__":
    framework = AnticipationAttentionFramework()
    
    framework.ingest_documents("documents")
    
    framework.chat_loop()