import json
import asyncio
import openai
import os
import logging
from typing import List, Dict, Any, Tuple
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncMedQAInference:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18", max_concurrent_calls: int = 20, batch_size: int = 50):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_concurrent_calls = max_concurrent_calls
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def _call_openai_api(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {e}")
                return None

    def _extract_question_and_options(self, prompt: str) -> tuple:
        """Extract question and options from the prompt string."""
        # Split the prompt into question and options sections
        parts = prompt.split("Options:")
        if len(parts) != 2:
            return None, None
        
        question = parts[0].replace("Question:", "").strip()
        options_text = parts[1].split("Choose the best answer")[0].strip()
        
        # Parse options
        options = []
        for line in options_text.split("\n"):
            line = line.strip()
            if line and line[0] in "ABCDE" and ". " in line:
                option = line.split(". ", 1)[1].strip()
                options.append(option)
        
        return question, options

    def _extract_answer_from_response(self, response: str) -> str:
        """Extract the answer letter from the response."""
        # Try to find answer pattern like "A. [answer]" or "Answer: A"
        if not response:
            return "Unknown"
            
        # Look for patterns like "A." or "Answer: A" at the start
        match = re.match(r'^([A-D])[\.\s]|^Answer:\s*([A-D])', response)
        if match:
            return match.group(1) or match.group(2)
            
        # Look for the answer letter in the first line
        first_line = response.split('\n')[0]
        if first_line[0] in "ABCD":
            return first_line[0]
            
        return "Unknown"

    async def process_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = question_data["prompt"]
        ground_truth = question_data.get("response", "")
        
        question, options = self._extract_question_and_options(prompt)
        if not question or not options:
            logger.error(f"Failed to parse question and options from prompt: {prompt[:100]}...")
            return None
        
        messages = [
                    {
                        "role": "system",
                        "content": '''You are a medical expert. For each question, choose the single best answer and provide a concise, evidence-based explanation.

                                        Respond in the following format:
                                        <answer_label>. <answer_text>  
                                        Explanation: <brief rationale>  

                                        Example:  
                                        D. Nitrofurantoin  
                                        Explanation: Nitrofurantoin is the best choice for treating urinary tract infections in pregnant women, as it is effective, safe, and well-tolerated, with no risk of teratogenicity in the second trimester.'''
                    },
                    {"role": "user", "content": prompt}
                ]


        response = await self._call_openai_api(messages)
        if response:
            model_answer = self._extract_answer_from_response(response)
            ground_truth_answer = self._extract_answer_from_response(ground_truth)
            
            return {
                "prompt": prompt,
                "model_response": response,
                "model_answer": model_answer,
                "ground_truth": ground_truth,
                "ground_truth_answer": ground_truth_answer,
                "question": question,
                "options": options,
                "is_correct": model_answer == ground_truth_answer
            }
        return None

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy and other metrics."""
        if not results:
            return {"accuracy": 0.0, "exact_match": 0.0}
            
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct", False))
        exact_matches = sum(1 for r in results if r.get("model_response", "").strip() == r.get("ground_truth", "").strip())
        
        return {
            "accuracy": correct / total,
            "exact_match": exact_matches / total,
            "total_questions": total,
            "correct_answers": correct,
            "exact_matches": exact_matches
        }

    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of questions concurrently."""
        tasks = [self.process_single_question(q) for q in batch]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def process_dataset(self, input_file: str, output_file: str, max_samples: int = None):
        try:
            results = []
            processed_count = 0
            current_batch = []
            
            # Read and process the dataset in batches
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_samples and processed_count >= max_samples:
                        break
                        
                    try:
                        question_data = json.loads(line.strip())
                        current_batch.append(question_data)
                        processed_count += 1
                        
                        # Process batch when it reaches batch_size
                        if len(current_batch) >= self.batch_size:
                            batch_results = await self.process_batch(current_batch)
                            results.extend(batch_results)
                            current_batch = []
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON line: {e}")
                        continue
                
                # Process remaining questions
                if current_batch:
                    batch_results = await self.process_batch(current_batch)
                    results.extend(batch_results)

            # Calculate and log metrics
            metrics = self.calculate_metrics(results)
            logger.info(f"Metrics: {metrics}")

            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            # Save metrics
            metrics_file = output_file.replace('.json', '_metrics.json')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Processed {len(results)} questions successfully")

        except Exception as e:
            logger.error(f"Error processing dataset: {e}")

async def main():
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Initialize the inference class with increased concurrency
    inferencer = AsyncMedQAInference(
        api_key=api_key,
        model="gpt-4o-mini-2024-07-18",
        max_concurrent_calls=20,  # Increased from 10 to 20
        batch_size=50  # Process 50 questions at a time
    )

    # Process the dataset
    input_file = "data/original_datasets/test_dataset.json"
    output_file = "data/qwen3/gpt4o_mini_test_predictions.json"
    
    await inferencer.process_dataset(
        input_file=input_file,
        output_file=output_file,
        max_samples=None  # Set to a number to limit samples
    )

if __name__ == "__main__":
    asyncio.run(main()) 