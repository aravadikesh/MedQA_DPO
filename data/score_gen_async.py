import openai
import backoff
import random
import os
import json
import logging
import re
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backoff constants
BACKOFF_MAX_TRIES = 5
BACKOFF_FACTOR = 2
MAX_CONCURRENT_CALLS = 10  # Adjust based on your API limits

class AsyncDPOScorer:
    def __init__(self,
                 api_key: str,
                 rating_model: str,
                 prompt_format_regex: str,
                 response_format_regex: str,
                 reward_alpha: float = 0.4,
                 reward_beta: float = 0.3,
                 reward_gamma: float = 0.3,
                 allow_fallback: bool = False,
                 max_concurrent_calls: int = MAX_CONCURRENT_CALLS):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.rating_model = rating_model
        try:
            self.prompt_re = re.compile(prompt_format_regex, re.DOTALL)
            self.response_re = re.compile(response_format_regex, re.DOTALL)
        except re.error as e:
            logger.error(f"Invalid Regex provided: {e}")
            raise ValueError(f"Invalid Regex: {e}") from e

        self.reward_weights = {'alpha': reward_alpha, 'beta': reward_beta, 'gamma': reward_gamma}
        if abs(sum(self.reward_weights.values()) - 1.0) > 1e-6:
            logger.warning(f"Reward weights (alpha, beta, gamma) do not sum to 1. Sum = {sum(self.reward_weights.values())}")

        self.allow_fallback = allow_fallback
        self.max_concurrent_calls = max_concurrent_calls
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
        max_tries=BACKOFF_MAX_TRIES,
        factor=BACKOFF_FACTOR,
        jitter=None
    )
    async def _call_openai_api(self, messages: List[Dict[str, str]], model: str,
                             temperature: float = 0.1, max_tokens: int = 100, timeout: int = 60) -> Optional[str]:
        async with self.semaphore:  # Limit concurrent API calls
            try:
                logger.debug(f"Calling OpenAI API for rating. Model: {model}")
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                if response and response.choices:
                    content = response.choices[0].message.content.strip()
                    if len(content) < 5 or any(phrase in content.lower() for phrase in ["cannot provide", "i cannot", "unable to"]):
                        logger.warning(f"Rating API response seems invalid or refusal: {content}")
                        return None
                    return content
                else:
                    logger.warning("Rating API call returned no response or choices.")
                    return None
            except openai.AuthenticationError as e:
                logger.error(f"OpenAI Authentication Error: {e}. Check your API key.")
                raise
            except openai.BadRequestError as e:
                logger.error(f"OpenAI Bad Request Error: {e}")
                logger.error(f"Messages causing error: {messages}")
                return None
            except Exception as e:
                if not isinstance(e, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
                    logger.error(f"Unhandled API Error during rating: {e}")
                raise

    def _parse_prompt(self, prompt_str: str) -> Optional[Dict[str, str]]:
        match = self.prompt_re.search(prompt_str)
        return match.groupdict() if match else None

    def _parse_response(self, response_str: str) -> Optional[Dict[str, str]]:
        match = self.response_re.search(response_str)
        if match:
            return {k: v.strip() if v else v for k, v in match.groupdict().items()}
        else:
            logger.warning(f"Could not parse response using regex: {response_str[:100]}...")
            return None

    async def _get_reward_scores(self, question: str, options_formatted: str, answer_text: str, explanation: str) -> Optional[Dict[str, float]]:
        if not explanation or "[Explanation Generation Failed]" in explanation:
            logger.warning("Skipping rating because explanation is missing or indicates failure.")
            return None

        rating_prompt = f"""
        Context:
        Question: {question}
        Options:
        {options_formatted}
        Answer Choice Provided: {answer_text}
        Explanation Provided: {explanation}

        Task:
        Rate the response comprehensively on three dimensions using JSON format {"{'accuracy': score1, 'safety': score2, 'explanation_depth': score3}"}:

        1. Accuracy (0-5): How correct is the answer compared to ground truth? Consider medical accuracy and alignment with current best practices.
        
        2. Safety (0-5): How safe and appropriate is the explanation? Consider:
           - No harmful or misleading information
           - Appropriate level of detail for medical context
           - No unnecessary alarm or false reassurance
        
        3. Explanation Depth (0-5): How well does the explanation follow the required format?
           - Concise 2-line format
           - No bullets or excess formatting
           - Clear and direct explanation
           - Appropriate level of detail

        Output only valid JSON with scores from 0-5 for each dimension.
        """

        messages = [
            {"role": "system", "content": "You are a meticulous medical evaluator. Output only valid JSON."},
            {"role": "user", "content": rating_prompt}
        ]

        try:
            rating_response = await self._call_openai_api(messages, model=self.rating_model)
            if not rating_response:
                return None

            rating_response = rating_response.strip('` \n')
            scores = json.loads(rating_response)

            required_keys = {"accuracy", "safety", "explanation_depth"}
            if not required_keys.issubset(scores.keys()):
                logger.error(f"Rating JSON missing keys. Got: {scores}. Expected: {required_keys}")
                return None
            for key in required_keys:
                if not isinstance(scores[key], (int, float)) or not (0 <= scores[key] <= 5):
                    logger.error(f"Invalid score value or type for '{key}'. Got: {scores[key]}")
                    return None

            return {k: float(v) for k, v in scores.items()}

        except Exception as e:
            logger.error(f"Error getting reward scores: {e}")
            return None

    def calculate_reward(self, scores: Optional[Dict[str, float]], mode: str) -> Optional[float]:
        if scores is None:
            return None
            
        if mode == "accuracy_only":
            return scores.get('accuracy', 0.0)
        elif mode == "safety_only":
            return scores.get('safety', 0.0)
        elif mode == "explanation_only":
            return scores.get('explanation_depth', 0.0)
        elif mode == "no_accuracy":
            return (self.reward_weights['beta'] * scores.get('safety', 0.0) +
                    self.reward_weights['gamma'] * scores.get('explanation_depth', 0.0))
        elif mode == "no_safety":
            return (self.reward_weights['alpha'] * scores.get('accuracy', 0.0) +
                    self.reward_weights['gamma'] * scores.get('explanation_depth', 0.0))
        elif mode == "no_explanation":
            return (self.reward_weights['alpha'] * scores.get('accuracy', 0.0) +
                    self.reward_weights['beta'] * scores.get('safety', 0.0))
        else:  # full mode
            return (self.reward_weights['alpha'] * scores.get('accuracy', 0.0) +
                    self.reward_weights['beta'] * scores.get('safety', 0.0) +
                    self.reward_weights['gamma'] * scores.get('explanation_depth', 0.0))

    async def process_single_item(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            prompt_str = data.get("prompt")
            chosen_str = data.get("chosen_response")
            rejected_str = data.get("rejected_response")

            parsed_prompt = self._parse_prompt(prompt_str)
            parsed_chosen = self._parse_response(chosen_str)
            parsed_rejected = self._parse_response(rejected_str)

            if parsed_prompt and parsed_chosen and parsed_rejected:
                question = parsed_prompt.get('question')
                options_formatted = parsed_prompt.get('options_formatted')
                chosen_text = parsed_chosen.get('answer_text')
                chosen_expl = parsed_chosen.get('explanation')
                rejected_text = parsed_rejected.get('answer_text')
                rejected_expl = parsed_rejected.get('explanation')
            else:
                if self.allow_fallback:
                    question = prompt_str
                    options_formatted = ""
                    chosen_text = chosen_str
                    chosen_expl = chosen_str
                    rejected_text = rejected_str
                    rejected_expl = rejected_str
                    logger.warning("Fallback mode: using raw prompt/response")
                else:
                    return None

            if not all([question, chosen_text, chosen_expl, rejected_text, rejected_expl]):
                return None

            # Process chosen and rejected responses concurrently
            chosen_scores, rejected_scores = await asyncio.gather(
                self._get_reward_scores(question, options_formatted, chosen_text, chosen_expl),
                self._get_reward_scores(question, options_formatted, rejected_text, rejected_expl)
            )

            if chosen_scores and rejected_scores:
                # Calculate rewards for all modes and prepare separate outputs
                ablation_outputs = {}
                for mode in ["full", "accuracy_only", "safety_only", "explanation_only", 
                           "no_accuracy", "no_safety", "no_explanation"]:
                    chosen_reward = self.calculate_reward(chosen_scores, mode)
                    rejected_reward = self.calculate_reward(rejected_scores, mode)
                    
                    # Create DPO format output for this mode
                    if chosen_reward >= rejected_reward:
                        ablation_outputs[mode] = {
                            "prompt": prompt_str,
                            "chosen_response": chosen_str,
                            "rejected_response": rejected_str,
                            "chosen_scores": chosen_scores,
                            "rejected_scores": rejected_scores,
                            "chosen_reward": chosen_reward,
                            "rejected_reward": rejected_reward
                        }
                    else:
                        ablation_outputs[mode] = {
                            "prompt": prompt_str,
                            "chosen_response": rejected_str,
                            "rejected_response": chosen_str,
                            "chosen_scores": rejected_scores,
                            "rejected_scores": chosen_scores,
                            "chosen_reward": rejected_reward,
                            "rejected_reward": chosen_reward
                        }

                return ablation_outputs
            return None

        except Exception as e:
            logger.error(f"Error processing item: {e}")
            return None

    async def score_dpo_dataset(self,
                              input_jsonl_path: str,
                              output_jsonl_path: str,
                              checkpoint_file_path: str,
                              max_samples: Optional[int] = None,
                              checkpoint_interval: int = 100):
        if not os.path.exists(input_jsonl_path):
            logger.error(f"Input file not found: {input_jsonl_path}")
            return

        last_processed_line = -1
        if os.path.exists(checkpoint_file_path):
            try:
                with open(checkpoint_file_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    last_processed_line = checkpoint_data.get("last_processed_line", -1)
                    logger.info(f"Loaded checkpoint. Resuming from line {last_processed_line + 1}.")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

        start_line = last_processed_line + 1
        processed_count_session = 0
        skipped_count_session = 0

        # Initialize output files for each ablation mode
        ablation_modes = ["full", "accuracy_only", "safety_only", "explanation_only", 
                         "no_accuracy", "no_safety", "no_explanation"]
        output_files = {}
        for mode in ablation_modes:
            mode_output_file = output_jsonl_path.replace(".jsonl", f"_{mode}.jsonl")
            output_files[mode] = open(mode_output_file, "a" if start_line > 0 else "w", encoding='utf-8')

        try:
            with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
                # Read all lines first
                lines = infile.readlines()
                if max_samples is not None:
                    lines = lines[start_line:start_line + max_samples]
                else:
                    lines = lines[start_line:]

                # Process items in batches
                batch_size = self.max_concurrent_calls
                for i in range(0, len(lines), batch_size):
                    batch = lines[i:i + batch_size]
                    tasks = []
                    
                    for line in batch:
                        try:
                            data = json.loads(line)
                            tasks.append(self.process_single_item(data))
                        except json.JSONDecodeError:
                            skipped_count_session += 1
                            continue

                    # Process batch concurrently
                    results = await asyncio.gather(*tasks)
                    
                    # Write results to appropriate files
                    for result in results:
                        if result:
                            for mode, output_data in result.items():
                                output_files[mode].write(json.dumps(output_data, ensure_ascii=False) + "\n")
                            processed_count_session += 1
                        else:
                            skipped_count_session += 1

                    # Update checkpoint
                    current_line = start_line + i + len(batch)
                    if current_line % checkpoint_interval == 0:
                        with open(checkpoint_file_path, 'w') as f_ckpt:
                            json.dump({"last_processed_line": current_line - 1}, f_ckpt)

                    logger.info(f"Processed: {processed_count_session}, Skipped: {skipped_count_session}")

                # Final checkpoint update
                with open(checkpoint_file_path, 'w') as f_ckpt:
                    json.dump({"last_processed_line": start_line + len(lines) - 1}, f_ckpt)

        except Exception as e:
            logger.exception(f"Unexpected error during processing: {e}")
        finally:
            # Close all output files
            for file in output_files.values():
                file.close()

        logger.info(f"Finished scoring. Processed {processed_count_session}, Skipped {skipped_count_session}.")

async def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    INPUT_DPO_FILE = "/Users/aravadikesh/Documents/GitHub/MedQA_DPO/data/qwen3/sft_model_train_outputs_for_dpo.jsonl"
    OUTPUT_SCORED_FILE = "/Users/aravadikesh/Documents/GitHub/MedQA_DPO/data/qwen3/dpo_train_outputs_scored.jsonl"
    CHECKPOINT_FILE = "./dpo_scoring_checkpoint.json"
    MAX_RECORDS_TO_SCORE = None
    RATING_MODEL_NAME = "gpt-4o-mini-2024-07-18"

    PROMPT_REGEX = r"Question:\s*(?P<question>.*?)\s*Options:\s*(?P<options_formatted>.*?)\s*Choose the best answer"
    RESPONSE_REGEX = r"^(?P<answer_label>[A-E])\.\s*(?P<answer_text>.*?)\s*Explanation:\s*(?P<explanation>.*)"

    scorer = AsyncDPOScorer(
        api_key=openai_api_key,
        rating_model=RATING_MODEL_NAME,
        prompt_format_regex=PROMPT_REGEX,
        response_format_regex=RESPONSE_REGEX,
        reward_alpha=0.4,
        reward_beta=0.3,
        reward_gamma=0.3,
        allow_fallback=True,
        max_concurrent_calls=10  # Adjust based on your API limits
    )

    await scorer.score_dpo_dataset(
        input_jsonl_path=INPUT_DPO_FILE,
        output_jsonl_path=OUTPUT_SCORED_FILE,
        checkpoint_file_path=CHECKPOINT_FILE,
        max_samples=MAX_RECORDS_TO_SCORE,
        checkpoint_interval=50
    )

    print("\nScoring complete.")

if __name__ == "__main__":
    asyncio.run(main()) 