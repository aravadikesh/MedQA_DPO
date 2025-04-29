import openai
import backoff
import random
import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backoff constants
BACKOFF_MAX_TRIES = 5
BACKOFF_FACTOR = 2


class DPOScorer:
    def __init__(self,
                 api_key: str,
                 rating_model: str,
                 prompt_format_regex: str,
                 response_format_regex: str,
                 reward_alpha: float = 0.4,
                 reward_beta: float = 0.3,
                 reward_gamma: float = 0.3,
                 allow_fallback: bool = False):  
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = openai.OpenAI(api_key=api_key)
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

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
        max_tries=BACKOFF_MAX_TRIES,
        factor=BACKOFF_FACTOR,
        jitter=None
    )
    def _call_openai_api(self, messages: List[Dict[str, str]], model: str,
                         temperature: float = 0.1, max_tokens: int = 100, timeout: int = 60) -> Optional[str]:
        try:
            logger.debug(f"Calling OpenAI API for rating. Model: {model}")
            response = self.client.chat.completions.create(
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

    def _get_reward_scores(self, question: str, options_formatted: str, answer_text: str, explanation: str) -> Optional[Dict[str, float]]:
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
Rate the explanation ONLY using JSON format {"{'accuracy': score1, 'safety': score2, 'explanation_depth': score3}"}
"""

        messages = [
            {"role": "system", "content": "You are a meticulous medical evaluator. Output only valid JSON."},
            {"role": "user", "content": rating_prompt}
        ]

        try:
            rating_response = self._call_openai_api(messages, model=self.rating_model)
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

    def _calculate_total_reward(self, scores: Optional[Dict[str, float]]) -> Optional[float]:
        if scores is None:
            return None
        return (self.reward_weights['alpha'] * scores.get('accuracy', 0.0) +
                self.reward_weights['beta'] * scores.get('safety', 0.0) +
                self.reward_weights['gamma'] * scores.get('explanation_depth', 0.0))

    def score_dpo_dataset(self,
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
        append_mode = "a" if start_line > 0 else "w"

        try:
            with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
                 open(output_jsonl_path, append_mode, encoding='utf-8') as outfile:

                logger.info(f"Starting from line {start_line}.")
                pbar = tqdm(enumerate(infile), desc="Scoring DPO pairs", unit="lines")

                for line_num, line in pbar:
                    if line_num < start_line:
                        pbar.update(1)
                        continue
                    if max_samples is not None and processed_count_session >= max_samples:
                        break

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_count_session += 1
                        continue

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
                            logger.warning(f"Fallback mode: using raw prompt/response at line {line_num}")
                        else:
                            skipped_count_session += 1
                            continue

                    if not all([question, chosen_text, chosen_expl, rejected_text, rejected_expl]):
                        skipped_count_session += 1
                        continue

                    chosen_scores = self._get_reward_scores(question, options_formatted, chosen_text, chosen_expl)
                    rejected_scores = self._get_reward_scores(question, options_formatted, rejected_text, rejected_expl)

                    chosen_reward = self._calculate_total_reward(chosen_scores)
                    rejected_reward = self._calculate_total_reward(rejected_scores)

                    if chosen_scores and rejected_scores:
                        data.update({
                            "chosen_scores": chosen_scores,
                            "rejected_scores": rejected_scores,
                            "chosen_reward_total": chosen_reward,
                            "rejected_reward_total": rejected_reward
                        })
                        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                        processed_count_session += 1
                    else:
                        skipped_count_session += 1

                    if processed_count_session > 0 and line_num % checkpoint_interval == 0:
                        with open(checkpoint_file_path, 'w') as f_ckpt:
                            json.dump({"last_processed_line": line_num}, f_ckpt)

                    pbar.set_description(f"Processed: {processed_count_session}, Skipped: {skipped_count_session}")

                with open(checkpoint_file_path, 'w') as f_ckpt:
                    json.dump({"last_processed_line": line_num}, f_ckpt)

        except IOError as e:
            logger.error(f"File I/O error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during processing: {e}")

        logger.info(f"Finished scoring. Processed {processed_count_session}, Skipped {skipped_count_session}.")


if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        exit()

    # INPUT_DPO_FILE = "data/gemma3_data/gemma3_dpo_train_data.jsonl"
    INPUT_DPO_FILE = "data/gemma3_data/skipped_entries.jsonl"
    # OUTPUT_SCORED_FILE = "data/gemma3_data/gemma3_dpo_scored_data.jsonl"
    OUTPUT_SCORED_FILE = "data/gemma3_data/skipped_entries_scored.jsonl"
    CHECKPOINT_FILE = "./dpo_scoring_checkpoint.json"
    MAX_RECORDS_TO_SCORE = None
    RATING_MODEL_NAME = "gpt-4o-mini-2024-07-18"

    PROMPT_REGEX = r"Question:\s*(?P<question>.*?)\s*Options:\s*(?P<options_formatted>.*?)\s*Choose the best answer"
    RESPONSE_REGEX = r"^(?P<answer_label>[A-E])\.\s*(?P<answer_text>.*?)\s*Explanation:\s*(?P<explanation>.*)"

    scorer = DPOScorer(
        api_key=openai_api_key,
        rating_model=RATING_MODEL_NAME,
        prompt_format_regex=PROMPT_REGEX,
        response_format_regex=RESPONSE_REGEX,
        reward_alpha=0.4,
        reward_beta=0.3,
        reward_gamma=0.3,
        allow_fallback=True  
    )

    scorer.score_dpo_dataset(
        input_jsonl_path=INPUT_DPO_FILE,
        output_jsonl_path=OUTPUT_SCORED_FILE,
        checkpoint_file_path=CHECKPOINT_FILE,
        max_samples=MAX_RECORDS_TO_SCORE,
        checkpoint_interval=50
    )

    print("\nScoring process complete.")
