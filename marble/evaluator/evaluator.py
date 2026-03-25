"""
Evaluator module for tracking metrics and evaluating agent performance.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from ruamel.yaml import YAML

from marble.agent import BaseAgent
from marble.environments import BaseEnvironment
from marble.llms.model_prompting import model_prompting
from marble.utils.logger import get_logger


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", text or "").strip()


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_fragment(text: str, expect_array: bool = False) -> str:
    cleaned = _strip_code_fences(_strip_think_blocks(text))
    opener, closer = ("[", "]") if expect_array else ("{", "}")
    start = cleaned.find(opener)
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return cleaned[start : idx + 1]
    return ""


def fill_prompt_template(template: str, values: Dict[str, Any]) -> str:
    """Replace only known placeholders to avoid KeyError from JSON braces."""
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


class Evaluator:
    """
    Evaluator class for tracking metrics like task completion success rate and token consumption.
    """
    def __init__(self, metrics_config: Dict[str, Any]):
        """
        Initialize the Evaluator with the specified metrics.

        Args:
            metrics_config (Dict[str, Any]): Configuration for the metrics to track.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.metrics_config = metrics_config
        self.metrics:Dict[str, Any] = {
            "task_completion": [],
            "token_consumption": [],
            "planning_score": [],
            "communication_score": [],
            "task_evaluation": {},
            "total_milestones": 0,
            "agent_kpis": {},
            "code_quality": {}
        }
        prompts_path = Path(__file__).resolve().parent / "evaluator_prompts.json"
        with prompts_path.open('r', encoding='utf-8') as f:
            self.evaluation_prompts = json.load(f)

        evaluate_llm_config = self.metrics_config.get('evaluate_llm', {})
        self.llm = evaluate_llm_config.get('model', 'gpt-3.5-turbo') if isinstance(evaluate_llm_config, dict) else evaluate_llm_config



    def update(self, environment: BaseEnvironment, agents: List[BaseAgent]) -> None:
        """
        Update the metrics based on the current state of the environment and agents.

        Args:
            environment (BaseEnvironment): The environment instance.
            agents (List[BaseAgent]): List of agent instances.
        """
        # For task completion, check if the environment indicates the task is done
        if environment.is_task_completed():
            self.metrics["task_completion"].append(1)
        else:
            self.metrics["task_completion"].append(0)

        # For token consumption, sum up the tokens used by agents in this iteration
        total_tokens = sum(agent.get_token_usage() for agent in agents)
        self.metrics["token_consumption"].append(total_tokens)

    def evaluate_communication(self, task: str, communications: str) -> None:
        """
        Evaluate communication between agents and update the communication score.

        Args:
            task (str): The task description.
            communications (str): The communication logs between agents.
        """
        # Get the communication prompt
        communication_prompt_template = self.evaluation_prompts["Graph"]["Communication"]["prompt"]
        # Fill in the placeholders {task} and {communications}
        prompt = fill_prompt_template(
            communication_prompt_template,
            {"task": task, "communications": communications},
        )
        # Call the language model
        result = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]
        # Parse the score from result.content
        assert isinstance(result.content, str)
        score = self.parse_score(result.content)
        # Update the metric
        self.metrics["communication_score"].append(score)

    def evaluate_planning(self, summary: str, agent_profiles: str, agent_tasks: str, results: str) -> None:
        """
        Evaluate planning and self-coordination among agents and update the planning score.

        Args:
            summary (str): Last round summary.
            agent_profiles (str): Profiles of agents.
            agent_tasks (str): Tasks assigned to agents.
            results (str): Results of the next round.
        """
        # Get the planning prompt
        planning_prompt_template = self.evaluation_prompts["Graph"]["Planning"]["prompt"]
        # Fill in the placeholders
        prompt = fill_prompt_template(
            planning_prompt_template,
            {
                "summary": summary,
                "agent_profiles": agent_profiles,
                "agent_tasks": agent_tasks,
                "results": results,
            },
        )
        # Call the language model
        result = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]
        # Parse the score from result.content
        assert isinstance(result.content, str)
        score = self.parse_score(result.content)
        # Update the metric
        self.metrics["planning_score"].append(score)

    def evaluate_kpi(self, task: str, agent_results: str) -> None:
        """
        Evaluate milestones achieved and update agent KPIs.

        Args:
            task (str): The task description.
            agent_results (str): The results from the agents.
        """
        # Get the KPI prompt
        MAX_LENGTH = 7200

        if len(agent_results) > MAX_LENGTH:
            agent_results = agent_results[:MAX_LENGTH] + "..."
        kpi_prompt_template = self.evaluation_prompts["Graph"]["KPI"]["prompt"]
        # Fill in the placeholders {task} and {agent_results}
        prompt = fill_prompt_template(
            kpi_prompt_template,
            {"task": task, "agent_results": agent_results},
        )
        # Call the language model
        result = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]
        # Parse the milestones from result.content
        assert isinstance(result.content, str)
        milestones = self.parse_milestones(result.content)
        # Keep only structured milestone objects to avoid runtime errors on noisy LLM outputs.
        milestones = [m for m in milestones if isinstance(m, dict)]
        # Update the metrics
        self.metrics["total_milestones"] += len(milestones)
        for milestone in milestones:
            # Support both legacy and prompt-defined keys.
            agents = milestone.get("contributing_agents") or milestone.get("agents", [])
            for agent_id in agents:
                if agent_id in self.metrics["agent_kpis"]:
                    self.metrics["agent_kpis"][agent_id] += 1
                else:
                    self.metrics["agent_kpis"][agent_id] = 1
        self.logger.debug(f"LLM Response: {result.content}")

    def evaluate_task_research(self, task: str, result: str) -> None:
        """
        Evaluate the final research idea based on innovation, safety, and feasibility.

        Args:
            task (str): The task description.
            result (str): The final research idea.
        """
        # Get the research evaluation prompt
        research_prompt_template = self.evaluation_prompts["research"]["task_evaluation"]["prompt"]
        # Fill in the placeholders {task} and {result}
        prompt = fill_prompt_template(
            research_prompt_template,
            {"task": task, "result": result},
        )
        # Call the language model
        llm_response = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]
        # Parse the ratings from llm_response.content
        assert isinstance(llm_response.content, str)
        ratings = self.parse_research_ratings(llm_response.content)
        # Update the metrics
        if ratings:
            self.metrics["task_evaluation"] = ratings
        else:
            self.logger.error("Failed to parse research ratings.")

    def evaluate_task_world(self, task: str, result: str) -> None:
        """
        Evaluate the final world idea based on Effectiveness of Strategies, Progress and Outcome and Interaction Dynamics

        Args:
            task (str): The task description.
            result (str): The final world idea.
        """
        # change the prompt to evaluate buyer and seller
        # world_prompt_template = self.evaluation_prompts["world"]["task_evaluation"]["buyer_prompt"]
        world_prompt_template = self.evaluation_prompts["world"]["task_evaluation"]["seller_prompt"]
        prompt = fill_prompt_template(
            world_prompt_template,
            {"task": task, "result": result},
        )

        llm_response = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]

        ratings = self.parse_task_world_evaluation(llm_response.content)

        self.metrics["task_evaluation"]["buyer"] = ratings["buyer"]
        self.metrics["task_evaluation"]["seller"] = ratings["seller"]


    def parse_task_world_evaluation(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the JSON ratings from the LLM response for buyer and seller evaluation.
        If the response is not a valid JSON, return default scores of -1.

        Args:
            llm_response (str): The LLM response containing the ratings in JSON format.

        Returns:
            Dict[str, Any]: A dictionary containing parsed ratings for buyer and seller.
        """
        default_ratings = {
            "buyer": {
                "effectiveness_of_strategies": -1,
                "progress_and_outcome": -1,
                "interaction_dynamics": -1
            },
            "seller": {
                "effectiveness_of_strategies": -1,
                "progress_and_outcome": -1,
                "interaction_dynamics": -1
            }
        }

        try:
            match = re.search(r'\{[\s\S]*\}', llm_response)
            if not match:
                return default_ratings

            json_str = match.group(0)

            ratings = json.loads(json_str)

            if "buyer" not in ratings and "seller" not in ratings:
                return default_ratings

            parsed_ratings = {
                "buyer": {
                    "effectiveness_of_strategies": int(ratings["buyer"].get("effectiveness_of_strategies", -1)),
                    "progress_and_outcome": int(ratings["buyer"].get("progress_and_outcome", -1)),
                    "interaction_dynamics": int(ratings["buyer"].get("interaction_dynamics", -1))
                } if "buyer" in ratings else default_ratings["buyer"],
                "seller": {
                    "effectiveness_of_strategies": int(ratings["seller"].get("effectiveness_of_strategies", -1)),
                    "progress_and_outcome": int(ratings["seller"].get("progress_and_outcome", -1)),
                    "interaction_dynamics": int(ratings["seller"].get("interaction_dynamics", -1))
                } if "seller" in ratings else default_ratings["seller"]
            }

            return parsed_ratings

        except (json.JSONDecodeError, KeyError, ValueError):
            return default_ratings  # 解析失败则返回默认评分

    def evaluate_task_db(self, task: str, result: str, labels: List[str], pred_num: int, root_causes: List[str]) -> None:
        """
        Evaluate the final database idea based on Data Quality, Data Security, and Data Privacy.

        Args:
            task (str): The task description.
            result (str): The final root cause analysis.
            labels (List[str]): The list of root cause labels.
            pred_num (int): The number of predicted root causes.
            root_causes (List[str]): The root cause labels.
        """
        # Evaluation will take place separately as it might not follow the
        # requested format
        self.metrics["task_evaluation"] = {
            'root_cause': root_causes,
            'predicted': result,
        }

    def parse_research_ratings(self, assistant_answer: str) -> Dict[str, int]:
        """
        Parse the research ratings from the assistant's answer.

        Args:
            assistant_answer (str): The assistant's answer containing the ratings.

        Returns:
            Dict[str, int]: The parsed ratings.
        """
        try:
            json_str = _extract_json_fragment(assistant_answer, expect_array=False)
            if not json_str:
                self.logger.debug("No JSON found in assistant's answer for research ratings. Trying fallback extraction.")
                # Fallback: Try to extract individual ratings from plain text
                return self._extract_ratings_from_text(assistant_answer)

            ratings = json.loads(json_str)
            if not isinstance(ratings, dict):
                return {}
            # Ensure ratings are integers
            ratings_dict: Dict[str, int] = {k: int(v) for k, v in ratings.items()}
            return ratings_dict
        except (json.JSONDecodeError, ValueError, TypeError):
            self.logger.debug("Failed to parse JSON from assistant's answer for research ratings. Trying fallback extraction.")
            # Fallback: Try to extract individual ratings from plain text
            return self._extract_ratings_from_text(assistant_answer)

    def _extract_ratings_from_text(self, text: str) -> Dict[str, int]:
        """
        Fallback method to extract innovation, safety, and feasibility ratings from plain text.
        Looks for patterns like "innovation: 4", "Innovation (4)", "Innovation is 4", etc.
        
        Args:
            text (str): The text to extract ratings from.
            
        Returns:
            Dict[str, int]: Dictionary with extracted ratings or empty dict if extraction fails.
        """
        ratings = {}
        # Pattern to match: innovation/safety/feasibility followed by various separators and a number 1-5
        # Handles: "innovation: 4", "innovation=4", "innovation is 4", "innovation 4", "innovation (4)"
        patterns = {
            'innovation': r'innovation[\s:=is]+([\(\[]?)(\d)[\)\]]?',
            'safety': r'safety[\s:=is]+([\(\[]?)(\d)[\)\]]?',
            'feasibility': r'feasibility[\s:=is]+([\(\[]?)(\d)[\)\]]?',
        }
        
        text_lower = text.lower()
        for key, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    # Extract the digit (group 2 because group 1 is optional parenthesis)
                    value = int(match.group(2))
                    if 1 <= value <= 5:
                        ratings[key] = value
                        self.logger.debug(f"Extracted {key}: {value} from text")
                except (ValueError, IndexError):
                    pass
        
        if ratings:
            self.logger.info(f"Successfully extracted ratings from text via fallback: {ratings}")
            return ratings

        # Secondary fallback for minimal outputs like "4 5 4" or "4,5,4".
        nums = [int(n) for n in re.findall(r"\b[1-5]\b", text)]
        if len(nums) >= 3:
            fallback = {
                "innovation": nums[0],
                "safety": nums[1],
                "feasibility": nums[2],
            }
            self.logger.info(
                f"Extracted research ratings from positional fallback: {fallback}"
            )
            return fallback

        self.logger.debug("Could not extract any valid ratings from text")
        
        return ratings

    def parse_score(self, assistant_answer: str) -> int:
        """
        Parse the score from the assistant's answer based on the strict JSON format requirement.

        Args:
            assistant_answer (str): The assistant's answer containing the score.

        Returns:
            int: The parsed score. Returns 3 (default score) if parsing fails.
        """
        try:
            content = _strip_code_fences(_strip_think_blocks(assistant_answer))

            json_str = _extract_json_fragment(content, expect_array=False)
            if json_str:
                try:
                    rating_data = json.loads(json_str)
                    if isinstance(rating_data, dict) and "rating" in rating_data:
                        score = int(rating_data["rating"])
                        if 1 <= score <= 5:
                            self.logger.debug(f"Successfully parsed score: {score}")
                            return score
                        else:
                            self.logger.warning(f"Score {score} out of valid range (1-5)")
                except json.JSONDecodeError:
                    self.logger.debug("Failed to parse JSON from response")
                except (ValueError, TypeError):
                    self.logger.warning("Invalid score format in JSON")
                except KeyError:
                    self.logger.warning("Missing 'rating' key in JSON response")

            # If JSON parsing fails, try to find a single digit between 1-5
            numbers = re.findall(r'\b[1-5]\b', content)
            if numbers:
                score = int(numbers[0])
                self.logger.debug(f"Found score using regex: {score}")
                return score

            # If all parsing attempts fail, return default score
            self.logger.warning("No valid score found, using default score (3)")
            return 3

        except Exception as e:
            self.logger.error(f"Unexpected error parsing score: {e}")
            return 3

    def finalize(self) -> None:
        """
        Finalize the evaluation, compute final metrics, and log or save the results.
        """
        total_tasks = len(self.metrics["task_completion"])
        tasks_completed = sum(self.metrics["task_completion"])
        success_rate = tasks_completed / total_tasks if total_tasks > 0 else 0

        total_tokens = sum(self.metrics["token_consumption"])
        avg_tokens_per_iteration = total_tokens / total_tasks if total_tasks > 0 else 0

        self.logger.info(f"Task Completion Success Rate: {success_rate * 100:.2f}%")
        self.logger.info(f"Total Token Consumption: {total_tokens}")
        self.logger.info(f"Average Tokens per Iteration: {avg_tokens_per_iteration}")

        # Additional metrics can be computed and logged here

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the computed metrics.

        Returns:
            Dict[str, Any]: The computed metrics.
        """
        return {
            "success_rate": sum(self.metrics["task_completion"]) / len(self.metrics["task_completion"]) if self.metrics["task_completion"] else 0,
            "total_tokens": sum(self.metrics["token_consumption"]),
            "avg_tokens_per_iteration": sum(self.metrics["token_consumption"]) / len(self.metrics["token_consumption"]) if self.metrics["token_consumption"] else 0
        }

    def parse_milestones(self, assistant_answer: str) -> List[Dict[str, Any]]:
        """
        Parse the milestones from the assistant's answer.

        Args:
            assistant_answer (str): The assistant's answer containing the milestones.

        Returns:
            List[Dict[str, Any]]: The list of milestones.
        """
        # Preprocess to handle escaped newlines and unnecessary symbols
        try:
            normalized_answer = _strip_code_fences(_strip_think_blocks(assistant_answer))

            # First try to parse a JSON list directly.
            list_json = _extract_json_fragment(normalized_answer, expect_array=True)
            if list_json:
                milestones = json.loads(list_json)
                if isinstance(milestones, list):
                    return [m for m in milestones if isinstance(m, dict)]

            # Then try object payloads like {"milestones": [...]}
            obj_json = _extract_json_fragment(normalized_answer, expect_array=False)
            if obj_json:
                payload = json.loads(obj_json)
                if isinstance(payload, dict) and isinstance(payload.get("milestones"), list):
                    return [m for m in payload["milestones"] if isinstance(m, dict)]

            self.logger.debug("No milestone JSON payload found in assistant answer.")
            return []
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse JSON from assistant's answer.")
            return []

        except Exception as e:
            self.logger.warning(f"Error processing milestones: {e}")
            return []

    def parse_code_quality_scores(self, assistant_answer: str) -> Dict[str, int]:
        """
        Parse the code quality scores from the assistant's answer.

        Args:
            assistant_answer (str): The assistant's answer containing the code quality scores.

        Returns:
            Dict[str, int]: The parsed code quality scores.
        """
        try:
            content = assistant_answer.strip()

            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                scores = json.loads(json_str)

                required_keys = {
                    "instruction_following",
                    "executability",
                    "consistency",
                    "quality"
                }

                if all(key in scores for key in required_keys):
                    validated_scores = {}
                    for key in required_keys:
                        try:
                            score = int(scores[key])
                            if 1 <= score <= 5:
                                validated_scores[key] = score
                            else:
                                validated_scores[key] = 1  # 默认最低分
                        except (ValueError, TypeError):
                            validated_scores[key] = 1  # 默认最低分
                    return validated_scores

            self.logger.error("Invalid code quality scores format in response")
            return {
                "instruction_following": 1,
                "executability": 1,
                "consistency": 1,
                "quality": 1
            }

        except Exception as e:
            self.logger.error(f"Error parsing code quality scores: {e}")
            return {
                "instruction_following": 1,
                "executability": 1,
                "consistency": 1,
                "quality": 1
            }

    def evaluate_code_quality(self, task: str, code_result: str) -> None:
        """
        Evaluate the code quality based on stricter criteria.
        """
        try:
            config_path = "marble/configs/coding_config/coding_config.yaml"
            if not os.path.exists(config_path):
                self.logger.error("Config file not found")
                return

            yaml = YAML()
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f)

            full_task_description = config['task']['content']

            requirements_start = "1. Implementation requirements:\n"
            requirements_end = "\n\n2. Project structure:"
            requirements = full_task_description[
                full_task_description.find(requirements_start) + len(requirements_start):
                full_task_description.find(requirements_end)
            ].strip()

            solution_path = "marble/workspace/solution.py"
            solution_content = ""
            if os.path.exists(solution_path):
                with open(solution_path, 'r', encoding='utf-8') as f:
                    solution_content = f.read()

            code_quality_prompt_template = """
                    [Context]
                    **Task Description:**
                    {task_description}

                    **Implementation Requirements:**
                    {requirements}

                    **Current Solution:**
                    {solution}

                    [System]
                    This evaluation requires strict scoring and deduction. The scores should not be generous, and deductions should be applied for every issue found.

                    ### **Evaluation Criteria**
                    1. **Instruction-Following:** Does the code fulfill all the requirements of the task? Deduct points for unmet or partially met requirement from the task instructions.
                    2. **Executability:** Is the code syntactically correct and executable? Deduct points for any syntax errors, missing imports, or runtime errors.
                    3. **Consistency:** Is the code consistent in variable naming, formatting, and logic? Deduct points for inconsistent variable naming, formatting issues, or contradictory logic.
                    4. **Quality:** Is the code well-documented, clear, and modular? Deduct points for poor documentation, unclear logic, or lack of modular design.

                    ### **Scoring**
                    - **1 point:** Below Average - Significant issues that need addressing.
                    - **2 points:** Average - Noticeable areas for improvement.
                    - **3 points:** Good - Minor issues or improvements needed.
                    - **4 points:** Excellent - Almost or fully satisfies the criterion.
                    - **5 points:** Legendary - Flawless, perfectly satisfies the criterion, and exceeds expectations.

                    **Do not give the same scores for different criteria, such as 3 for instruction-following, 3 for executability, 3 for consistency, and 3 for quality.**
                    If you give the same scores for the 4 criteria, you have to add or deduct 1 point randomly for one or two criteria.

                    ### **Question**
                    Based on the criteria, evaluate the code and output the scores for each criterion in the following JSON format:
                    {{
                        "instruction_following": score,
                        "executability": score,
                        "consistency": score,
                        "quality": score
                    }}
            """

            # Fill in the template
            prompt = code_quality_prompt_template.format(
                task_description=full_task_description,
                requirements=requirements,
                solution=solution_content
            )

            # Call the LLM
            response = model_prompting(
                llm_model=self.llm,
                messages=[{"role": "user", "content": prompt}],
                return_num=1,
                max_token_num=4096,
                temperature=0.0,
                top_p=None,
                stream=None,
            )[0]

            scores = self.parse_code_quality_scores(response.content)

            if scores:
                self.metrics["code_quality"] = scores
                self.logger.info(f"Code quality evaluated strictly: {scores}")
            else:
                self.logger.error("Failed to parse code quality scores.")
                self.metrics["code_quality"] = {
                    "instruction_following": 1,
                    "executability": 1,
                    "consistency": 1,
                    "quality": 1
                }
            self.logger.debug(f"LLM Response: {response.content}")

        except Exception as e:
            self.logger.error(f"Error in code quality evaluation: {e}")
            self.metrics["code_quality"] = {
                "instruction_following": 1,
                "executability": 1,
                "consistency": 1,
                "quality": 1
            }
