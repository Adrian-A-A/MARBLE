"""
Base agent module.
"""

import json
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from litellm.utils import token_counter

from marble.environments import BaseEnvironment, CodingEnvironment, WebEnvironment
from marble.llms.model_prompting import model_prompting
from marble.memory import BaseMemory, SharedMemory
from marble.utils.logger import get_logger

EnvType = Union[BaseEnvironment, WebEnvironment, CodingEnvironment]
AgentType = TypeVar("AgentType", bound="BaseAgent")


def convert_to_str(result: Any) -> str:
    if isinstance(result, bool):
        return str(result)  # Turn into 'True' or 'False'
    elif isinstance(result, dict):
        return json.dumps(result)  # dict to JSON string
    else:
        return str(result)  # handle other types


def _normalize_action_name(action_name: str) -> str:
    return "".join(ch for ch in action_name.lower() if ch.isalnum())


def _safe_json_loads_object(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

def _extract_int_from_text(text: str, keys: List[str]) -> Optional[int]:
    for key in keys:
        pattern = rf"(?i){re.escape(key)}\s*[:=]\s*\$?(-?\d+)"
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    money_match = re.search(r"\$(\d{3,7})", text)
    if money_match:
        return int(money_match.group(1))
    return None

def _extract_str_from_text(text: str, keys: List[str]) -> Optional[str]:
    for key in keys:
        pattern = rf"(?is){re.escape(key)}\s*[:=]\s*\"([^\"]+)\""
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        pattern = rf"(?im)^\s*{re.escape(key)}\s*[:=]\s*(.+?)\s*$"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None

def _infer_action_args_from_text(action_name: str, raw_text: str) -> Dict[str, Any]:
    inferred: Dict[str, Any] = {}
    lowered = action_name.lower()

    if lowered == "offer_price":
        price = _extract_int_from_text(raw_text, ["price", "offered_price"]) 
        if price is not None:
            inferred["price"] = price
        reason = _extract_str_from_text(raw_text, ["reason"]) 
        if reason:
            inferred["reason"] = reason

    elif lowered == "reject_and_counter":
        counter = _extract_int_from_text(raw_text, ["counter_price", "price"]) 
        if counter is not None:
            inferred["counter_price"] = counter
        reason = _extract_str_from_text(raw_text, ["reason"]) 
        if reason:
            inferred["reason"] = reason

    elif lowered == "inquire_intentions":
        question = _extract_str_from_text(raw_text, ["question"]) 
        if not question:
            match = re.search(r"(?im)^\s*inquire intentions\s*:\s*(.+)$", raw_text)
            if match:
                question = match.group(1).strip()
        if question:
            inferred["question"] = question

    elif lowered == "provide_information":
        info_type = _extract_str_from_text(raw_text, ["info_type", "type"]) 
        details = _extract_str_from_text(raw_text, ["details", "detail", "information"]) 
        if info_type:
            inferred["info_type"] = info_type
        if details:
            inferred["details"] = details

    elif lowered == "new_communication_session":
        target_agent_id = _extract_str_from_text(raw_text, ["target_agent_id"]) 
        message = _extract_str_from_text(raw_text, ["message"]) 
        if target_agent_id:
            inferred["target_agent_id"] = target_agent_id
        if message:
            inferred["message"] = message

    return inferred


def _ensure_required_action_args(
    action_name: str, args: Dict[str, Any], result_content: str
) -> Dict[str, Any]:
    ensured = dict(args)
    lowered = action_name.lower()

    if lowered == "offer_price" and "price" not in ensured:
        inferred_price = _extract_int_from_text(result_content, ["price", "offered_price"])
        ensured["price"] = inferred_price if inferred_price is not None else 13500

    if lowered == "reject_and_counter" and "counter_price" not in ensured:
        inferred_counter = _extract_int_from_text(result_content, ["counter_price", "price"])
        ensured["counter_price"] = inferred_counter if inferred_counter is not None else 14000

    if lowered == "inquire_intentions" and "question" not in ensured:
        inferred_question = _extract_str_from_text(result_content, ["question"])
        ensured["question"] = (
            inferred_question
            if inferred_question
            else "What price range are you expecting so we can move toward agreement?"
        )

    if lowered == "provide_information":
        if "info_type" not in ensured:
            ensured["info_type"] = "Market Comparison"
        if "details" not in ensured:
            brief = (result_content or "").strip().replace("\n", " ")
            ensured["details"] = brief[:240] if brief else "Providing additional negotiation context."

    return ensured


class BaseAgent:
    """
    Base class for all agents.
    """

    def __init__(
        self,
        config: Dict[str, Union[Any, Dict[str, Any]]],
        env: EnvType,
        shared_memory: Union[SharedMemory, None] = None,
        model: str = "openai/qwen2.5:0.5b",
    ):
        """
        Initialize the agent.

        Args:
            config (dict): Configuration for the agent.
            env (EnvType): Environment for the agent.
            shared_memory (BaseMemory, optional): Shared memory instance.
        """
        agent_id = config.get("agent_id")
        if isinstance(model, dict):
            self.llm = model.get("model", "openai/qwen2.5:0.5b")
        else:
            self.llm = model
        assert isinstance(agent_id, str), "agent_id must be a string."
        assert env is not None, "agent must has an environment."
        self.env: EnvType = env
        self.actions: List[str] = []
        self.agent_id: str = agent_id
        self.agent_graph = None
        self.profile = config.get("profile", "")
        self.system_message = (
            f'You are "{self.agent_id}": "{self.profile}"\n'
            f"As a role-playing agent, you embody a dynamic character with unique traits, motivations, and skills. "
            f"Your goal is to engage not only with users but also with other agents in the environment. "
            f"Collaborate, compete, or form alliances as you navigate through immersive storytelling and challenges. "
            f"Interact meaningfully with fellow agents, contributing to the evolving narrative and responding creatively "
            f"to their actions. Maintain consistency with your character's background and personality, and be prepared to adapt "
            f"to the evolving dynamics of the scenario. Remember, your responses should enhance the experience and encourage "
            f"user engagement while enriching interactions with other agents."
        )
        self.memory = BaseMemory()
        self.shared_memory = SharedMemory()
        self.relationships: Dict[str, str] = {}
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Agent '{self.agent_id}' initialized.")
        self.token_usage = 0
        self.task_history: List[str] = []
        self.msg_box: Dict[str, Dict[str, List[Tuple[int, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.children: List[BaseAgent] = []
        self.parent: Optional[BaseAgent] = None
        self.FORWARD_TO = 0
        self.RECV_FROM = 1
        self.session_id: str = ""
        self.strategy = config.get("strategy", "default")
        self.reasoning_prompts = {
            "default": "",
            "cot": (
                "Think through this step by step:\n"
                "1. What is the main objective of this task?\n"
                "2. What information and resources do I have available?\n"
                "3. What approach would be most effective?\n"
                "4. What specific actions should I take?\n"
            ),
            "reflexion": (
                "Follow the reflection process:\n"
                "1. Initial thoughts on the task\n"
                "2. Analysis of available options\n"
                "3. Potential challenges and solutions\n"
                "4. Final approach decision\n"
            ),
            "react": (
                "Follow the ReAct framework:\n"
                "Observation: What do I notice about this task?\n"
                "Thought: What are my considerations?\n"
                "Action: What specific action should I take?\n"
                "Result: What do I expect to achieve?\n"
            ),
        }

    def set_agent_graph(self, agent_graph: Any) -> None:
        self.agent_graph = agent_graph

    def perceive(self, state: Any) -> Any:
        """
        Agent perceives the environment state.

        Args:
            state (Any): The current state of the environment.

        Returns:
            Any: Processed perception data.
        """
        return state.get("task_description", "")

    def act(self, task: str) -> Any:
        """
        Agent decides on an action to take.

        Args:
            task (str): The task to perform.

        Returns:
            Any: The action decided by the agent.
        """
        self.task_history.append(task)
        self.logger.info(f"Agent '{self.agent_id}' acting on task '{task}'.")
        tools = [
            self.env.action_handler_descriptions[name]
            for name in self.env.action_handler_descriptions
        ]
        available_agents: Dict[str, Any] = {}
        assert (
            self.agent_graph is not None
        ), "Agent graph is not set. Please set the agent graph using the set_agent_graph method first."
        for agent_id_1, agent_id_2, relationship in self.agent_graph.relationships:
            if agent_id_1 != self.agent_id and agent_id_2 != self.agent_id:
                continue
            else:
                if agent_id_1 == self.agent_id:
                    profile = self.agent_graph.agents[agent_id_2].get_profile()
                    agent_id = agent_id_2
                elif agent_id_2 == self.agent_id:
                    profile = self.agent_graph.agents[agent_id_1].get_profile()
                    agent_id = agent_id_1
                available_agents[agent_id] = {
                    "profile": profile,
                    "role": f"{agent_id_1} {relationship} {agent_id_2}",
                }
        self.available_agents = available_agents
        # Create the enum description with detailed information about each agent
        agent_descriptions = [
            f"{agent_id} ({info['role']} - {info['profile']})"
            for agent_id, info in available_agents.items()
        ]
        # Add communicate_to function description
        new_communication_session_description = {
            "type": "function",
            "function": {
                "name": "new_communication_session",
                "description": "Send a message to a specific target agent based on existing relationships, and begin communication",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent_id": {
                            "type": "string",
                            "description": "The ID of the target agent to communicate with. Available agents:\n"
                            + "\n".join([f"- {desc}" for desc in agent_descriptions]),
                            "enum": list(
                                self.relationships.keys()
                            ),  # Dynamically list available target agents
                        },
                        "message": {
                            "type": "string",
                            "description": "The initial message to send to the target agent",
                        },
                    },
                    "required": ["target_agent_id", "message"],
                    "additionalProperties": False,
                },
            },
        }
        tools.append(new_communication_session_description)
        reasoning_prompt = self.reasoning_prompts.get(self.strategy, "")
        self.logger.info(
            f"Agent {self.agent_id} using {self.strategy} strategy with prompt:\n{reasoning_prompt}"
        )

        act_task = (
            f"You are {self.agent_id}: {self.profile}\n"
            f"{reasoning_prompt}\n"  # 使用已经获取的 reasoning_prompt
            f"This is your task: {task}\n"
            f"These are the ids and profiles of other agents you can interact with:\n"
            f"{agent_descriptions}"
            f"But you do not have to communcate with other agents.\n"
            f"You can also solve the task by calling other functions to solve it by yourself.\n"
            f"These are your memory: {self.memory.get_memory_str()}\n"
        )
        self.logger.info(f"Complete prompt for agent {self.agent_id}:\n{act_task}")

        if len(tools) == 0:
            result = model_prompting(
                llm_model=self.llm,
                messages=[{"role": "user", "content": act_task}],
                return_num=1,
                max_token_num=512,
                temperature=0.0,
                top_p=None,
                stream=None,
            )[0]
        else:
            result = model_prompting(
                llm_model=self.llm,
                messages=[{"role": "user", "content": act_task}],
                return_num=1,
                max_token_num=512,
                temperature=0.0,
                top_p=None,
                stream=None,
                tools=tools,
                tool_choice="auto",
            )[0]
        messages = [
            {"role": "usr", "content": act_task},
            {"role": "sys", "content": result.content},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        communication = None
        result_from_function_str = None
        if result.tool_calls:
            function_call = result.tool_calls[0]
            function_name = function_call.function.name
            assert function_name is not None
            raw_function_args = function_call.function.arguments
            parsed_function_args = _safe_json_loads_object(raw_function_args)
            if parsed_function_args is None:
                self.logger.debug(
                    "Tool-call arguments were invalid JSON; attempting text fallback args parsing."
                )
                mapped_action_name, mapped_action_args = self._extract_text_action_fallback(
                    result.content or ""
                )
                if mapped_action_name == function_name and mapped_action_args:
                    function_args = mapped_action_args
                else:
                    function_args = _infer_action_args_from_text(
                        function_name,
                        f"{raw_function_args}\n{result.content or ''}",
                    )
            else:
                function_args = parsed_function_args
            function_args = _ensure_required_action_args(
                function_name,
                function_args,
                result.content or "",
            )
            if function_name != "new_communication_session":
                try:
                    result_from_function = self.env.apply_action(
                        agent_id=self.agent_id,
                        action_name=function_name,
                        arguments=function_args,
                    )
                    result_from_function_str = convert_to_str(result_from_function)
                except Exception as apply_exc:
                    self.logger.warning(
                        f"Tool-call action execution failed for '{function_name}': {apply_exc}"
                    )
                    result_from_function = {
                        "success": False,
                        "error": str(apply_exc),
                        "action_name": function_name,
                        "arguments": function_args,
                    }
                    result_from_function_str = convert_to_str(result_from_function)
            else:  # function_name == "new_communication_session"
                if "target_agent_id" not in function_args or "message" not in function_args:
                    self.logger.warning(
                        "new_communication_session tool-call missing required arguments."
                    )
                    result_from_function = {
                        "success": False,
                        "error": "missing required arguments",
                        "arguments": function_args,
                    }
                    result_from_function_str = convert_to_str(result_from_function)
                    communication = None
                else:
                    self.session_id = uuid.uuid4()  # new session id
                    target_agent_id = function_args["target_agent_id"]
                    message = function_args["message"]
                    result_from_function = self._handle_new_communication_session(
                        target_agent_id=target_agent_id,
                        message=message,
                        session_id=self.session_id,
                        task=task,
                        turns=5,
                    )
                    result_from_function_str = convert_to_str(result_from_function)
                    communication = result_from_function.get("full_chat_history", None)
            self.memory.update(
                self.agent_id,
                {
                    "type": "action_function_call",
                    "action_name": function_name,
                    "args": function_args,
                    "result": result_from_function,
                },
            )

            self.logger.info(
                f"Agent '{self.agent_id}' called '{function_name}' with args '{function_args}'."
            )
            self.logger.info(
                f"Agent '{self.agent_id}' obtained result '{result_from_function}'."
            )

        else:
            result_content = result.content if result.content else ""
            mapped_action_name, mapped_action_args = self._extract_text_action_fallback(
                result_content
            )
            if mapped_action_name:
                try:
                    result_from_function = self.env.apply_action(
                        agent_id=self.agent_id,
                        action_name=mapped_action_name,
                        arguments=mapped_action_args,
                    )
                    result_from_function_str = convert_to_str(result_from_function)
                    self.memory.update(
                        self.agent_id,
                        {
                            "type": "action_function_call",
                            "action_name": mapped_action_name,
                            "args": mapped_action_args,
                            "result": result_from_function,
                            "source": "text_fallback",
                        },
                    )
                    self.logger.info(
                        f"Agent '{self.agent_id}' used text fallback action '{mapped_action_name}' with args '{mapped_action_args}'."
                    )
                except Exception as fallback_exc:
                    self.logger.warning(
                        "Text fallback action parsing succeeded but apply_action failed: "
                        f"{fallback_exc}"
                    )
                    self.memory.update(
                        self.agent_id,
                        {"type": "action_response", "result": result.content},
                    )
                    self.logger.info(
                        f"Agent '{self.agent_id}' acted with result '{result}'."
                    )
            else:
                self.memory.update(
                    self.agent_id, {"type": "action_response", "result": result.content}
                )
                self.logger.info(f"Agent '{self.agent_id}' acted with result '{result}'.")
        result_content = result.content if result.content else ""
        self.token_usage += self._calculate_token_usage(task, result_content)
        output = "Result from the model:" + result_content + "\n"
        if result_from_function_str:
            output += "Result from the function:" + result_from_function_str
        return output, communication

    def _extract_text_action_fallback(
        self, result_content: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parse plain-text or JSON assistant output into an environment action.
        This is used when tool_calls are unavailable/rejected by backend.
        """
        if not result_content.strip():
            return None, {}

        action_candidates: Dict[str, str] = {}
        for action_name, description in self.env.action_handler_descriptions.items():
            action_candidates[_normalize_action_name(action_name)] = action_name
            if isinstance(description, dict):
                function_data = description.get("function", {})
                function_name = function_data.get("name")
                if isinstance(function_name, str):
                    action_candidates[_normalize_action_name(function_name)] = action_name

        action_value: Optional[str] = None
        action_args: Dict[str, Any] = {}

        # Try parsing a JSON object from the output first.
        try:
            stripped = result_content.strip()
            if stripped.startswith("```"):
                stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
                stripped = re.sub(r"\s*```$", "", stripped)
            json_start = stripped.find("{")
            json_end = stripped.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                payload = json.loads(stripped[json_start : json_end + 1])
                if isinstance(payload, dict):
                    action_value = (
                        payload.get("action")
                        or payload.get("action_name")
                        or payload.get("tool")
                        or payload.get("function")
                    )
                    args_value = (
                        payload.get("action_parameters")
                        or payload.get("parameters")
                        or payload.get("args")
                    )
                    if isinstance(args_value, dict):
                        action_args = args_value
        except Exception:
            pass

        # Fallback to section parsing if JSON is absent.
        if not isinstance(action_value, str) or not action_value.strip():
            action_match = re.search(r"(?im)^\s*action\s*:\s*(.+?)\s*$", result_content)
            if action_match:
                action_value = action_match.group(1).strip().strip('"')

        # Fallback: detect natural-language intent such as
        # "I will call `get_environment_info`" or "use placeBlock".
        if not isinstance(action_value, str) or not action_value.strip():
            prose_call_match = re.search(
                r"(?is)(?:call|use|invoke|run|execute)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
                result_content,
            )
            if prose_call_match:
                action_value = prose_call_match.group(1).strip()

        # Fallback: find explicit backticked action names from available actions.
        if not isinstance(action_value, str) or not action_value.strip():
            for candidate_norm, candidate_action_name in action_candidates.items():
                if re.search(
                    rf"(?is)`{re.escape(candidate_action_name)}`|\b{re.escape(candidate_action_name)}\b",
                    result_content,
                ):
                    action_value = candidate_action_name
                    break

        if not action_args:
            params_match = re.search(
                r"(?is)action\s*parameters\s*:\s*(\{.*?\})",
                result_content,
            )
            if params_match:
                try:
                    parsed_args = json.loads(params_match.group(1))
                    if isinstance(parsed_args, dict):
                        action_args = parsed_args
                except Exception:
                    action_args = {}

        if not isinstance(action_value, str) or not action_value.strip():
            return None, {}

        normalized = _normalize_action_name(action_value)
        mapped_action_name = action_candidates.get(normalized)
        if not mapped_action_name:
            for candidate_norm, candidate_action_name in action_candidates.items():
                if normalized in candidate_norm or candidate_norm in normalized:
                    mapped_action_name = candidate_action_name
                    break

        if not mapped_action_name:
            return None, {}

        if not action_args:
            action_args = _infer_action_args_from_text(mapped_action_name, result_content)

        return mapped_action_name, action_args

    def _calculate_token_usage(self, task: Optional[str], result: Optional[str]) -> int:
        """
        Calculate token usage based on input and output lengths.

        Args:
            task (Optional[str]): The input task.
            result (Optional[str]): The output result.

        Returns:
            int: The number of tokens used.
        """
        safe_task = task or ""
        safe_result = result or ""
        token_count = (len(safe_task) + len(safe_result)) // 4
        return token_count

    def get_token_usage(self) -> int:
        """
        Get the total token usage by the agent.

        Returns:
            int: The total tokens used by the agent.
        """
        return self.token_usage

    def send_message(
        self, session_id: str, target_agent: AgentType, message: str
    ) -> None:
        """Send a message to the target agent within the specified session.

        Args:
            session_id (str): The identifier for the current session.
            target_agent (BaseAgent): The agent to whom the message is being sent.
            message (str): The message content to be sent.
        """
        self.msg_box[session_id][target_agent.agent_id].append(
            (self.FORWARD_TO, message)
        )

        self.logger.info(
            f"Agent {self.agent_id} sent message to {target_agent.agent_id}: {message}"
        )

        target_agent.receive_message(session_id, self, message)

    def receive_message(
        self, session_id: str, from_agent: AgentType, message: str
    ) -> None:
        """Receive a message from another agent within the specified session.

        Args:
            session_id (str): The identifier for the current session.
            from_agent (BaseAgent): The agent sending the message.
            message (str): The content of the received message.
        """
        self.session_id = session_id

        # Store the received message in the message box for the sending agent.
        self.msg_box[session_id][from_agent.agent_id].append((self.RECV_FROM, message))
        self.logger.info(
            f"Agent {self.agent_id} received message from {from_agent.agent_id}: {message[:10]}..."
        )

    def seralize_message(self, session_id: str = "") -> str:
        seralized_msg = ""

        # Check if session_id is provided
        if session_id:
            # Serialize messages for a specific session
            session_ids = [session_id] if session_id in self.msg_box else []
        else:
            # Serialize messages for all sessions
            session_ids = list(self.msg_box.keys())

        for sid in session_ids:
            seralized_msg += f"In Session {sid} \n"
            session_msg = self.msg_box[sid]

            for target_agent_id in session_msg:
                msg_list = session_msg[target_agent_id]
                for direction, msg_content in msg_list:
                    if direction == self.FORWARD_TO:
                        seralized_msg += f"From {self.agent_id} to {target_agent_id}: "
                    else:
                        seralized_msg += f"From {target_agent_id} to {self.agent_id}: "
                    seralized_msg += msg_content + "\n"

        return seralized_msg

    def get_profile(self) -> Union[str, Any]:
        """
        Get the agent's profile.

        Returns:
            str: The agent's profile.
        """
        return self.profile

    def _handle_new_communication_session(
        self,
        target_agent_id: str,
        message: str,
        session_id: str,
        task: str,
        turns: int = 5,
    ) -> Dict[str, Any]:
        """
        Handler for the new communication function. This will start a session using a random uuid
        and arrage communication between two agents until matter is resolved.

        Args:
            target_agent_id (str): The ID of the target agent
            message (str): The message to send
            session_id (str): Session ID of chat between two agents
            task (str): Task of source agent
            turns (int): Maximum number of allowed turns of communication

        Returns:
            Dict[str, Any]: Result of the communication attempt
        """
        initial_communication = self._handle_communicate_to(
            target_agent_id, message, session_id
        )
        if not initial_communication["success"]:
            return initial_communication
        assert (
            self.agent_graph is not None
        ), "Agent graph is not set. Please set the agent graph using the set_agent_graph method first."
        agents = [self.agent_graph.agents.get(target_agent_id), self]
        for t in range(turns):
            session_current_agent = agents[t % 2]
            session_current_agent_id = session_current_agent.agent_id
            session_other_agent = agents[(t + 1) % 2]
            session_other_agent_id = session_other_agent.agent_id

            agent_descriptions = [
                f"{session_other_agent_id} (session_other_agent.profile)"
            ]
            communicate_to_description = {
                "type": "function",
                "function": {
                    "name": "communicate_to",
                    "description": "Send a message to a specific target agent:"
                    + "\n".join([f"- {desc}" for desc in agent_descriptions]),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The initial message to send to the target agent",
                            },
                        },
                        "required": ["target_agent_id", "message"],
                        "additionalProperties": False,
                    },
                },
            }

            communicate_task = (
                f"These are your memory: {session_current_agent.memory}\n"
                f"The task is: {task}. \n"
                f"Please respond to {session_other_agent_id}({session_other_agent.profile}). \n"
                f"Your previous chat history: {session_current_agent.seralize_message(session_id=self.session_id)}.\n"
                f"You should answer to this question {session_current_agent.msg_box[self.session_id][session_other_agent_id][-1][1]} using your memory, and other relevant context. \n"
                f"Return <end-of-session> if you cannot answer using information you have right now. \n"
                f"You are talking to {session_other_agent_id}. You cannot talk with anyone else.\n"
                f"From {session_current_agent_id} to {session_other_agent_id}:"
            )
            result = model_prompting(
                llm_model=self.llm,
                messages=[
                    {"role": "system", "content": session_current_agent.system_message},
                    {"role": "user", "content": communicate_task},
                ],
                return_num=1,
                max_token_num=512,
                temperature=0.0,
                top_p=None,
                stream=None,
                tools=[communicate_to_description],
                tool_choice="required",
            )[0]
            messages = [
                {"role": "system", "content": session_current_agent.system_message},
                {"role": "user", "content": communicate_task},
                {"role": "system", "content": result.content},
            ]
            self.token_usage += token_counter(model=self.llm, messages=messages)
            if result.tool_calls:
                function_call = result.tool_calls[0]
                function_name = function_call.function.name
                assert function_name is not None
                parsed_function_args = _safe_json_loads_object(
                    function_call.function.arguments
                )
                if parsed_function_args is None:
                    self.logger.warning(
                        "Session tool-call arguments were invalid JSON; trying inferred args."
                    )
                    function_args = _infer_action_args_from_text(
                        function_name,
                        f"{function_call.function.arguments}\n{result.content or ''}",
                    )
                    if not function_args:
                        continue
                else:
                    function_args = parsed_function_args
                if function_name == "communicate_to":
                    if "message" not in function_args:
                        self.logger.warning(
                            "communicate_to tool-call missing required message argument; skipping this turn."
                        )
                        continue
                    message = function_args["message"]
                    print(message)
                    session_current_agent._handle_communicate_to(
                        target_agent_id=session_other_agent_id,
                        message=message,
                        session_id=session_current_agent.session_id,
                    )
                    if "<end-of-session>" in message:
                        break

        # summarize chat history
        system_message_summary = (
            "You are an advanced summarizer agent designed to condense and clarify the history of conversations between multiple agents. "
            "Your task is to analyze dialogues from various participants and generate a cohesive summary that captures the key points, themes, and decisions made throughout the interactions.\n\n"
            "Your primary objectives are:\n\n"
            "1. Contextual Analysis: Carefully review the entire conversation history to understand the context, including the roles of different agents and the progression of discussions.\n\n"
            "2. Identify Key Themes: Extract the main themes, topics, and significant moments in the dialogue, noting any recurring issues or points of contention.\n\n"
            "3. Summarize Conversations: Create a clear and concise summary that outlines the conversation's flow, important exchanges, decisions made, and any action items that emerged. Ensure that the summary reflects the contributions of each agent without losing the overall narrative.\n\n"
            "4. Highlight Outcomes: Emphasize any conclusions reached or actions agreed upon by the agents, providing a sense of closure to the summarized conversation.\n\n"
            "5. Engage with User Input: If the user has specific interests or focuses within the conversation, inquire to tailor the summary accordingly, ensuring it meets their needs.\n\n"
            "When composing the summary, maintain clarity, coherence, and logical organization. Your goal is to provide a comprehensive yet succinct overview that enables users to understand the essence of the multi-agent dialogue at a glance."
        )
        summary_task = (
            f"These are an chat history: {session_current_agent.seralize_message(session_id=self.session_id)}\n"
            f"Please summarize information in the chat history relevant to the task: {task}."
        )
        result = model_prompting(
            llm_model=self.llm,
            messages=[
                {"role": "system", "content": system_message_summary},
                {"role": "user", "content": summary_task},
            ],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]
        messages = [
            {"role": "system", "content": system_message_summary},
            {"role": "user", "content": summary_task},
            {"role": "system", "content": result.content},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        self.memory.update(
            self.agent_id,
            {
                "type": "action_communicate",
                "action_name": "communicate_to",
                # "args": function_args,
                "result": result.content if result.content else "",
            },
        )
        return {
            "success": True,
            "message": f"Successfully completed session {session_id}",
            "full_chat_history": session_current_agent.seralize_message(
                session_id=self.session_id
            ),
            "session_id": result.content if result.content else "",
        }

    def _handle_communicate_to(
        self, target_agent_id: str, message: str, session_id: str
    ) -> Dict[str, Any]:
        """
        Handler for the communicate_to function.

        Args:
            target_agent_id (str): The ID of the target agent
            message (str): The message to send
            session_id (str): Session ID of chat between two agents

        Returns:
            Dict[str, Any]: Result of the communication attempt
        """
        old_session_id = self.session_id
        try:
            self.session_id = session_id
            linked_by_graph = False
            assert (
                self.agent_graph is not None
            ), "Agent graph is not set. Please set the agent graph using the set_agent_graph method first."
            for a1_id, a2_id, rel in self.agent_graph.relationships:
                if a1_id == self.agent_id or a2_id == self.agent_id:
                    linked_by_graph = True
                    break

            if not self.agent_graph or not linked_by_graph:
                return {
                    "success": False,
                    "error": f"No relationship found with agent {target_agent_id}",
                }

            target_agent = self.agent_graph.agents.get(target_agent_id)
            if not target_agent:
                return {
                    "success": False,
                    "error": f"Target agent {target_agent_id} not found in agent graph",
                }

            # Send the message using the existing send_message method
            self.send_message(self.session_id, target_agent, message)

            return {
                "success": True,
                "message": f"Successfully sent message to agent {target_agent_id}",
                "session_id": session_id,
            }

        except Exception as e:
            self.session_id = old_session_id
            return {"success": False, "error": f"Error sending message: {str(e)}"}

    def plan_task(self) -> Optional[str]:
        """
        Plan the next task based on the original tasks input, the agent's memory, task history, and its profile/persona.

        Returns:
            str: The next task description.
        """
        self.logger.info(f"Agent '{self.agent_id}' is planning the next task.")

        # Retrieve all memory entries for this agent
        memory_str = self.memory.get_memory_str()
        task_history_str = ", ".join(self.task_history)

        # Incorporate agent's profile/persona in decision making
        persona = self.get_profile()

        # Use memory entries, persona, and task history to determine the next task
        next_task = model_prompting(
            llm_model=self.llm,
            messages=[
                {
                    "role": "user",
                    "content": f"Agent '{self.agent_id}' should prioritize tasks that align with their role: {persona}. Based on the task history: {task_history_str}, and memory: {memory_str}, what should be the next task?",
                }
            ],
            return_num=1,
            max_token_num=512,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0].content
        messages = [
            {
                "role": "user",
                "content": f"Agent '{self.agent_id}' should prioritize tasks that align with their role: {persona}. Based on the task history: {task_history_str}, and memory: {memory_str}, what should be the next task?",
            },
            {"role": "system", "content": next_task},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        self.logger.info(
            f"Agent '{self.agent_id}' plans next task based on persona: {next_task}"
        )

        return next_task

    def _is_task_completed(self, result: Any) -> bool:
        """
        Determine if the task is completed based on the result of the last action.

        Args:
            result (Any): The result from the last action.

        Returns:
            bool: True if task is completed, False otherwise.
        """
        # Placeholder logic; implement actual completion criteria
        if isinstance(result, str):
            return "completed" in result.lower()
        return False

    def _define_next_task_based_on_result(self, result: Any) -> str:
        """
        Define the next task based on the result of the last action.

        Args:
            result (Any): The result from the last action.

        Returns:
            str: The next task description.
        """
        # Placeholder logic; implement actual task definition
        if isinstance(result, str):
            if "error" in result.lower():
                return "Retry the previous action."
            else:
                return "Proceed to the next step based on the result."
        return "Analyze the result and determine the next task."

    def _is_response_satisfactory(self, response: Any) -> bool:
        """
        Determine if the response is satisfactory.

        Args:
            response (Any): The response from the last action.

        Returns:
            bool: True if satisfactory, False otherwise.
        """
        # Placeholder logic; implement actual response evaluation
        if isinstance(response, str):
            return "success" in response.lower()
        return False

    def _define_next_task_based_on_response(self, response: Any) -> str:
        """
        Define the next task based on the response of the last action.

        Args:
            response (Any): The response from the last action.

        Returns:
            str: The next task description.
        """
        # Placeholder logic; implement actual task definition
        if isinstance(response, str):
            if "need more information" in response.lower():
                return "Gather additional information required to proceed."
            else:
                return "Address the issues identified in the response."
        return "Review the response and determine the next steps."

    def plan_tasks_for_children(self, task: str) -> Dict[str, Any]:
        """
        Plan tasks for children agents based on the given task and children's profiles.
        """
        self.logger.info(f"Agent '{self.agent_id}' is planning tasks for children.")
        children_profiles = {
            child.agent_id: child.get_profile() for child in self.children
        }
        prompt = (
            f"You are agent '{self.agent_id}'. Based on the overall task:\n{task}\n\n"
            f"And your children's profiles:\n"
        )
        for child_id, profile in children_profiles.items():
            prompt += f"- {child_id}: {profile}\n"
        prompt += "\nAssign specific tasks to your children agents to help accomplish the overall task. Provide the assignments in JSON format:\n\n"
        prompt += "{\n"
        '  "child_agent_id": "Task description",\n'
        '  "another_child_agent_id": "Task description"\n'
        "}\n"
        response = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "system", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.7,
            top_p=1.0,
        )[0]
        messages = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": response.content},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        try:
            tasks_for_children: Dict[str, Any] = json.loads(
                response.content if response.content else "{}"
            )
            self.logger.info(
                f"Agent '{self.agent_id}' assigned tasks to children: {tasks_for_children}"
            )
            return tasks_for_children
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse tasks for children: {e}")
            return {}

    def summarize_results(
        self, children_results: Dict[str, Any], own_result: Any
    ) -> Any:
        """
        Summarize the results from children agents and own result.
        """
        self.logger.info(f"Agent '{self.agent_id}' is summarizing results.")
        summary = self.process_children_results(children_results)
        summary += f"\nOwn result: {own_result}"
        return summary

    def process_children_results(self, children_results: Dict[str, Any]) -> str:
        """
        Process the results from children agents using model prompting.
        """
        prompt = "Summarize the results from children agents:\n"
        for agent_id, result in children_results.items():
            prompt += f"- Agent '{agent_id}': {result}\n"
        response = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "system", "content": prompt}],
            return_num=1,
            max_token_num=512,
            temperature=0.7,
            top_p=1.0,
        )[0]
        summary = response.content if response.content else ""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": summary},
        ]
        self.token_usage += token_counter(model=self.llm, messages=messages)
        return summary

    def plan_next_agent(
        self, result: Any, agent_profiles: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Choose the next agent to pass the task to and provide a planning task, based on the result and profiles of other agents.

        Args:
            result (Any): The result from the agent's action.
            agent_profiles (Dict[str, Dict[str, Any]]): Profiles of all other agents.

        Returns:
            Tuple[Optional[str], Optional[str]]: The agent_id of the next agent and the planning task, or (None, None) if no suitable agent is found.
        """
        self.logger.info(f"Agent '{self.agent_id}' is planning the next step.")

        # Prepare the prompt for the LLM
        prompt = (
            f"As Agent '{self.agent_id}' with profile: {self.profile}, "
            f"you have completed your part of the task with the result:\n{result}\n\n"
            "Here are the profiles of other available agents:\n"
        )
        for agent_id, profile_info in agent_profiles.items():
            if agent_id != self.agent_id:  # Exclude self
                prompt += f"- Agent ID: {agent_id}\n"
                prompt += f"  Profile: {profile_info['profile']}\n"
        prompt += (
            "\nBased on the result and the agent profiles provided, select the most suitable agent to continue the task and provide a brief plan for the next agent to execute. "
            "Respond in the following format:\n"
            '{"agent_id": "<next_agent_id>", "planning_task": "<description of the next planning task>"}\n'
            "You must follow the json format or the system will crash as we fail to interpret the response."
        )

        # Use the LLM to select the next agent and create a planning task
        response = model_prompting(
            llm_model=self.llm,
            messages=[{"role": "system", "content": prompt}],
            return_num=1,
            max_token_num=256,
            temperature=0.7,
            top_p=1.0,
        )[0].content or ""
        self.token_usage += token_counter(
            model=self.llm,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "system", "content": response},
            ],
        )
        # Parse the response to extract the agent ID and planning task
        next_agent_id: Optional[str] = None
        planning_task: Optional[str] = None

        try:
            assert isinstance(response, str)
            response = response.strip()
            if not response:
                raise ValueError("Empty planner response")

            cleaned_response = re.sub(r"(?is)<think>.*?</think>", "", response).strip()
            if cleaned_response.startswith("```") and cleaned_response.endswith("```"):
                cleaned_response = re.sub(r"^```(?:json)?\s*", "", cleaned_response)
                cleaned_response = re.sub(r"\s*```$", "", cleaned_response)
                cleaned_response = cleaned_response.strip()

            # Accept either pure JSON or text containing a JSON object.
            response_data = _safe_json_loads_object(cleaned_response)
            if response_data is None:
                start = cleaned_response.find("{")
                end = cleaned_response.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    response_data = {}
                else:
                    response_data = _safe_json_loads_object(cleaned_response[start : end + 1]) or {}

            if not response_data:
                # Fallback for key-value text formats.
                agent_match = re.search(r'(?im)\bagent[_\s-]*id\b\s*[:=]\s*["\']?([\w-]+)', cleaned_response)
                task_match = re.search(r'(?is)\bplanning[_\s-]*task\b\s*[:=]\s*["\']?(.+?)(?:$|\n\w+\s*:)', cleaned_response)
                if agent_match:
                    response_data["agent_id"] = agent_match.group(1).strip()
                if task_match:
                    response_data["planning_task"] = task_match.group(1).strip().strip('"\'')

            next_agent_id = response_data.get("agent_id")
            planning_task = response_data.get("planning_task")
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            self.logger.debug(
                f"Agent '{self.agent_id}' received an invalid response format from the LLM."
            )

        if next_agent_id in agent_profiles and next_agent_id != self.agent_id:
            self.logger.info(
                f"Agent '{self.agent_id}' selected '{next_agent_id}' as the next agent with plan: '{planning_task}'."
            )
            return next_agent_id, planning_task
        else:
            self.logger.debug(
                f"Agent '{self.agent_id}' did not select a valid next agent."
            )
            return None, None
