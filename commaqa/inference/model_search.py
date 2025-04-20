import copy
import heapq
import json
import logging


class ParticipantModel:
    """Base class for all participant models. Each model must implement query()."""
    def query(self, state, debug=False):
        raise NotImplementedError("Implement this for controller integration")

    def return_model_calls(self):
        raise NotImplementedError("Implement to track model call counts")


class ModelController:
    """Controls flow between different models using a state graph."""
    def __init__(self, model_list, data_class):
        if "start_state" not in model_list or "end_state" not in model_list:
            raise ValueError("Missing start/end states")
        self.model_list = model_list
        self.data_class = data_class

    def execute(self, state, debug=False):
        """Execute the next model in the state sequence."""
        if state.next not in self.model_list:
            logging.error(f"Invalid next state: {state.next}")
            return []
        
        model_func = self.model_list[state.next]
        model_output = model_func(state, debug=debug)
        return [model_output] if not isinstance(model_output, list) else model_output

    def init_data(self, data_instance):
        """Initialize data object for processing."""
        return self.data_class(data_instance)

    @property
    def start_state(self):
        return self.model_list["start_state"]

    @property
    def end_state(self):
        return self.model_list["end_state"]


class SearchState:
    """Tracks search state including data, score, and next command."""
    def __init__(self, json_data, command, score=0.0):
        self._data = json_data
        self._score = score
        self._next = command

    def copy(self):
        """Create deep copy of the state."""
        return SearchState(
            copy.deepcopy(self._data),
            copy.deepcopy(self._next),
            copy.deepcopy(self._score)
        )
    
    # Required for heapq operations
    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    # Property accessors
    @property
    def data(self): return self._data
    @property
    def score(self): return self._score
    @property
    def next(self): return self._next
    @next.setter
    def next(self, value): self._next = value
    @data.setter
    def data(self, value): self._data = value


class QuestionSearchBase:
    """Base class for question answering systems."""
    def __init__(self, model_controller):
        self.controller = model_controller

    def find_answer_decomp(self, json_input, debug=False):
        raise NotImplementedError("Implement search logic")

    def return_qid_prediction(self, example, override_answer_by=None, debug=False, silent=False):
        """Process example and return (qid, answer, reasoning_chain)."""
        final_state, _ = self.find_answer_decomp(example, debug)
        
        if not final_state:
            if not silent:
                print(f"{example['question']} FAILED!")
            return (example["qid"], "", f"\n{example['qid']}\n{example['question']}")

        data = final_state._data
        chain = f"\n{example['qid']}\n{example['question']}\n{data.get_printable_reasoning_chain()}\nS: {final_state._score}"

        answer = data.get(override_answer_by, "") if override_answer_by else data.get_last_answer()
        
        try:
            json_answer = json.loads(answer)
            answer = json_answer if isinstance(json_answer, (list, str)) else answer
        except ValueError:
            pass

        return (example["qid"], answer, chain)


class BestFirstDecomposer(QuestionSearchBase):
    """Best-first search implementation for question decomposition."""
    def find_answer_decomp(self, json_input, debug=False):
        initial_state = SearchState(
            self.controller.init_data(json_input),
            self.controller.start_state
        )

        heap = [initial_state]
        
        while heap:
            current_state = heapq.heappop(heap)
            
            # Termination condition
            if current_state.next == self.controller.end_state:
                if current_state.data.has_tasks():
                    new_task = current_state.data.pop_task()
                    new_state = current_state.copy()
                    if new_task.task_question:
                        new_state.data.add_qgen(new_task.task_question)
                    new_state.next = new_task.task_participant
                    heapq.heappush(heap, new_state)
                    continue
                return current_state, heap

            # Execute next command
            for new_state in self.controller.execute(current_state, debug):
                heapq.heappush(heap, new_state)

        return None, []  # Search failed