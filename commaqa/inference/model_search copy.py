import copy
import heapq
import json
import logging


from commaqa.inference.data_instances import BasicDataInstance
from log.logging_config import logger, LYELLOW, RESET


class ParticipantModel(object):
    """Base model in this case for coordinating different models. Provides a general
    class to structure all contributing models (in this case, by defining a single
    function `query`, which is the single method that is called for each model).

    """

    def query(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        """
        raise NotImplementedError("Must implement to work inside of controller!")

    def return_model_calls(self):
        """
        :return: a dict of <model_name, number of calls> made by this participant
        """
        raise NotImplementedError("Must implement to work inside of controller!")


class ModelController(object):
    """This class is a `ModelController` that takes multiple (arbitrary)
    models and a control specification of how to interface the different
    models (which can be thought of as a kind of state graph). For example

    """

    def __init__(self, model_list, data_class=BasicDataInstance):
        """Create an instance of a ComplexModel

        :param model_list: a list of models with identifiers and
          control flow.
        :type model_list: dict
        """
        if "start_state" not in model_list:
            raise ValueError("Must specify start state")
        if "end_state" not in model_list:
            raise ValueError("Must specify end state")
        self.model_list = model_list
        self.data_class = data_class

    def execute(self, state, debug=False):
        if state.next not in self.model_list:
            self.logger.error(f"Can not handle next state: {LYELLOW}{state.next}{RESET}")
            return []
        try:
            model_func = self.model_list[state.next]
            model_output = model_func(state, debug=debug)  # model query
            if not isinstance(model_output, list):
                return [model_output]
            return model_output  # List of SearchState
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise ValueError(f"Error caught during model execution: {e}")

    def init_data(self, data_instance):
        """Create an initialized version of the data object
        that will get through around.

        :param data_instance: any arbitrary piece of data.
        :rtype: self.data_class
        """
        return self.data_class(data_instance)

    @property
    def start_state(self):
        return self.model_list["start_state"]

    @property
    def end_state(self):
        return self.model_list["end_state"]

    @property
    def logger(self):
        """Returns a logger instance"""
        level = ".".join([__name__, type(self).__name__])
        return logging.getLogger(level)


# utility class for controlling and recording search state


class SearchState(object):
    """Tracks and records the state of a given search."""

    def __init__(self, json_data, command, score=0.0):
        """Keep track of different stages in the state

        :param json_data: some basic, json represntation of data
        """
        self._data = json_data
        self._score = score
        self._next = command

    def copy(self):
        """Does a deep copy of the state

        :returns: new search state
        """
        new_data = copy.deepcopy(self._data)
        new_score = copy.deepcopy(self._score)
        new_next = copy.deepcopy(self._next)

        return SearchState(new_data, new_next, new_score)

    # important to implement to work
    # with the heap datastructures
    def __lt__(self, other):
        if self.score < other.score:
            return True
        return False

    def __eq__(self, other):
        if self.score == other.score:
            return True
        return False

    @property
    def data(self):
        return self._data

    @property
    def score(self):
        return self._score

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, value):
        self._next = value

    @data.setter
    def data(self, value):
        self._data = value


class QuestionSearchBase(object):
    def __init__(self, model_controller):
        """Create a `QuestionDecomposer instance`

        :param model_ensemble: a collection of models with control instructions
        """
        self.controller = model_controller

    def find_answer_decomp(self, json_input, debug=False):
        """Main question decomposition function

        :param json_input: the input to all of the models.
        """
        raise NotImplementedError

    def return_qid_prediction(
        self,
        example,
        override_answer_by=None,
        debug=False,
        silent=False,
    ):
        final_state, other_states = self.find_answer_decomp(example, debug=debug)

        if final_state is None:
            if not silent:
                print(example["question"] + " FAILED!")
            chain = "\n" + example["qid"] + "\n" + example["question"]
            if not silent:
                print("\n")
            return (example["qid"], "", chain)
        else:
            data = final_state._data
            chain = "\n" + example["qid"] + "\n" + example["question"]
            chain += "\n" + data.get_printable_reasoning_chain()
            chain += "\nS: " + str(final_state._score)

            logger.debug(f"chain: {LYELLOW}{chain}{RESET}")

            if override_answer_by is not None:
                if override_answer_by not in data:
                    logger.warning(f"WARNING: The key {override_answer_by} is not present in the data dict.")
                final_answer = data.get(override_answer_by, "")
                if not isinstance(final_answer, str):
                    final_answer = json.dumps(final_answer)
            else:
                final_answer = data.get_last_answer()

            try:
                json_answer = json.loads(final_answer)
                # use this only if list (ignore numbers, etc)
                if isinstance(json_answer, list) or isinstance(json_answer, str):
                    final_answer = json_answer
            except ValueError:
                # Not a valid json ignore
                pass

            return (example["qid"], final_answer, chain)


class BestFirstDecomposer(QuestionSearchBase):
    def find_answer_decomp(self, json_input, debug=False):
        """Run the question decomposer. The main function here is to use
        the controller to pass around inputs to the different models, then
        keep a track of the search state and terminate when the shortest path
        has been found.
        """
        init_input = json_input["question"] if json_input["question"] else "UNKNOWN"
        logger.info(f"[START QUERY]: {LYELLOW}{init_input}{RESET}")

        start_command = self.controller.start_state
        start_data = self.controller.init_data(json_input)
        init_state = SearchState(start_data, start_command,  score=0.0,)

        heap = []  # 小根堆
        heapq.heappush(heap, init_state)
        while True:
            if len(heap) == 0:
                logger.debug(f"[FAILED]: {LYELLOW}{init_input}{RESET}s")
                return None, []

            current_state = heapq.heappop(heap)
            logger.info(f"[MIN_STATE]: command={LYELLOW}{current_state.next}{RESET}")
            # if current_state.next is None:
            # print(current_state.data.get_printable_reasoning_chain())
            #     current_state.next = current_state.data.get_last_generator()
            # end state
            if current_state.next == self.controller.end_state:
                if current_state.data.has_tasks():
                    new_task = current_state.data.pop_task()
                    # print("popped task!")
                    # print(new_task)
                    new_state = current_state.copy()
                    if new_task.task_question:
                        new_state.data.add_qgen(new_task.task_question)
                    new_state.next = new_task.task_participant
                    heapq.heappush(heap, new_state)
                    continue
                else:
                    logger.debug("[TERMINATED]")
                    return current_state, heap

            logger.info(f'Executing command: {LYELLOW}{current_state.next}{RESET}')
            states = self.controller.execute(current_state, debug=debug)
            for new_state in states:
                heapq.heappush(heap, new_state)
