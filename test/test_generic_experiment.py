from unittest import TestCase
from unittest.mock import patch

from openai.types import Completion
from openai.types.chat import ChatCompletion

from prompttools.experiment import ChatCompletionExperiment, CompletionExperiment


class TestGenericExperiment(TestCase):
    def test_completion_callable_receives_every_argument_combination(self):
        calls = []

        def complete(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"text": kwargs["prompt"].upper()}]}

        with patch.dict("os.environ", {"DEBUG": "1", "SENTRY_OPT_OUT": "1"}, clear=True):
            experiment = CompletionExperiment(complete, model=["local"], prompt=["one", "two"], temperature=[0, 1])
            self.addCleanup(experiment.queue.shutdown)
            experiment.run()

        self.assertEqual(len(calls), 4)
        self.assertEqual(experiment.full_df["response"].tolist(), ["ONE", "ONE", "TWO", "TWO"])
        self.assertEqual(experiment._get_model_names(), ["local"] * 4)
        self.assertTrue((experiment.score_df["latency"] >= 0).all())

    def test_chat_callable_receives_whole_conversations(self):
        conversations = [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]]
        calls = []

        def chat(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": kwargs["messages"][-1]["content"].upper()}}]}

        experiment = ChatCompletionExperiment(chat, model=["local"], messages=conversations)
        self.addCleanup(experiment.queue.shutdown)
        experiment.run()

        self.assertEqual([call["messages"] for call in calls], conversations)
        self.assertEqual(experiment.full_df["response"].tolist(), ["ONE", "TWO"])
        self.assertTrue(experiment._is_chat())

    def test_sdk_response_objects_are_supported(self):
        responses = [
            (
                CompletionExperiment,
                {"prompt": ["one"]},
                Completion(
                    id="completion",
                    created=0,
                    model="local",
                    object="text_completion",
                    choices=[{"text": "completed", "index": 0, "finish_reason": "stop", "logprobs": None}],
                ),
                "completed",
            ),
            (
                ChatCompletionExperiment,
                {"messages": [[{"role": "user", "content": "one"}]]},
                ChatCompletion(
                    id="chat",
                    created=0,
                    model="local",
                    object="chat.completion",
                    choices=[
                        {"message": {"role": "assistant", "content": "chatted"}, "index": 0, "finish_reason": "stop"}
                    ],
                ),
                "chatted",
            ),
        ]
        for experiment_class, parameters, response, expected in responses:
            with self.subTest(experiment=experiment_class.__name__):
                experiment = experiment_class(lambda **kwargs: response, model=["local"], **parameters)
                self.addCleanup(experiment.queue.shutdown)
                experiment.run()
                self.assertEqual(experiment.full_df["response"].tolist(), [expected])
                self.assertEqual(experiment.full_df["response_id"].tolist(), [response.id])

    def test_initialize_keeps_the_callable_outside_the_parameter_grid(self):
        def complete(**kwargs):
            return {"choices": [{"text": kwargs["prompt"]}]}

        experiment = CompletionExperiment.initialize(
            complete,
            test_parameters={"prompt": ["one", "two"]},
            frozen_parameters={"model": "local", "temperature": 0},
        )
        self.addCleanup(experiment.queue.shutdown)
        experiment.run()
        self.assertEqual(experiment.full_df["response"].tolist(), ["one", "two"])
        self.assertEqual(experiment.full_df["temperature"].tolist(), [0, 0])

    def test_invalid_callable_is_rejected_before_starting_a_queue(self):
        with self.assertRaises(TypeError):
            CompletionExperiment(None, model=["local"], prompt=["one"])

    def test_parameter_choices_must_be_nonempty_lists(self):
        for choices in ["prompt", [], None]:
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                CompletionExperiment(lambda **kwargs: {}, model=["local"], prompt=choices)
