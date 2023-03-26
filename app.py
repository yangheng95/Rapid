import os
import zipfile

import gradio as gr
import nltk
import pandas as pd
import requests
from flask import Flask

from anonymous_demo import TADCheckpointManager
from textattack.attack_recipes import (
    BAEGarg2019,
    PWWSRen2019,
    TextFoolerJin2019,
    PSOZang2020,
    IGAWang2019,
    GeneticAlgorithmAlzantot2018,
    DeepWordBugGao2018,
    CLARE2020,
)
from textattack.attack_results import SuccessfulAttackResult
from utils import SentAttacker, get_agnews_example, get_sst2_example, get_amazon_example, get_imdb_example, diff_texts

nltk.download("omw-1.4")

sent_attackers = {}
tad_classifiers = {}

attack_recipes = {
    "bae": BAEGarg2019,
    "pwws": PWWSRen2019,
    "textfooler": TextFoolerJin2019,
    "pso": PSOZang2020,
    "iga": IGAWang2019,
    "ga": GeneticAlgorithmAlzantot2018,
    "deepwordbug": DeepWordBugGao2018,
    "clare": CLARE2020,
}

app = Flask(__name__)


def init():
    if not os.path.exists("TAD-SST2"):
        z = zipfile.ZipFile("checkpoints.zip", "r")
        z.extractall(os.getcwd())

    for attacker in ["pwws", "bae", "textfooler", "deepwordbug"]:
        for dataset in [
            "agnews10k",
            "amazon",
            "sst2",
            # 'imdb'
        ]:
            if "tad-{}".format(dataset) not in tad_classifiers:
                tad_classifiers[
                    "tad-{}".format(dataset)
                ] = TADCheckpointManager.get_tad_text_classifier(
                    "tad-{}".format(dataset).upper()
                )

            sent_attackers["tad-{}{}".format(dataset, attacker)] = SentAttacker(
                tad_classifiers["tad-{}".format(dataset)], attack_recipes[attacker]
            )
            tad_classifiers["tad-{}".format(dataset)].sent_attacker = sent_attackers[
                "tad-{}pwws".format(dataset)
            ]


cache = set()


def generate_adversarial_example(dataset, attacker, text=None, label=None):
    if not text or text in cache:
        if "agnews" in dataset.lower():
            text, label = get_agnews_example()
        elif "sst2" in dataset.lower():
            text, label = get_sst2_example()
        elif "amazon" in dataset.lower():
            text, label = get_amazon_example()
        elif "imdb" in dataset.lower():
            text, label = get_imdb_example()

    cache.add(text)

    result = None
    attack_result = sent_attackers[
        "tad-{}{}".format(dataset.lower(), attacker.lower())
    ].attacker.simple_attack(text, int(label))
    if isinstance(attack_result, SuccessfulAttackResult):
        if (
                attack_result.perturbed_result.output
                != attack_result.original_result.ground_truth_output
        ) and (
                attack_result.original_result.output
                == attack_result.original_result.ground_truth_output
        ):
            # with defense
            result = tad_classifiers["tad-{}".format(dataset.lower())].infer(
                attack_result.perturbed_result.attacked_text.text
                + "!ref!{},{},{}".format(
                    attack_result.original_result.ground_truth_output,
                    1,
                    attack_result.perturbed_result.output,
                ),
                print_result=True,
                defense="pwws",
            )

    if result:
        classification_df = {}
        classification_df["is_repaired"] = result["is_fixed"]
        classification_df["pred_label"] = result["label"]
        classification_df["confidence"] = round(result["confidence"], 3)
        classification_df["is_correct"] = result["ref_label_check"]

        advdetection_df = {}
        if result["is_adv_label"] != "0":
            advdetection_df["is_adversarial"] = {
                "0": False,
                "1": True,
                0: False,
                1: True,
            }[result["is_adv_label"]]
            advdetection_df["perturbed_label"] = result["perturbed_label"]
            advdetection_df["confidence"] = round(result["is_adv_confidence"], 3)
            # advdetection_df['ref_is_attack'] = result['ref_is_adv_label']
            # advdetection_df['is_correct'] = result['ref_is_adv_check']

    else:
        return generate_adversarial_example(dataset, attacker)

    return (
        text,
        label,
        result["restored_text"],
        result["label"],
        attack_result.perturbed_result.attacked_text.text,
        diff_texts(text, text),
        diff_texts(text, attack_result.perturbed_result.attacked_text.text),
        diff_texts(text, result["restored_text"]),
        attack_result.perturbed_result.output,
        pd.DataFrame(classification_df, index=[0]),
        pd.DataFrame(advdetection_df, index=[0]),
    )


def run_demo(dataset, attacker, text=None, label=None):

    try:
        data = {
            "dataset": dataset,
            "attacker": attacker,
            "text": text,
            "label": label,
        }
        response = requests.post('https://rpddemo.pagekite.me/api/generate_adversarial_example', json=data)
        result = response.json()
        print(response.json())
        return (
            result["text"],
            result["label"],
            result["restored_text"],
            result["result_label"],
            result["perturbed_text"],
            result["text_diff"],
            result["perturbed_diff"],
            result["restored_diff"],
            result["output"],
            pd.DataFrame(result["classification_df"]),
            pd.DataFrame(result["advdetection_df"]),
        )
    except Exception as e:
        print(e)
        return generate_adversarial_example(dataset, attacker, text, label)

if __name__ == "__main__":

    init()

    demo = gr.Blocks()

    with demo:
        gr.Markdown("<h1 align='center'>Reactive Perturbation Defocusing for Textual Adversarial Defense</h1>")
        gr.Markdown("<h3 align='center'>Clarifications</h2>")
        gr.Markdown("""
    - This demo has no mechanism to ensure the adversarial example will be correctly repaired by RPD. The repair success rate is actually the performance reported in the paper (approximately up to 97%).
    - The adversarial example and repaired adversarial example may be unnatural to read, while it is because the attackers usually generate unnatural perturbations. RPD does not introduce additional unnatural perturbations.
    - To our best knowledge, Reactive Perturbation Defocusing is a novel approach in adversarial defense. RPD significantly (>10% defense accuracy improvement) outperforms the state-of-the-art methods.
    - The DeepWordBug is an unknown attacker to the adversarial detector and reactive defense module. DeepWordBug has different attacking patterns from other attackers and shows the generalizability and robustness of RPD.
    - To help the review & evaluation of ACL2023, we will host this demo on a GPU device to speed up the inference process in the next month. Then it will be deployed on a CPU device in the future.
    """)
        gr.Markdown("<h2 align='center'>Natural Example Input</h2>")
        with gr.Group():
            with gr.Row():
                input_dataset = gr.Radio(
                    choices=["SST2", "AGNews10K", "Amazon"],
                    value="SST2",
                    label="Select a testing dataset and an adversarial attacker to generate an adversarial example.",
                )
                input_attacker = gr.Radio(
                    choices=["BAE", "PWWS", "TextFooler", "DeepWordBug"],
                    value="PWWS",
                    label="Choose an Adversarial Attacker for generating an adversarial example to attack the model.",
                )
            with gr.Group():
                with gr.Row():
                    input_sentence = gr.Textbox(
                        placeholder="Input a natural example...",
                        label="Alternatively, input a natural example and its original label to generate an adversarial example.",
                    )
                    input_label = gr.Textbox(
                        placeholder="Original label...", label="Original Label"
                    )

        button_gen = gr.Button(
            "Generate an adversarial example to repair using RPD (GPU: < 1 minute, CPU: 1-10 minutes)",
            variant="primary",
        )

        gr.Markdown("<h2 align='center'>Generated Adversarial Example and Repaired Adversarial Example</h2>")

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    output_original_example = gr.Textbox(label="Original Example")
                    output_original_label = gr.Textbox(label="Original Label")
                with gr.Row():
                    output_adv_example = gr.Textbox(label="Adversarial Example")
                    output_adv_label = gr.Textbox(label="Predicted Label of the Adversarial Example")
                with gr.Row():
                    output_repaired_example = gr.Textbox(
                        label="Repaired Adversarial Example by RPD"
                    )
                    output_repaired_label = gr.Textbox(label="Predicted Label of the Repaired Adversarial Example")

        gr.Markdown("<h2 align='center'>Example Difference (Comparisons)</p>")
        gr.Markdown("""
    <p align='center'>The (+) and (-) in the boxes indicate the added and deleted characters in the adversarial example compared to the original input natural example.</p>
        """)
        ori_text_diff = gr.HighlightedText(
            label="The Original Natural Example",
            combine_adjacent=True,
        )
        adv_text_diff = gr.HighlightedText(
            label="Character Editions of Adversarial Example Compared to the Natural Example",
            combine_adjacent=True,
        )
        restored_text_diff = gr.HighlightedText(
            label="Character Editions of Repaired Adversarial Example Compared to the Natural Example",
            combine_adjacent=True,
        )

        gr.Markdown(
            "## <h2 align='center'>The Output of Reactive Perturbation Defocusing</p>"
        )
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    output_is_adv_df = gr.DataFrame(
                        label="Adversarial Example Detection Result"
                    )
                    gr.Markdown(
                        "The is_adversarial field indicates if an adversarial example is detected. "
                        "The perturbed_label is the predicted label of the adversarial example. "
                        "The confidence field represents the confidence of the predicted adversarial example detection. "
                    )
            with gr.Column():
                with gr.Group():
                    output_df = gr.DataFrame(
                        label="Repaired Standard Classification Result"
                    )
                    gr.Markdown(
                        "If is_repaired=true, it has been repaired by RPD. "
                        "The pred_label field indicates the standard classification result. "
                        "The confidence field represents the confidence of the predicted label. "
                        "The is_correct field indicates whether the predicted label is correct."
                    )

        # Bind functions to buttons
        button_gen.click(
            fn=run_demo,
            inputs=[input_dataset, input_attacker, input_sentence, input_label],
            outputs=[
                output_original_example,
                output_original_label,
                output_repaired_example,
                output_repaired_label,
                output_adv_example,
                ori_text_diff,
                adv_text_diff,
                restored_text_diff,
                output_adv_label,
                output_df,
                output_is_adv_df,
            ],
        )

    demo.queue(2).launch()

