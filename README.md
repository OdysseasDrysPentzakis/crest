# CREST (Contrastive Edits with Sparse Rationalization)

This repository contains the code for [CREST: A Joint Framework for Rationalization and Counterfactual Text Generation](https://arxiv.org/abs/), accepted at ACL 2023.

CREST consists of two stages: Counterfactual Generation and Rationalization. 
You can find the instructions for running each stage below.


## Requirements and Installation

This code was tested on `Python 3.8.10`. To install, follow these steps:

1. In a **virtual environment**, first install Cython: `pip install cython`
2. Clone the [Eigen](https://gitlab.com/libeigen/eigen.git) repository to the main folder: `git clone git@gitlab.com:libeigen/eigen.git`
3. Clone the [LP-SparseMAP](https://github.com/nunonmg/lp-sparsemap) fork repository to main folder, and follow the installation instructions found there
   - Follow this fix in case of compilation error: https://github.com/deep-spin/lp-sparsemap/issues/9
4. Install the requirements: `pip install -r requirements.txt`
5. Install the package: `pip install .` (or in editable mode if you want to make changes: `pip install -e .`)


## Repo Organization

The repo is organized as follows:
```shell
.
├── configs  # Config files for training the models
│   ├── agreement_regularization  # Config files for training the models with agreement regularization
│   ├── data_augmentation  # Config files for training the models with data augmentation
│   ├── editor  # Config files for training the editor
│   └── masker  # Config files for training the masker
├── data  # Data files
│   ├── edits  # Edits generated by CREST in a tsv format
│   └── rationales  # Rationales generated by the rationalizers
├── experiments  # Experiments results, including checkpoints and logs
├── rationalizers
│   ├── custom_hf_datasets  # Custom datasets for Hugging Face's Datasets library
│   ├── data_modules  # Data modules for PyTorch Lightning
│   ├── explainers  # Explainer modules
│   ├── lightning_models  # PyTorch Lightning models
│   └── modules  # Extra PyTorch modules
├── notebooks  # Jupyter notebooks for data analysis and model evaluation
└── scripts  # Scripts for running experiments and extracting counterfactuals
```


## Resources:

We provide the counterfactuals generated by CREST and MiCE for the IMDB and SNLI datasets:

| Dataset      | Method                  | File Link                                            |
|--------------|-------------------------|------------------------------------------------------|
| Revised IMDB | MiCE with binary search | [Link](data/edits_paper/imdb/mice_binary_search.csv) |
| Revised IMDB | MiCE with 30% masking   | [Link](data/edits_paper/imdb/mice_30_masking.csv)    |
| Revised IMDB | MiCE with 50% masking   | [Link](data/edits_paper/imdb/mice_50_masking.csv)    |
| Revised IMDB | CREST with 30% masking  | [Link](data/edits_paper/imdb/crest_30_masking.tsv)   |
| Revised IMDB | CREST with 50% masking  | [Link](data/edits_paper/imdb/crest_50_masking.tsv)   |
| Revised SNLI | MiCE with binary search | [Link](data/edits_paper/snli/mice_binary_search.csv) |
| Revised SNLI | MiCE with 30% masking   | [Link](data/edits_paper/snli/mice_30_masking.csv)    |
| Revised SNLI | MiCE with 50% masking   | [Link](data/edits_paper/snli/mice_50_masking.csv)    |
| Revised SNLI | CREST with 30% masking  | [Link](data/edits_paper/snli/crest_30_masking.tsv)   |
| Revised SNLI | CREST with 50% masking  | [Link](data/edits_paper/snli/crest_50_masking.tsv)   |


## CREST-Generation

The generation stage is divided into two phases: 
1. training a masker (a rationalizer). 
2. training an editor (a LLM).


### Training a Masker

To train a masker, first define a config file with the hyperparameters of the model. 
Take a look at the config files in the `configs/masker` folder for examples. 
The meaning of the relevant hyperparameters is described in the table below.

Then, run the following command (e.g., for training a masker with 30% masking on the IMDB dataset):

```bash
python3 rationalizers train --config configs/masker/imdb_sparsemap_30p.yaml
```

After training, the rationalizer will be saved to the path informed in the `default_root_dir` option.

This phase uses the following hyperparameters:

| Hyperparam              | Default | Description                                                                                                          |
|-------------------------|---------|----------------------------------------------------------------------------------------------------------------------|
| tokenizer 					| `'t5-small'` | Pre-trained tokenizer from the Hugging Face hub. If None, a nltk's WordPunct tokenizer is used                       |
| gen_arch 					| `'t5-small'` | Pre-trained LM from the Hugging Face hub used as the generator                                                       |
| gen_emb_requires_grad 		| `False` | Determines if the generator's embedding layer is trainable (`True`) or frozen (`False`)                              |
| gen_encoder_requires_grad 	| `False` | Determines if the generator's encoding layers are trainable (`True`) or frozen (`False`)                             |
| gen_use_decoder 				| `False` | Specifies if the generator's decoder module (if applicable) is used                                                  |
| pred_arch 					| `'t5-small'` | Pre-trained LM from the Hugging Face hub used as the predictor. Other options include `lstm` or `masked_average`.    |
| pred_emb_requires_grad 		| `False` | Determines if the predictor's embedding layer is trainable (`True`) or frozen (`False`). (`False`)                   |
| pred_encoder_requires_grad 	| `True` | Determines if the predictor's encoding layers are trainable (`True`) or frozen (`False`) (`False`)                   |
| pred_output_requires_grad 	| `True` | Determines if the predictor's final output layers are trainable (`True`) or frozen (`False`). (`False`)              |
| pred_bidirectional 			| `False` | Specifies if the predictor is bidirectional (for `lstm`)                                                             |
| dropout 						| `0.1` | Dropout for the predictor's output layers                                                                            |
| shared_gen_pred 				| `False` | Specifies if the weights of the generator and the predictor are shared                                               |
| explainer 					| `'sparsemap'` | Explainer choice. See all options [here](rationalizers/explainers/__init__.py)                                       |
| explainer_pre_mlp 			|  `True` | Specifies if a trainable MLP is included before the explainer                                                        |
| explainer_requires_grad 		| `True` | Determines if the explainer is trainable or frozen, including the pre-MLP                                            |
| sparsemap_budget 			| `30` | Sequence budget for the SparseMAP explainer                                                                          |
| sparsemap_transition 		| `0.1` | Transition weight for the SparseMAP explainer                                                                        |
| sparsemap_temperature 		| `0.01` | Temperature for training with SparseMAP explainer                                                                    |
| selection_vector 			| `'zero'` | Which vector to use to represent differentiable masking: `mask` for [MASK], `pad` for [PAD], and `zero` for 0 vectors |
| selection_faithfulness 		| `True` | Whether to perform masking on the original input x (`True`) or on the hidden states h (`False`)                      |
| selection_mask 				| `False` | Whether to also mask elements during self-attention, rather than only masking input vectors                          |



### Training an editor

To train an editor, first define a config file with the hyperparameters of the model. 
Check the config files in the `configs/editor` folder for examples.

- Make sure to inform the path of the rationalizer trained in the previous phase via the `factual_ckpt` argument in the config file.
- Make sure all the previous hyperparameters defined above are kept intact for training the editor. Alternatively, keep them undefined, in which case they will be loaded with the pre-trained rationalizer.

Then, run the following command (e.g., for training a T5-small editor on the IMDB dataset):

```bash
python3 rationalizers train --config configs/editor/imdb_sparsemap_30p.yaml
```

After training, the editor will be saved to the path informed in the `default_root_dir` option.

This phase uses the following hyperparameters:

| Hyperparam 				| Default                                                                                          | Description                                                                                                                                                                                           |
|-------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| factual_ckpt 			| `None` 	   				                                                                                  | Path to the pre-trained rationalizer checkpoint                                                                                                                                                       |
| cf_gen_arch 				| `'t5-small'` 				                                                                                | The name of a pre-trained LM from the huggingface hub to use as the editor                                                                                                                            |
| cf_prepend_label_type 	| `'gold'` 	   				                                                                                | Whether to prepend gold (`gold`) or predicted (`pred`) labels to the input of the editor                                                                                                              |
| cf_z_type 		 		| `'pred'`                                                                                         | Whether to use the factual rationalizers' rationales (`pred`) or gold rationales, when available (`gold`)                                                                                             |
| cf_task_name 		 	| `'binary_classification'`                                                                        | The name of the task at hand, used to create the name of prepend labels: `binary_classification`, `nli`, `nli_no_neutrals`                                                                            |
| cf_classify_edits 	 	| `True` 					                                                                                     | Whether to classify the edits after generation                                                                                                                                                        |
| cf_generate_kwargs 		| `do_sample: False, num_beams: 15, early_stopping: True, length_penalty: 1.0, no_repeat_ngram: 2` | Generation options passed to [huggingface's generate method](https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) |


> **Note:** You can get better counterfactuals by using a larger language model as the editor, e.g., t5-base or t5-large. However, this will increase the training time.


### Extracting Counterfactuals

To extract counterfactuals from the editor (e.g., for the Revised IMDB dataset), run:

```bash
python3 scripts/get_edits.py \
    --ckpt-name "foo" \
    --ckpt-path "path/to/editor/checkpoint" \
    --dm-name "revised_imdb" \
    --dm-dataloader "test" \
    --num-beams 15
```

The counterfactuals will be saved in a file named `data/edits/{dm_name}_{dm_dataloader}_beam_{num_beams}_{ckpt_name}_raw.tsv`.

For more information about how to generate counterfactuals, check the instructions in the [scripts](scripts/) folder.


### Evaluating Counterfactuals

Before proceeding, install the evaluation requirements with `pip install -r requirements_eval.txt`.

To analyze the counterfactuals produced by the editor, follow the instructions in the counterfactual analysis notebooks 
for (IMDB)[notebooks/counterfactual_analysis_imdb.ipynb] and (SNLI)[notebooks/counterfactual_analysis_snli.ipynb].
The evaluation includes the following metrics:
- _validity_ computed with off-the-shelf classifiers
- _fluency_ computed with GPT-2 large
- _diversity_ computed with self-BLEU
- _closeness_ computed with normalized edit distance

---


## CREST-Rationalization

Before starting the rationalization process, we need to generate counterfactuals and extract rationales for
all training examples. To do this, we can use the `get_edits.py` script. For example:

```bash
python3 scripts/get_edits.py \
  --ckpt_name "foo" \
  --ckpt_path "path/to/editor.ckpt" \
  --dm_name "imdb" \
  --dm_dataloader "train" \
  --num_beams 15
```
This will use a pre-trained editor to produce edits for all training examples from the "imdb" dataset and save the results in a file named `data/edits/{dm_name}_{dm_dataloader}_beam_{num_beams}_{ckpt_name}.tsv`.

Next, train a new rationalizer that incorporates these edits by running (e.g., for training a SparseMAP rationalizer on the IMDB dataset):

```bash
python3 rationalizers train --config configs/agreement_regularization/imdb_sparsemap_30p.yaml
```

The trained rationalizer will be saved to the path specified in the `default_root_dir` option.

This phase uses the following hyperparameters:

| Hyperparam                | Default               | Description                                                                                                |
|---------------------------|-----------------------|------------------------------------------------------------------------------------------------------------|
| synthetic_edits_path      | `None`                | Path to counterfactuals for all training examples (in order)                                               |
| filter_invalid_edits      | `False`               | Whether to disregard counterfactuals predicted wrongly by the original rationalizer                        |
 | ff_lbda                   | `1.0`                 | Weight for the factual loss                                                                                |
 | cf_lbda                   | `0.01`                | Weight for the counterfactuals loss                                                                        |
 | expl_lbda                 | `0.001`               | Weight for the explainer loss                                                                              |
 | sparsemap_budget_strategy | `'adaptive_dynamic'`  | Strategy for setting the budget for the SparseMAP explainer: `'fixed'`, `'adaptive'`, `'adaptive_dynamic'` |


### Evaluating Models

Check the script [scripts/evaluate_model.py](scripts/evaluate_model.py) to evaluate the models on in-domain and out-of-domain data.

The running commands for all steps can be found in the [run_steps.sh](run_steps.sh) script.

---

## Interpretability Analysis

To analyze the interpretability of the rationales produced by the rationalizer, check out the notebooks in the [notebooks](notebooks/) folder.

The analysis includes the following metrics:

- **Plausibility:**
   1. Extract rationales with:
    ```bash
    python3 scripts/get_rationales.py \
        --ckpt-name "foo" \
        --ckpt-path "path/to/rationalizer.ckpt" \
        --dm-name "movies" \
        --dm-dataloader "test"
    ```
    2. The rationales will be saved in a file named `data/rationales/{dm_name}_{dm_dataloader}_{ckpt_name}.tsv`.
    3. Follow the instructions in the [notebooks/plausibility_imdb.ipynb](notebooks/plausibility_imdb.ipynb) notebook.


- **Forward Simulation**:

   1. Train a student model:
    ```bash
    python3 scripts/forward_simulation.py \
        --student-type "bow" \
        --train-data "path/to/train_edits.tsv" \
        --test-data "path/to/test_edits.tsv" \
        --batch-size 16 \
        --seed 0
    ```
   2. Save the path of the checkpoint of the student model.
   3. Follow the instructions in the [notebooks/forward_simulation.ipynb](notebooks/forward_simulation.ipynb) notebook.


- **Counterfactual Simulation**:

   1. Extract edits with:
    ```bash
    python3 scripts/get_edits.py \
      --ckpt-name "foo" \
      --ckpt-path "path/to/editor.ckpt" \
      --ckpt-path-factual "path/to/another/masker.ckpt" \
      --dm-name "imdb" \
      --dm-dataloader "train" \
      --num-beams 15
    ```
    2. The extracted edits will be saved in a file named `data/edits/{dm_name}_{dm_dataloader}_beam_{num_beams}_{ckpt_name}_factual.tsv`.
    3. Follow the instructions in the [notebooks/counterfactual_simulation.ipynb](notebooks/counterfactual_simulation.ipynb) notebook.



## Human Evaluation Tool

The tool used for the human evaluation study is available here: https://github.com/mtreviso/TextRankerJS. 
Check also the [Online Demo](https://mtreviso.github.io/TextRankerJS/index_likert.html).


## Citation

If you found our work/code useful, consider citing our paper:

```bibtex
todo...
```


