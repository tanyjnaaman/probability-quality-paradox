# A Fundamental Trade-off in Aligned Language Models and its Relation to Sampling Adaptors

This is the repository containing code to replicate the experiments in [our paper]().

## Environment set up 
Create a python environment and then run:
```
pip install -r requirements.txt
```

## Toy Experiment
To run our toy data experiment, run all cells in `/src/experiments/toy_data_experiments`.

## Language Model Experiment
To run our empirical experiment, we will need to: 
1. Generate strings with different sampling adaptors using an RLHF-tuned model.
2. Score the generated corpora and compute their log-probabilities under the prior and RLHF-tuned model with the sampling adaptor. 
3. Run the Independent Metropolis Hastings Algorithm. 

To do so, you can use our files or run the scripts to generate everything from scratch. 

### Option 1: download our files and generate the figures
1. Download `data.zip` from [here]().
2. Unzip the files into `src/language_model_experiments/data`.
3. Download the cached intermediate outputs `.cache.zip` from [here]().
4. Unzip the files into `src/language_model_experiments/.cache`.
5. Generate the figures with 
```
bash create_figures.sh
```

Option 2: Run the experiment from scratch
1. Generate all 25 corpora with:
```
bash generate.sh
```
2. Score all 25 corpora with:
```
bash score_with_correction.sh
```
3. Run the Independent Metropolis Hastings algorithm and generate the figures with:
```
create_figures.sh
```
