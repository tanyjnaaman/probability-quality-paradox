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
1. Download `data.zip` from [here](https://drive.google.com/file/d/1P6rrRhY4_5yDIlINVdOeszy9Os8uhepd/view?usp=sharing) and `data_dpo.zip` from [here](https://drive.google.com/file/d/1pY4xLoWloX-Xmoi-DjZQoZRCypMquLYj/view?usp=sharing).
2. Unzip the files into `src/language_model_experiments/data` and `src/language_model_experiments/data_dpo` respectively.
3. Download the cached intermediate outputs `.cache.zip` from [here](https://drive.google.com/file/d/1-wDMUnbWKto9U4ns51z7Eg7TlTRfKgqB/view?usp=sharing) and `.cache_dpo.zip` from [here](https://drive.google.com/file/d/1NvYaeXFF4jP9amdOGhS42b3fkwWl6JDS/view?usp=sharing).
4. Unzip the files into `src/language_model_experiments/.cache` and `src/language_model_experiments/.cache_dpo` respectively.
5. Generate the figures with 
```
bash create_figures.sh
```

### Option 2: Run the experiment from scratch
1. Generate all 25 corpora with both a RLHF-tuned and DPO-tuned model with:
```
bash generate.sh
```
This should generatee `.csv` files of generated text.
2. Score all 25 corpora with:
```
bash score_with_correction.sh
```
Comment out files and model settings as needed. 
For every `.csv` file, this should produce 3 files scoring the reward and log-probabilities of the generated strings.
When this is done, move all files to `src/language_model_experiments/data`  and `src/language_model_experiments/data_dpo`, for the RLHF-tuned model and DPO-tuned model, respectively.

3. Run the Independent Metropolis Hastings algorithm and generate the figures with:
```
bash create_figures.sh
```