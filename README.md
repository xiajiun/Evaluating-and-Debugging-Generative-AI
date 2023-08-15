# Evaluating-and-Debugging-Generative-AI
https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/

Short course of Deep Learning.AI

> This course will introduce you to Machine Learning Operations tools that manage this workload. You will learn to use the Weights & Biases platform which makes it easy to track your experiments, run and version your data, and collaborate with your team.

1. Introduction
2. Instrument W&B
   ```
   # pip install wandb
   import wandb

   # 1. Organize your hyperparameters
   config = {'learning_rate': 0.001}
   #config = wandb.config 

   # 2. Start wandb run
   wandb.init(project='gpt5',
              job_type="train",
              config=config}

   # Model training here

   # 3. Log metrics over time to visualize performance
   wandb.log({"loss": loss})

   #4. When working in a notebook, finish
   wandb.finish()   
   ```
3. Training a Diffusion Model with W&B
   - Keep track of the loss and relevant metrics
   - Sample images from the model during training
   - Safely store and version model checkpoints
   ```
   # create a wandb run
   run = wandb.init(project="dlai_sprite_diffusion", 
                 job_type="train", 
                 config=config)

   # we pass the config back from W&B
   config = wandb.config

   #save model periodically
   artifact_name = f"{wandb.run.id}_context_model"
   at = wandb.Artifact(artifact_name, type="model")
   at.add_file(ckpt_file)
   wandb.log_artifact(at, aliases=[f"epoch_{ep}"])

   #sample images to work from workspace
   samples, _ = sample_ddpm_context(nn_model, 
                                         noises, 
                                         ctx_vector[:config.num_samples])
   wandb.log({
            "train_samples": [
                wandb.Image(img) for img in samples.split(1)
            ]})
   ```
   - Link model in Model Industry

4. Evaluating Diffusion Models
   - Model registry: a central system of record for your models
     ```
     "Load the model from wandb artifacts"
     api = wandb.Api()
     artifact = api.artifact(model_artifact_name, type="model")
     model_path = Path(artifact.download())

     # recover model info from the registry
     producer_run = artifact.logged_by()
     ```
   - W&B Tables
     - Log, query, and analyze tabular data including rich media: images, videos, molecules, etc.
     - Compare changes precisely across models
     ```
     #create a wandb.Table to store our generations
     table = wandb.Table(columns=["input_noise", "ddpm", "ddim", "class"])
     # add data row by row to the Table
     table.add_data(wandb.Image(noise),
                   wandb.Image(ddpm_s), 
                   wandb.Image(ddim_s),
                   c)
     wandb.log({'predictions': table})
     ```
    - Create report

5. LLM Evaluation and Tracing with W&B
   - Using APIs with Tables
     - generating names using OpenAI `ChatCompletion`, and saving the resulting generations in W&B Tables. 
   - Tracking LLM chain spans with Tracer
     - to debug LLM chains and workflows
       ```
       import wandb
       from wandb.sdk.data_types.trace_tree import Trace
       ```
   - Tracking Langchain Agents
     
7. Finetuning a languange model

   Training from scratch
   - Long & expensive training runs
   - Expensive & difficult evaluations
   - Monitoring is critical
   - Ability to restore training from a checkpoint


   Fine-tuning
   - Efficient methods being developed
   - Expensive & difficult evaluations
    
  
   
9. Conclusion
