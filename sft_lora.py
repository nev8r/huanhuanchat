

from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,Trainer
from peft import LoraConfig, get_peft_model,TaskType
import pandas as pd
from datasets import Dataset
import swanlab
from swanlab.integration.transformers import SwanLabCallback

model_name = "./model/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer =  AutoTokenizer.from_pretrained(model_name)
config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r = 8,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","v_proj"],
    bias="none",
    inference_mode=False
)
df = pd.read_json("./data/huanhuan.json")
ds = Dataset.from_pandas(df)

model = get_peft_model(model,config)
model.print_trainable_parameters()

def process_func(examples):
    prompts = [f"指令：{instr}\n输出：{out}" for instr, out in zip(examples["instruction"], examples["output"])]
    
    inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=256,  
        padding="max_length", 
        return_tensors="pt"
    )
    
    labels = inputs["input_ids"].clone()
    for i, prompt in enumerate(prompts):
        output_start = prompt.find("输出：") + 3 
        prefix_length = len(tokenizer(prompt[:output_start], return_tensors="pt")["input_ids"][0])
        labels[i, :prefix_length] = -100
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],  # 必须包含attention_mask，区分真实token和填充的0
        "labels": labels
    }
tokenized_ds = ds.map(process_func, batched=True,remove_columns=ds.column_names)
tokenized_ds
training_args = TrainingArguments(
    output_dir="./lora_output",      # 模型保存路径
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,                       # 混合精度
    save_strategy="epoch",           # 每个 epoch 保存一次
    save_total_limit=2,              # 最多保留 2 个 checkpoint
    logging_steps=10,                # 每 10 step 记录一次 loss
    report_to=["none"],              # 不上报到 HF hub / TensorBoard
)

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="huanhuanchat", 
    experiment_name="last"
)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    callbacks=[swanlab_callback],
)

trainer.train()
