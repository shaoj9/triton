from vllm import LLM, SamplingParams

# Initialize with the FlexAttention backend to enable "one-pass" tree decoding
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_model="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    num_speculative_tokens=16, # Number of draft tokens generated per pass
    attention_backend="flex_attention", 
    trust_remote_code=True
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

prompts = ["Explain the benefits of speculative decoding using EAGLE."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Generated text: {output.outputs[0].text}")