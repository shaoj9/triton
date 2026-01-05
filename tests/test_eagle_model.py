from vllm import LLM, SamplingParams

def test_load_eagle_vllm(base_model_path, eagle_model_path):
    print(f"--- Starting vLLM EAGLE Loading Test ---")
    try:
        llm = LLM(
            model=base_model_path,
            speculative_config={
                "draft_model": eagle_model_path,
                "temperature": 0.0, # Example parameter
                "method": "eagle3" if "EAGLE-3" in eagle_model_path else "eagle"
            }
        )
        print("vLLM instance with EAGLE support initialized successfully.")
        
        # Example generation test
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=20)
        prompts = ["Hello, my name is", "The capital of France is"]
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            print(f"Prompt: {output.prompt}, Generated: {output.outputs[0].text}")

    except Exception as e:
        print(f"[TEST FAILED] An error occurred during vLLM initialization: {e}")

if __name__ == "__main__":
    test_load_eagle_vllm("mistralai/Mistral-7B-Instruct-v0.2", "yuhuili/EAGLE-Mistral-7B-Instruct-v0.2")