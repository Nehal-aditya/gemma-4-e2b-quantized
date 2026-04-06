import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or HF repo")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (4 or 8)")
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    print(f"Quantizing to {args.bits}-bit...")
    quantize_config = BaseQuantizeConfig(bits=args.bits, group_size=128, desc_act=False)
    quantized_model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        quantize_config,
        model_save_dir=args.output_dir,
        tokenizer=tokenizer
    )

    print(f"Quantized model saved to {args.output_dir}")

if __name__ == "__main__":
    main()