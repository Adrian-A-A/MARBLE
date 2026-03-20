import os

from litellm import completion

os.environ["OLLAMA_API_KEY"] = "af2c304065324d549ad3a0512c4605cb.XeY63kqybrSMWAGWArx4INoZ"


def generate_text(prompt):
    try:
        messages = [{"role": "user", "content": prompt}]

        response = completion(
            model="openai/qwen2.5:0.5b",
            messages=messages,
        )

        generated_text = response.choices[0].message.content
        return generated_text

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None


def main():
    prompt = "Please write a short essay about artificial intelligence."

    print("Starting content generation...")
    result = generate_text(prompt)

    if result:
        print("\nGenerated content:")
        print(result)
    else:
        print("\nGeneration failed")


if __name__ == "__main__":
    main()
