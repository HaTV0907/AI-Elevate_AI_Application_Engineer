from resume_generator import LlamaModel, ResumeGenerator
from sample_inputs import sample_inputs

try:
    model = LlamaModel()
    generator = ResumeGenerator(model)

    for i, user_data in enumerate(sample_inputs, 1):
        print(f"\n📄 Resume {i} for {user_data['name']}:")
        resume = generator.generate_resume(user_data)
        print(resume)
        print("-" * 80)

except Exception as e:
    print(f"❌ Error: {e}")
