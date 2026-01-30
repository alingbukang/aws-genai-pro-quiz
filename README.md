# aws-genai-pro-quiz

# AWS GenAI Pro Quiz

Practice for AWS GenAI Pro certification with AI-verified questions.

## Quick Start

```bash
# Take a 10-question quiz
python take_quiz.py

# Specify number of questions
python take_quiz.py -n 25
```

## Generate New Questions

Requires Azure OpenAI API access (~$10 per 100 questions).

```bash
# Setup
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Generate 100 verified questions
python generate_verified_questions.py -n 100 -o my_questions.json

# Take quiz with custom file
python take_quiz.py -q my_questions.json
```

## Files

| File | Description |
|------|-------------|
| `take_quiz.py` | Interactive quiz runner |
| `generate_verified_questions.py` | Generate questions with AI critic verification |
| `aws_genai_pro_100.json` | 100 pre-generated verified questions |

## Requirements

```bash
pip install -r requirements.txt
```

## Question Topics

- **Amazon Bedrock** (60%): Agents, Knowledge Bases, Guardrails, Fine-tuning
- **Amazon SageMaker** (20%): Training, Inference, MLOps, Monitoring  
- **Architecture** (10%): RAG, Vector DBs, Orchestration
- **AI Services** (10%): Kendra, Textract, Comprehend, Rekognition

## License

MIT