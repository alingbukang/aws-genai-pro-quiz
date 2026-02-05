# AWS GenAI Pro Quiz

Practice for AWS GenAI Pro certification with AI-verified questions.

## 🌐 Live Demo

**Try it now:** [https://alingbukang.github.io/aws-genai-pro-quiz/](https://alingbukang.github.io/aws-genai-pro-quiz/)

## Quick Start

### Web (Recommended)
```bash
# Start local server
python -m http.server 8000

# Open http://localhost:8000 in your browser
```

### CLI
```bash
# Take a 10-question quiz
python take_quiz.py

# Specify number of questions
python take_quiz.py -n 25
```

## 🚀 Deploy to GitHub Pages (Free Hosting)

1. **Fork or clone this repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/aws-genai-pro-quiz.git
   cd aws-genai-quiz
   ```

2. **Push to your GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

3. **Enable GitHub Pages**
   - Go to your repo on GitHub
   - Settings → Pages
   - Source: Deploy from branch
   - Branch: `main` / `root`
   - Click Save

4. **Access your quiz**
   ```
   https://YOUR_USERNAME.github.io/aws-genai-pro-quiz/
   ```

## Generate New Questions

Requires AI API access. See `.env.example` for supported providers.

```bash
# Setup
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Generate 100 verified questions
python generate_verified_questions.py -n 100 -o my_questions.json

# Take quiz with custom file
python take_quiz.py -q my_questions.json
```

### Approximate Cost (Azure OpenAI GPT-4o Example)

| Item | Tokens | Rate | Cost |
|------|--------|------|------|
| Generator prompts (input) | ~2,000 per question | $2.50/1M tokens | $0.005 |
| Generator responses (output) | ~500 per question | $10.00/1M tokens | $0.005 |
| Critic prompts (input) | ~2,500 per question | $2.50/1M tokens | $0.006 |
| Critic responses (output) | ~300 per question | $10.00/1M tokens | $0.003 |
| **Per question total** | | | **~$0.02** |
| **100 questions** | | | **~$2.00** |
| **With retries (~30%)** | | | **~$2.50-3.00** |

> **Note:** Actual costs vary based on question complexity and retry rates.


## Files

| File | Description |
|------|-------------|
| `index.html` | Web quiz (GitHub Pages) |
| `take_quiz.py` | CLI quiz runner |
| `generate_verified_questions.py` | Generate questions with AI critic |
| `aws_genai_pro.json` | 100 pre-generated verified questions |

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
