#!/usr/bin/env python3
"""
AWS GenAI Pro Exam Question Generator with Critic Verification

Generates high-quality AWS certification practice questions with:
- Dual-model verification (generator + critic)
- AWS documentation citations
- Quality scoring and filtering
- AWS GenAI Pro certification level difficulty

Usage:
    python generate_verified_questions.py -n 100 -o aws_genai_pro.json
"""

import json
import random
import os
import argparse
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

try:
    from openai import AzureOpenAI
except ImportError:
    print("❌ Required package missing. Run: pip install openai python-dotenv")
    exit(1)


# =============================================================================
# AWS Documentation References
# =============================================================================

AWS_DOCS = {
    "bedrock": {
        "base": "https://docs.aws.amazon.com/bedrock/latest/userguide/",
        "pages": {
            "agents": "agents.html",
            "knowledge-bases": "knowledge-base.html",
            "guardrails": "guardrails.html",
            "fine-tuning": "custom-models.html",
            "quotas": "quotas.html",
            "security": "security.html",
            "models": "models-supported.html",
            "inference": "inference.html",
            "provisioned": "prov-throughput.html",
        }
    },
    "sagemaker": {
        "base": "https://docs.aws.amazon.com/sagemaker/latest/dg/",
        "pages": {
            "training": "how-it-works-training.html",
            "inference": "deploy-model.html",
            "pipelines": "pipelines.html",
            "feature-store": "feature-store.html",
            "model-monitor": "model-monitor.html",
            "clarify": "clarify-configure-processing-jobs.html",
            "jumpstart": "studio-jumpstart.html",
        }
    },
    "ai-services": {
        "kendra": "https://docs.aws.amazon.com/kendra/latest/dg/",
        "textract": "https://docs.aws.amazon.com/textract/latest/dg/",
        "comprehend": "https://docs.aws.amazon.com/comprehend/latest/dg/",
        "rekognition": "https://docs.aws.amazon.com/rekognition/latest/dg/",
        "transcribe": "https://docs.aws.amazon.com/transcribe/latest/dg/",
    },
}


# =============================================================================
# AWS Facts for Critic Grounding
# =============================================================================

AWS_FACTS = {
    "bedrock": {
        "models": {
            "claude_3_5_sonnet": {"context_window": 200000, "multimodal": True, "fine_tuning": False},
            "claude_3_sonnet": {"context_window": 200000, "multimodal": True, "fine_tuning": False},
            "claude_3_haiku": {"context_window": 200000, "multimodal": True, "fine_tuning": False},
            "claude_3_opus": {"context_window": 200000, "multimodal": True, "fine_tuning": False},
            "titan_text_express": {"context_window": 8000, "multimodal": False, "fine_tuning": True},
            "titan_text_lite": {"context_window": 4000, "multimodal": False, "fine_tuning": True},
            "titan_text_premier": {"context_window": 32000, "multimodal": False, "fine_tuning": True},
            "llama_3_1_70b": {"context_window": 128000, "multimodal": False, "fine_tuning": False},
            "llama_3_1_8b": {"context_window": 128000, "multimodal": False, "fine_tuning": False},
            "llama_2_13b": {"context_window": 4096, "multimodal": False, "fine_tuning": True},
            "llama_2_70b": {"context_window": 4096, "multimodal": False, "fine_tuning": True},
            "cohere_command_r": {"context_window": 128000, "multimodal": False, "fine_tuning": True},
            "mistral_large": {"context_window": 32000, "multimodal": False, "fine_tuning": False},
            "mistral_small": {"context_window": 32000, "multimodal": False, "fine_tuning": True},
        },
        "fine_tuning": {
            "supported": True,
            "supported_models": ["Amazon Titan Text", "Llama 2", "Cohere Command", "Mistral Small"],
            "not_supported_models": ["Claude (any version)", "Llama 3", "Mistral Large"],
            "data_format": "JSONL",
            "min_samples": 100,
            "max_samples": 10000,
        },
        "knowledge_bases": {
            "vector_stores": ["OpenSearch Serverless", "Aurora PostgreSQL", "Pinecone", "Redis Enterprise", "MongoDB Atlas"],
            "chunking_strategies": ["Fixed size", "Default", "Semantic", "Hierarchical", "No chunking"],
            "max_data_sources": 5,
        },
        "guardrails": {
            "filter_types": ["Content filters", "Denied topics", "Word filters", "PII filters", "Contextual grounding"],
            "pii_actions": ["BLOCK", "ANONYMIZE"],
            "content_filter_strengths": ["NONE", "LOW", "MEDIUM", "HIGH"],
        },
        "provisioned_throughput": {
            "commitment_periods": ["1 month", "6 months"],
            "discount_6_month": "Up to 50%",
        },
        "agents": {
            "max_action_groups": 20,
            "max_api_schema_size": "300KB",
            "session_timeout_max": "1 hour",
        },
    },
    "sagemaker": {
        "endpoint_types": ["Real-time", "Serverless", "Asynchronous", "Multi-model", "Multi-container"],
        "training": {
            "spot_discount": "Up to 90%",
            "distributed_strategies": ["Data parallel", "Model parallel", "Sharded data parallel"],
        },
        "auto_scaling": {
            "types": ["Target tracking", "Step scaling", "Scheduled"],
            "cooldown_default": 300,
        },
        "clarify": {
            "bias_metrics": ["DPL", "KL", "JS", "LP", "TVD", "KS", "CDDL"],
            "explainability": ["SHAP", "Partial Dependence Plots"],
        },
    },
    "compliance": {
        "gdpr": "Requires data processing agreements, data residency, encryption, audit trails",
        "hipaa": "Bedrock is HIPAA eligible, requires BAA with AWS",
    },
}


# =============================================================================
# Question Topics
# =============================================================================

TOPICS = {
    "bedrock": [
        {"topic": "Bedrock Agents with action groups - autonomous multi-step task execution", "difficulty": "hard", "doc_ref": "agents"},
        {"topic": "Bedrock Agents with OpenAPI schemas - defining action group interfaces", "difficulty": "hard", "doc_ref": "agents"},
        {"topic": "Knowledge Bases chunking strategies - fixed vs semantic vs hierarchical", "difficulty": "hard", "doc_ref": "knowledge-bases"},
        {"topic": "Knowledge Bases hybrid search - semantic and keyword search fusion", "difficulty": "hard", "doc_ref": "knowledge-bases"},
        {"topic": "Knowledge Bases metadata filtering - pre vs post filtering", "difficulty": "medium", "doc_ref": "knowledge-bases"},
        {"topic": "Guardrails denied topics - custom topic policies", "difficulty": "medium", "doc_ref": "guardrails"},
        {"topic": "Guardrails PII filters - redaction vs blocking", "difficulty": "medium", "doc_ref": "guardrails"},
        {"topic": "Guardrails content filters - threshold configuration", "difficulty": "medium", "doc_ref": "guardrails"},
        {"topic": "Guardrails contextual grounding - hallucination detection", "difficulty": "hard", "doc_ref": "guardrails"},
        {"topic": "Fine-tuning vs continued pre-training decision criteria", "difficulty": "hard", "doc_ref": "fine-tuning"},
        {"topic": "Fine-tuning hyperparameters optimization", "difficulty": "hard", "doc_ref": "fine-tuning"},
        {"topic": "Fine-tuning data preparation - JSONL format", "difficulty": "medium", "doc_ref": "fine-tuning"},
        {"topic": "Provisioned Throughput - 1 month vs 6 month commitment", "difficulty": "medium", "doc_ref": "provisioned"},
        {"topic": "Provisioned Throughput model units calculation", "difficulty": "hard", "doc_ref": "provisioned"},
        {"topic": "Cross-region inference - failover and latency optimization", "difficulty": "hard", "doc_ref": "inference"},
        {"topic": "Batch inference jobs - S3 input/output processing", "difficulty": "medium", "doc_ref": "inference"},
        {"topic": "Model evaluation jobs - automated benchmarking", "difficulty": "medium", "doc_ref": "models"},
        {"topic": "Multi-modal Claude 3 - image analysis capabilities", "difficulty": "medium", "doc_ref": "models"},
        {"topic": "Converse API vs InvokeModel API patterns", "difficulty": "medium", "doc_ref": "inference"},
        {"topic": "Streaming responses - InvokeModelWithResponseStream", "difficulty": "medium", "doc_ref": "inference"},
        {"topic": "Bedrock VPC endpoints - PrivateLink configuration", "difficulty": "hard", "doc_ref": "security"},
        {"topic": "Bedrock IAM policies - least privilege patterns", "difficulty": "medium", "doc_ref": "security"},
        {"topic": "Cross-account model sharing patterns", "difficulty": "hard", "doc_ref": "security"},
        {"topic": "Prompt caching - latency and cost optimization", "difficulty": "medium", "doc_ref": "inference"},
        {"topic": "Inference parameters - temperature, top_p, top_k", "difficulty": "medium", "doc_ref": "inference"},
        {"topic": "Token limits and context windows comparison", "difficulty": "medium", "doc_ref": "models"},
        {"topic": "Bedrock Flows - visual workflow orchestration", "difficulty": "medium", "doc_ref": "agents"},
        {"topic": "Model lifecycle - deprecation and migration", "difficulty": "hard", "doc_ref": "models"},
        {"topic": "CloudWatch metrics - InvocationLatency monitoring", "difficulty": "medium", "doc_ref": "quotas"},
        {"topic": "Throttling and quota management strategies", "difficulty": "medium", "doc_ref": "quotas"},
    ],
    "sagemaker": [
        {"topic": "SageMaker JumpStart vs Bedrock trade-offs", "difficulty": "hard", "doc_ref": "jumpstart"},
        {"topic": "Distributed training - data parallel vs model parallel", "difficulty": "hard", "doc_ref": "training"},
        {"topic": "Inference endpoints - real-time vs serverless vs async", "difficulty": "hard", "doc_ref": "inference"},
        {"topic": "Multi-model endpoints cost optimization", "difficulty": "hard", "doc_ref": "inference"},
        {"topic": "SageMaker Pipelines MLOps automation", "difficulty": "hard", "doc_ref": "pipelines"},
        {"topic": "Model Monitor drift detection", "difficulty": "medium", "doc_ref": "model-monitor"},
        {"topic": "SageMaker Clarify bias detection", "difficulty": "hard", "doc_ref": "clarify"},
        {"topic": "Feature Store online vs offline patterns", "difficulty": "medium", "doc_ref": "feature-store"},
        {"topic": "Hyperparameter tuning - Bayesian vs random", "difficulty": "medium", "doc_ref": "training"},
        {"topic": "Spot instances for training optimization", "difficulty": "medium", "doc_ref": "training"},
        {"topic": "Auto-scaling - target tracking vs step scaling", "difficulty": "hard", "doc_ref": "inference"},
        {"topic": "Model registry versioning workflows", "difficulty": "medium", "doc_ref": "pipelines"},
        {"topic": "Ground Truth labeling workflows", "difficulty": "medium", "doc_ref": "training"},
        {"topic": "Processing jobs for data preprocessing", "difficulty": "medium", "doc_ref": "training"},
    ],
    "architecture": [
        {"topic": "Vector database selection - OpenSearch vs Aurora pgvector", "difficulty": "hard", "doc_ref": "knowledge-bases"},
        {"topic": "RAG architecture - query routing and re-ranking", "difficulty": "hard", "doc_ref": "knowledge-bases"},
        {"topic": "Step Functions with Bedrock orchestration", "difficulty": "hard", "doc_ref": "agents"},
        {"topic": "Lambda integration for Bedrock pre/post processing", "difficulty": "medium", "doc_ref": "agents"},
        {"topic": "EventBridge event-driven GenAI patterns", "difficulty": "medium", "doc_ref": "agents"},
        {"topic": "API Gateway rate limiting for Bedrock", "difficulty": "medium", "doc_ref": "security"},
        {"topic": "ElastiCache for embedding caching", "difficulty": "medium", "doc_ref": "inference"},
        {"topic": "Multi-region GenAI disaster recovery", "difficulty": "hard", "doc_ref": "security"},
        {"topic": "Cost optimization - model selection and batching", "difficulty": "hard", "doc_ref": "provisioned"},
        {"topic": "Security architecture - encryption and VPC isolation", "difficulty": "hard", "doc_ref": "security"},
        {"topic": "Responsible AI - bias testing and human oversight", "difficulty": "hard", "doc_ref": "guardrails"},
        {"topic": "Hybrid Bedrock-SageMaker architectures", "difficulty": "hard", "doc_ref": "models"},
    ],
    "ai-services": [
        {"topic": "Amazon Kendra vs Knowledge Bases comparison", "difficulty": "hard", "doc_ref": "kendra"},
        {"topic": "Amazon Textract integration with Bedrock", "difficulty": "medium", "doc_ref": "textract"},
        {"topic": "Amazon Comprehend for NLP preprocessing", "difficulty": "medium", "doc_ref": "comprehend"},
        {"topic": "Amazon Transcribe to LLM pipelines", "difficulty": "medium", "doc_ref": "transcribe"},
        {"topic": "Rekognition vs multi-modal models trade-offs", "difficulty": "medium", "doc_ref": "rekognition"},
    ],
}


# =============================================================================
# Main Generator Class
# =============================================================================

class VerifiedQuizGenerator:
    """Generate and verify AWS GenAI quiz questions using dual-model approach."""
    
    def __init__(self, critic_enabled: bool = True):
        """Initialize with Azure OpenAI credentials."""
        load_dotenv()
        
        api_key = os.getenv('AZURE_OPENAI_KEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.generator_model = os.getenv('AZURE_GENERATOR_MODEL', 'gpt-4o')
        self.critic_model = os.getenv('AZURE_CRITIC_MODEL', 'gpt-4o')
        
        if not api_key or not endpoint:
            print("❌ Missing credentials in .env file")
            print("   Required: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT")
            print("   Optional: AZURE_GENERATOR_MODEL, AZURE_CRITIC_MODEL")
            exit(1)
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint
        )
        
        self.critic_enabled = critic_enabled
        self.stats = {"generated": 0, "approved": 0, "rejected": 0, "needs_review": 0}
    
    def _get_doc_url(self, category: str, doc_ref: str) -> str:
        """Get documentation URL for citation."""
        if category in ["bedrock", "sagemaker"]:
            base = AWS_DOCS[category]["base"]
            page = AWS_DOCS[category]["pages"].get(doc_ref, "")
            return f"{base}{page}"
        elif category == "ai-services":
            return AWS_DOCS["ai-services"].get(doc_ref, "")
        return ""
    
    def _generate_question(self, topic_info: Dict, category: str) -> Optional[Dict]:
        """Generate a single question."""
        topic = topic_info["topic"]
        difficulty = topic_info["difficulty"]
        doc_url = self._get_doc_url(category, topic_info.get("doc_ref", ""))
        
        prompt = f"""Generate ONE AWS GenAI Pro certification level exam question.

TOPIC: {topic}
DIFFICULTY: {difficulty.upper()}
CATEGORY: {category}

REQUIREMENTS:
1. Complex scenario with specific company context and constraints
2. Multiple competing requirements (cost vs performance vs security)
3. Technical depth: API names, limits, pricing implications
4. All 4 options must be plausible

FORMAT:
"A [industry] company needs to [requirement] with [constraints]. Which solution BEST meets these requirements?"

OUTPUT as JSON only:
{{
  "question": "150-250 word scenario...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct_answer": "A",
  "explanation": "Option A is correct because... Option B fails because...",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "category": "{category}",
  "aws_services": ["service1", "service2"],
  "documentation_reference": "{doc_url}"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.generator_model,
                messages=[
                    {"role": "system", "content": "You are an AWS certification exam writer creating professional-level questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            
            question = json.loads(content)
            question["generated_by"] = self.generator_model
            question["generated_at"] = datetime.now().isoformat()
            self.stats["generated"] += 1
            return question
            
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            return None
    
    def _critique_question(self, question: Dict) -> Dict:
        """Evaluate question quality using critic model."""
        facts_json = json.dumps(AWS_FACTS, indent=2)
        
        prompt = f"""Review this AWS certification question for accuracy.

AWS FACTS:
{facts_json}

QUESTION: {question['question']}

OPTIONS:
A: {question['options']['A']}
B: {question['options']['B']}
C: {question['options']['C']}
D: {question['options']['D']}

CORRECT ANSWER: {question['correct_answer']}
EXPLANATION: {question['explanation']}

EVALUATE:
1. Is the correct answer actually correct?
2. Are there factual errors about AWS services?
3. Could another option also be correct?

OUTPUT as JSON:
{{
  "overall_score": 8.0,
  "is_correct_answer_valid": true,
  "alternative_could_be_correct": false,
  "factual_errors": [],
  "potential_issues": [],
  "verification_status": "APPROVED",
  "confidence": 0.9
}}

SCORING: 7-10=APPROVE, 5-6=NEEDS_REVIEW, <5=REJECT
Only REJECT for clear factual errors. Be pragmatic for practice questions."""

        try:
            response = self.client.chat.completions.create(
                model=self.critic_model,
                messages=[
                    {"role": "system", "content": "You are an AWS expert reviewing exam questions. Be pragmatic, not perfectionist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            
            return json.loads(content)
            
        except Exception as e:
            return {"overall_score": 5, "verification_status": "NEEDS_REVIEW", "potential_issues": [str(e)]}
    
    def _process_question(self, topic_info: Dict, category: str, num: int, total: int) -> Optional[Dict]:
        """Generate and verify a single question."""
        print(f"\n[{num}/{total}] 📝 {topic_info['topic'][:50]}...")
        
        question = self._generate_question(topic_info, category)
        if not question:
            return None
        
        print(f"   ✅ Generated ({topic_info['difficulty']})")
        
        if not self.critic_enabled:
            question["verification_status"] = "UNVERIFIED"
            return question
        
        print(f"   🔍 Verifying...")
        critique = self._critique_question(question)
        question["critique"] = critique
        
        score = critique.get("overall_score", 0)
        has_errors = bool(critique.get("factual_errors", []))
        alt_correct = critique.get("alternative_could_be_correct", False)
        
        # Decision logic
        if has_errors or alt_correct or score < 4.5:
            reason = critique.get("factual_errors", ["Low score"])[:1]
            print(f"   ❌ REJECTED ({score:.1f}) - {reason}")
            question["verification_status"] = "REJECTED"
            self.stats["rejected"] += 1
            return None
        elif score >= 6.5:
            print(f"   ✅ APPROVED ({score:.1f})")
            question["verification_status"] = "APPROVED"
            self.stats["approved"] += 1
        elif score >= 5.5:
            print(f"   ⚠️ APPROVED with notes ({score:.1f})")
            question["verification_status"] = "APPROVED_WITH_NOTES"
            self.stats["approved"] += 1
        else:
            print(f"   ⚠️ NEEDS REVIEW ({score:.1f})")
            question["verification_status"] = "NEEDS_REVIEW"
            self.stats["needs_review"] += 1
        
        return question
    
    def generate_quiz(self, total: int, output_file: str):
        """Generate verified quiz."""
        print(f"\n{'='*60}")
        print("🤖 AWS GenAI Pro Question Generator")
        print(f"{'='*60}")
        print(f"🎯 Target: {total} questions")
        print(f"📝 Generator: {self.generator_model}")
        print(f"🔍 Critic: {self.critic_model}")
        print(f"⏱️ Est. time: {total * 20 // 60}-{total * 30 // 60} min")
        
        # Distribution: 60% Bedrock, 20% SageMaker, 10% Architecture, 10% AI Services
        dist = {
            "bedrock": int(total * 0.60),
            "sagemaker": int(total * 0.20),
            "architecture": int(total * 0.10),
            "ai-services": total - int(total * 0.90),
        }
        
        print(f"\n📦 Distribution: {dist}")
        
        questions = []
        qid = 1
        
        for category, target in dist.items():
            print(f"\n{'─'*40}\n📂 {category.upper()}\n{'─'*40}")
            
            pool = TOPICS.get(category, [])
            if not pool:
                continue
            
            extended = pool * (target // len(pool) + 2)
            random.shuffle(extended)
            
            count = 0
            attempts = 0
            max_attempts = target * 2
            
            while count < target and attempts < max_attempts:
                topic = extended[attempts % len(extended)]
                question = self._process_question(topic, category, qid, total)
                
                if question:
                    question['id'] = qid
                    questions.append(question)
                    qid += 1
                    count += 1
                
                attempts += 1
                time.sleep(0.3)
        
        # Save
        output = {
            "title": "AWS GenAI Pro - Practice Questions",
            "generated_at": datetime.now().isoformat(),
            "models": {"generator": self.generator_model, "critic": self.critic_model},
            "total_questions": len(questions),
            "statistics": self.stats,
            "questions": questions,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Questions: {len(questions)}")
        print(f"   Approved: {self.stats['approved']}")
        print(f"   Needs Review: {self.stats['needs_review']}")
        print(f"   Rejected: {self.stats['rejected']}")
        if self.stats['generated'] > 0:
            rate = self.stats['approved'] / self.stats['generated'] * 100
            print(f"📈 Approval rate: {rate:.0f}%")
        print(f"\n💾 Saved: {output_file}")
        print(f"📖 Run: python take_quiz.py -q {output_file}")
        print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate verified AWS GenAI Pro certification questions'
    )
    parser.add_argument('-n', '--num', type=int, default=50, help='Number of questions')
    parser.add_argument('-o', '--output', default='verified_quiz.json', help='Output file')
    parser.add_argument('--no-critic', action='store_true', help='Skip verification')
    
    args = parser.parse_args()
    
    generator = VerifiedQuizGenerator(critic_enabled=not args.no_critic)
    generator.generate_quiz(args.num, args.output)


if __name__ == "__main__":
    main()
