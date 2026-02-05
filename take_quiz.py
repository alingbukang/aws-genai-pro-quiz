#!/usr/bin/env python3
"""AWS GenAI Pro Quiz - Practice for AWS GenAI Pro certification."""

import json
import random
import argparse
from pathlib import Path

def run_quiz(quiz_file: str, num_questions: int = 10):
    """Run interactive quiz from a JSON question file."""
    path = Path(quiz_file)
    if not path.exists():
        print(f"❌ {quiz_file} not found!")
        return
    
    questions = json.loads(path.read_text(encoding='utf-8')).get('questions', [])
    if not questions:
        print("❌ No questions found!")
        return
    
    random.shuffle(questions)
    selected = questions[:num_questions]
    
    print(f"\n{'='*70}\n📝 AWS GenAI Pro Quiz\n{'='*70}")
    print(f"Answering {len(selected)} questions\n")
    
    score = 0
    for i, q in enumerate(selected, 1):
        print(f"\n{'─'*70}\nQuestion {i}/{len(selected)}\n{'─'*70}\n")
        print(q['question'] + "\n")
        for opt, text in q['options'].items():
            print(f"  {opt}) {text}")
        
        answer = input("\nYour answer (A/B/C/D): ").strip().upper()
        if answer == q['correct_answer']:
            print("\n✅ Correct!")
            score += 1
        else:
            print(f"\n❌ Incorrect. Answer: {q['correct_answer']}")
        print(f"\n📖 {q['explanation']}")
        input("\nEnter to continue...")
    
    pct = score / len(selected) * 100
    print(f"\n{'='*70}\n📊 SCORE: {score}/{len(selected)} ({pct:.0f}%)\n{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWS GenAI Pro certification practice quiz")
    parser.add_argument('-q', '--quiz-file', default='aws_genai_pro.json')
    parser.add_argument('-n', '--num-questions', type=int, default=10)
    args = parser.parse_args()
    
    try:
        run_quiz(args.quiz_file, args.num_questions)
    except KeyboardInterrupt:
        print("\n\n👋 Quiz ended.")
