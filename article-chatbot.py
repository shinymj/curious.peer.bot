# 추가질문 생성하고 추가 답변 입력까지 성공

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_community.chat_message_histories import ChatMessageHistory
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Any

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

class ArticleUnderstandingBot:
    def __init__(self, article_text: str, min_score: int = 10):
        self.llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.article = article_text
        self.chat_history = ChatMessageHistory()
        self.min_score = min_score
        
        # Stage 1: Initial Understanding Assessment
        self.initial_questions_prompt = PromptTemplate(
            input_variables=["article"],
            template="""
            다음 글에 대한 기본적인 이해를 평가하기 위한 3가지 핵심 질문을 생성해주세요:
            {article}
            
            질문은 다음을 평가할 수 있어야 합니다:
            1. 핵심 개념 이해도
            2. 주요 논점 파악
            3. 논리적 설명 능력
            """
        )
        
        self.assessment_prompt = PromptTemplate(
            input_variables=["article", "questions", "response"],
            template="""
            다음 답변을 평가해주세요:
            
            원문: {article}
            질문: {questions}
            답변: {response}
            
            각 항목을 0-5점으로 평가하고, 구체적인 피드백을 제공해주세요:
            1. 핵심 개념 이해도
            2. 주요 논점 파악
            3. 논리적 설명 능력
            
            JSON 형식으로 반환:
            {{
                "scores": {{
                    "concept": int,
                    "main_points": int,
                    "explanation": int
                }},
                "total": int,
                "feedback": str,
                "areas_for_improvement": List[str]
            }}
            """
        )
        
        # Remedial Learning
        self.remedial_prompt = PromptTemplate(
            input_variables=["article", "areas_for_improvement"],
            template="""
            다음 부족한 부분들에 대한 이해를 돕기 위한 추가 질문을 2-3개 생성해주세요:
            
            원문: {article}
            부족한 부분: {areas_for_improvement}
            
            질문은 구체적이고 학생의 이해를 돕는 방향이어야 합니다.
            """
        )
        
        # Stage 2: Critical Thinking
        self.critical_prompt = PromptTemplate(
            input_variables=["article", "response"],
            template="""
            학생의 답변을 바탕으로 비판적 사고를 위한 심층 질문을 생성해주세요:
            
            원문: {article}
            학생 답변: {response}
            
            다음 영역에서 1-2개의 질문을 생성해주세요:
            1. 주장의 타당성 검증
            2. 대안적 관점 고려
            3. 실제 적용 가능성
            4. 잠재적 한계점
            """
        )
        
        # Response Quality Check
        self.quality_check_prompt = PromptTemplate(
            input_variables=["response", "question"],
            template="""
            다음 답변의 깊이와 품질을 평가해주세요:
            
            질문: {question}
            답변: {response}
            
            JSON 형식으로 반환:
            {{
                "quality": "sufficient" 또는 "needs_depth",
                "feedback": str,
                "suggested_followup": str if needs_depth else null
            }}
            """
        )
        
        # Stage 3: Final Synthesis
        self.synthesis_prompt = PromptTemplate(
            input_variables=["article", "conversation_history"],
            template="""
            전체 대화를 바탕으로 학생이 최종 생각을 정리할 수 있는 구조를 제시해주세요:
            
            원문: {article}
            대화 내용: {conversation_history}
            
            다음 구조로 작성하도록 안내해주세요:
            1. 글의 핵심 주장
            2. 비판적 분석
            3. 자신의 관점과 근거
            """
        )

    def start_initial_assessment(self) -> Dict[str, Any]:
        """Stage 1: Generate initial questions and start assessment"""
        chain = self.initial_questions_prompt | self.llm
        questions_message = chain.invoke({"article": self.article})
        # Extract the actual content from the Message object
        questions = questions_message.content if hasattr(questions_message, 'content') else str(questions_message)
        return {"questions": questions}
    
    def assess_understanding(self, questions: str, response: str) -> Dict[str, Any]:
        """Evaluate student's understanding and determine next steps"""
        assess_chain = self.assessment_prompt | self.llm
        result_message = assess_chain.invoke({
            "article": self.article,
            "questions": questions,
            "response": response
        })
        result = result_message.content if hasattr(result_message, 'content') else str(result_message)
        assessment = json.loads(result)
        
        if assessment["total"] < self.min_score:
            remedial_chain = self.remedial_prompt | self.llm
            remedial_message = remedial_chain.invoke({
                "article": self.article,
                "areas_for_improvement": assessment["areas_for_improvement"]
            })
            remedial_questions = remedial_message.content if hasattr(remedial_message, 'content') else str(remedial_message)
            return {
                "status": "needs_remedial",
                "feedback": assessment["feedback"],
                "remedial_questions": remedial_questions
            }
        
        return {
            "status": "ready_for_critical",
            "feedback": assessment["feedback"]
        }
    
    def generate_critical_questions(self, response: str) -> str:
        """Stage 2: Generate critical thinking questions"""
        chain = self.critical_prompt | self.llm
        message = chain.invoke({
            "article": self.article,
            "response": response
        })
        return message.content if hasattr(message, 'content') else str(message)
    
    def check_response_quality(self, question: str, response: str) -> Dict[str, Any]:
        """Evaluate the quality of student's critical thinking response"""
        chain = self.quality_check_prompt | self.llm
        message = chain.invoke({
            "question": question,
            "response": response
        })
        result = message.content if hasattr(message, 'content') else str(message)
        return json.loads(result)
    
    def guide_synthesis(self) -> str:
        """Stage 3: Guide final synthesis"""
        chain = self.synthesis_prompt | self.llm
        messages = self.chat_history.messages
        conversation_history = "\n".join([msg.content for msg in messages])
        message = chain.invoke({
            "article": self.article,
            "conversation_history": conversation_history
        })
        return message.content if hasattr(message, 'content') else str(message)

def clean_text(text: str) -> str:
    """Clean the input text from potential markdown or special characters"""
    # Remove markdown headers
    text = text.replace('#', '')
    # Add more cleaning rules if needed
    return text.strip()

def main():
    print("""
아티클 입력 가이드:
1. 일반 텍스트로 입력해주세요 (마크다운이나 특수문자 없이)
2. 여러 줄 입력이 가능합니다
3. 입력을 완료하려면 빈 줄(Enter 두 번)을 입력하세요
예시:
제목
이것은 본문입니다.
두 번째 문단입니다.
(빈 줄 입력으로 완료)
    """)
    
    # Get article input
    print("\n아티클을 입력해주세요:")
    article_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        article_lines.append(clean_text(line))
    
    article = "\n".join(article_lines)
    
    # Validate article input
    if not article.strip():
        print("Error: Article text cannot be empty. Please provide an article to discuss.")
        return
        
    print("\nStarting discussion about the article...")
    bot = ArticleUnderstandingBot(article)
    
    # Stage 1: Initial Assessment
    initial = bot.start_initial_assessment()
    print("\nTo understand your comprehension of the article, please answer these questions:")
    print(initial["questions"])
    
    # Get student response and assess
    print("\nPlease provide your response (press Enter twice when done):")
    response_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        response_lines.append(line)
    
    response = "\n".join(response_lines)
    if not response.strip():
        print("Error: Response cannot be empty. Please provide your thoughts about the article.")
        return
        
    assessment = bot.assess_understanding(initial["questions"], response)
    
    if assessment["status"] == "needs_remedial":
        print("\n=== 평가 결과 ===")
        print(f"피드백: {assessment['feedback']}")
        print("\n=== 추가 질문 ===")
        print(assessment["remedial_questions"])
        
        # Get response for remedial questions
        print("\n추가 질문에 대한 답변을 입력해주세요 (완료하려면 Enter 두 번):")
        remedial_response_lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            remedial_response_lines.append(line)
            
        remedial_response = "\n".join(remedial_response_lines)
        if not remedial_response.strip():
            print("Error: Response cannot be empty.")
            return
            
        # Reassess understanding with remedial response
        reassessment = bot.assess_understanding(assessment["remedial_questions"], remedial_response)
        # Continue remedial loop...
    else:
        # Stage 2: Critical Thinking
        critical_questions = bot.generate_critical_questions(response)
        print("Critical Thinking Questions:", critical_questions)
        
        # Get and evaluate critical thinking response
        critical_response = input("Your critical analysis: ")
        quality_check = bot.check_response_quality(critical_questions, critical_response)
        
        if quality_check["quality"] == "sufficient":
            # Stage 3: Final Synthesis
            synthesis_guide = bot.guide_synthesis()
            print("Synthesis Guide:", synthesis_guide)
        else:
            print("Follow-up Question:", quality_check["suggested_followup"])

if __name__ == "__main__":
    main()