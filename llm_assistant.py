import streamlit as st
import os
import openai
from groq import Groq
from typing import List
import random

class LLMAssistant:
    def __init__(self, model_type="groq"):
        self.model_type = model_type
        self.openai_client = None
        self.groq_client = None
        self.conversation_context = []
        
        if model_type == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
        elif model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
        
        self.system_prompt = """You are an AI sales assistant helping a sales representative during a live call. 
        
        Your role is to provide SHORT, actionable suggestions based on the conversation transcript. 
        
        Guidelines:
        - Keep suggestions to 1-2 sentences maximum
        - Focus on sales techniques, objection handling, and relationship building
        - Provide specific, actionable advice
        - Use these categories:
          💡 Tip - General sales advice
          ⚠️ Reminder - Important things not to forget
          ❗ Alert - Urgent actions or red flags
          🎯 Close - Closing opportunities
        
        Only respond with the suggestion, starting with the appropriate emoji category.
        If no specific advice is needed, respond with "No suggestions at this time."
        """
    
    def get_suggestions(self, transcript_chunk: str) -> List[str]:
        """Get AI suggestions based on transcript"""
        try:
            if not transcript_chunk.strip():
                return []
                
            if self.model_type == "groq" and self.groq_client:
                response = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",  # Groq's fast Llama model
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Recent conversation: {transcript_chunk}"}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                
                suggestion = response.choices[0].message.content.strip()
                if suggestion and suggestion != "No suggestions at this time.":
                    return [suggestion]
                return []
            elif self.model_type == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Recent conversation: {transcript_chunk}"}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                
                suggestion = response.choices[0].message.content.strip()
                if suggestion and suggestion != "No suggestions at this time.":
                    return [suggestion]
                return []
            else:
                # Mock suggestions for demo
                mock_suggestions = [
                    "💡 Tip: Ask about their current challenges",
                    "⚠️ Reminder: Mention the ROI benefits",
                    "❗ Alert: Customer mentioned budget concerns",
                    "🎯 Close: Good time to ask for next steps"
                ]
                return [random.choice(mock_suggestions)]
                
        except Exception as e:
            st.error(f"LLM error: {e}")
            return []