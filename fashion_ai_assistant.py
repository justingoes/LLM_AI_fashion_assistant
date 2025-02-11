import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import chainlit as cl
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Constants
OPENAI_MODEL = "gpt-3.5-turbo"
CACHE_SIMILARITY_THRESHOLD = 0.975

# Seasonal preferences configuration
SEASONAL_PREFERENCES = {
    "january": ["winter coats", "warm layers", "wool sweaters"],
    "february": ["winter to spring transition pieces", "light layers"],
    "march": ["spring essentials", "light jackets", "transitional pieces"],
    "april": ["spring dresses", "light layers", "rain gear"],
    "may": ["spring to summer pieces", "lightweight fabrics"],
    "june": ["summer dresses", "breathable fabrics", "bright colors"],
    "july": ["summer essentials", "lightweight", "beach-ready pieces"],
    "august": ["late summer pieces", "transitional items"],
    "september": ["fall essentials", "light layers", "autumn colors"],
    "october": ["fall layers", "cozy knits", "boots"],
    "november": ["winter preparation", "warm layers", "cozy pieces"],
    "december": ["winter essentials", "holiday attire", "formal wear"]
}

# Clothing categories
CLOTHING_CATEGORIES = [
    "Dresses",
    "Tops",
    "Bottoms",
    "Outerwear",
    "Suits",
    "Jumpsuits",
    "Knits",
    "Activewear",
]

# Use cases
USE_CASES = [
    "Casual",
    "Work",
    "Formal",
    "Party",
    "Wedding Guest",
    "Vacation",
    "Date Night",
    "Special Occasion",
]

class Meta(BaseModel):
    """Response metadata"""
    sources: List[str] = []
    tags: List[str] = []

class SearchResponse(BaseModel):
    """Search response model"""
    suggestions: List[str]
    meta: Meta
    use_cases: List[str] = []
    seasonality: str = ""

@dataclass
class StyleSuggestion:
    """Style suggestion data class"""
    items: List[str]
    use_cases: List[str]
    seasonality: str
    exact_matches: List[str] = None
    similar_matches: List[str] = None
    is_inappropriate: bool = False

class FashionAIAssistant:
    """Fashion AI Assistant core class"""
    
    def __init__(self):
        self.client = AsyncOpenAI()
        self._cache = {}
        
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query text"""
        query = query.lower()
        # Standardize common terms
        replacements = {
            r"\bwedding\b": "wedding guest",
            r"\bsuit\b": "suit jacket & pants",
            r"\bblack tie optional\b": "cocktail",
            r"\bmother of the bride\b": "formal wedding guest",
            r"\bslutty\b": "sexy",  # More appropriate term
        }
        
        for pattern, replacement in replacements.items():
            query = re.sub(pattern, replacement, query)
            
        return query.strip()

    def _get_seasonal_preferences(self) -> Tuple[str, List[str]]:
        """Get current month's seasonal preferences"""
        current_month = datetime.now().strftime("%B").lower()
        preferences = SEASONAL_PREFERENCES.get(current_month, [])
        return current_month, preferences

    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        current_month, seasonal_preferences = self._get_seasonal_preferences()
        
        return f"""You are an expert fashion advisor with deep knowledge of current trends and timeless style.

Key Responsibilities:
1. Provide personalized clothing recommendations based on occasions, events, and style preferences
2. Consider seasonality and weather appropriateness
3. Focus on versatile pieces that can be styled multiple ways
4. Maintain professional and appropriate suggestions

Current Context:
- Current Month: {current_month}
- Seasonal Focus: {', '.join(seasonal_preferences)}
"""

    def _build_user_prompt(self, query: str) -> str:
        """Build user prompt"""
        current_month, seasonal_preferences = self._get_seasonal_preferences()
        seasonality_instruction = f"""
        If no specific season is mentioned, recommendations should align with current seasonal preferences:
        {', '.join(seasonal_preferences)}
        """
        
        return f"""Please provide fashion recommendations for the following request:
        "{query}"

        Instructions:
        1. Location Context: Default to New York City in {current_month} if no location specified
        2. Seasonal Appropriateness: {seasonality_instruction}
        3. Specificity Rules:
           - No accessories, shoes, or bags unless explicitly requested
           - For specific garment requests, focus only on that category
           - For designer requests, suggest their signature pieces
        4. Format Requirements:
           - Return response in valid JSON format
           - Include 3-5 specific item recommendations
           - Specify appropriate use cases
           - Indicate seasonality
        5. Response Structure:
        {{
            "items": ["item1", "item2", "item3"],
            "use_cases": ["occasion1", "occasion2"],
            "seasonality": "spring/summer or fall/winter",
            "is_inappropriate": false
        }}
        """

    async def _get_completion(
        self, 
        query: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Call OpenAI API to get completion"""
        try:
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(query)}
            ]
            
            response = ""
            completion = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in completion:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
                    
            return response
            
        except Exception as e:
            LOGGER.error(f"OpenAI API call failed: {str(e)}")
            raise

    def _post_process_suggestions(self, items: List[str]) -> List[str]:
        """Post-process recommendation results"""
        processed_items = []
        for item in items:
            # Remove pairing suggestions
            item = re.sub(r"(pair|wear) with.*", "", item, flags=re.IGNORECASE)
            # Standardize terms
            item = item.replace("cover-up", "loose blouse")
            processed_items.append(item.strip())
        return processed_items

    async def get_style_suggestions(self, query: str) -> StyleSuggestion:
        """Get style suggestions"""
        # Preprocess query
        processed_query = self._preprocess_query(query)
        
        # Check cache
        if processed_query in self._cache:
            return self._cache[processed_query]
        
        response = await self._get_completion(processed_query)
        
        try:
            suggestions = json.loads(response)
            
            # Post-process recommendations
            processed_items = self._post_process_suggestions(suggestions.get("items", []))
            
            result = StyleSuggestion(
                items=processed_items,
                use_cases=suggestions.get("use_cases", []),
                seasonality=suggestions.get("seasonality", ""),
                exact_matches=suggestions.get("exact_matches", []),
                similar_matches=suggestions.get("similar_matches", []),
                is_inappropriate=suggestions.get("is_inappropriate", False)
            )
            
            # Update cache
            self._cache[processed_query] = result
            return result
            
        except json.JSONDecodeError:
            LOGGER.error(f"Failed to parse response: {response}")
            return StyleSuggestion([], [], "")

# FastAPI application
app = FastAPI(title="Fashion AI Assistant")
assistant = FashionAIAssistant()

@app.post("/search")
async def search(query: str) -> SearchResponse:
    """Search endpoint"""
    suggestion = await assistant.get_style_suggestions(query)
    if suggestion.is_inappropriate:
        raise HTTPException(status_code=400, detail="Inappropriate content detected")
        
    return SearchResponse(
        suggestions=suggestion.items,
        meta=Meta(sources=[OPENAI_MODEL]),
        use_cases=suggestion.use_cases,
        seasonality=suggestion.seasonality
    )

# Chainlit chat interface
@cl.on_chat_start
async def start():
    """Chat start handler"""
    await cl.Message(content="Hello! I'm your AI Fashion Advisor. How can I help with your style today?").send()

@cl.on_message
async def main(message: cl.Message):
    """Message handler"""
    suggestion = await assistant.get_style_suggestions(message.content)
    
    if suggestion.is_inappropriate:
        await cl.Message(content="I apologize, but I cannot provide inappropriate clothing suggestions.").send()
        return
        
    response = "Recommended Outfits:\n"
    for idx, item in enumerate(suggestion.items, 1):
        response += f"{idx}. {item}\n"
        
    if suggestion.use_cases:
        response += f"\nSuitable for: {', '.join(suggestion.use_cases)}"
    if suggestion.seasonality:
        response += f"\nSeason: {suggestion.seasonality}"
        
    await cl.Message(content=response).send()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
