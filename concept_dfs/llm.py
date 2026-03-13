import os
from pydantic import BaseModel, Field
from typing import List
from litellm import completion

class ConceptResponse(BaseModel):
    explanation: str = Field(description="A clear, concise, and practical explanation of the concept.")
    keywords: List[str] = Field(description="List of exactly 3 to 5 related sub-concepts or advanced topics to explore next.")

def fetch_concept(query: str) -> ConceptResponse:
    model = os.environ.get("CONCEPT_DFS_MODEL", "gemini/gemini-2.5-pro")
    
    system_prompt = (
        "You are an expert researcher. Given a concept, provide a highly informative explanation "
        "and exactly 3 to 5 related child-concepts or sub-topics for further exploration."
    )
    
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Concept: {query}"}
        ],
        response_format=ConceptResponse
    )
    
    content = response.choices[0].message.content
    if isinstance(content, str):
        return ConceptResponse.model_validate_json(content)
    # If the provider integration parsed it already
    return ConceptResponse.model_validate(content)
