from typing import TypedDict
import os
from rich import print

class PipelineState(TypedDict):
    raw_input : str
    edited_text : str
    script_text :  str
    final_output : str




# print("Your Loaded Key is:", os.environ.get("GROQ_API_KEY"))


llm = ChatGroq(model="openai/gpt-oss-120b" , temperature=0.7, api_key=os.environ.get("GROQ_API_KEY")  )

def editor_node(state = PipelineState) -> dict:
    """Stage 1 : Cleans up grammar, removes typos and refines the tone."""

    prompt = (
        "You are a precise text editor. Your sole task is to correct grammatical mistakes, fix spelling errors, and improve overall readability."
        "Maintain the original context, tone, and meaning exactly as provided."
        "Preserve the author's voice without adding new information or rewriting the core message"
        "Return only the final, edited text. "
        f"Text : \n {state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {'edited_text' : response.content.strip()}



def scriptwriter_node(state = PipelineState) -> dict:
    """Stage 2 : Formats the clean text into an engaging video script style"""

    prompt = (
        """provided text into a highly engaging, funny, and conversational spoken script.
        Core Requirements:
        • Sound like a real person: Write exactly how a passionate human speaks. Use natural contractions, rhetorical questions, and enthusiastic pacing.
        • Inject humor: Weave in clever punchlines, witty observations, and relatable comedic timing.
        • Keep it interesting: Hook the audience immediately and maintain a dynamic, captivating rhythm throughout.
        • Maintain the core message: Make it wildly entertaining, but ensure the central message and context of the original text remain intact.
        Formatting Rules:
        • Return only the spoken text. Do not include stage directions, audio cues, labels (like "Speaker:"), introductory remarks, or markdown code blocks.
        • Write the response in plain text, optimized for a natural verbal performance."""
        f"Text : \n {state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {'script_text' : response.content.strip()}


def translator_node(state = PipelineState) -> dict:
    """Stage 3 : Translates the script into natural flowing Hinglish"""

    prompt = (
        """You are a native Hinglish speaker and natural voice artist. Your task is to translate the provided script into authentic, high-energy Hinglish (a seamless, natural blend of Hindi and English).
        Core Requirements:
        • Capture the natural flow: Do not translate sentence-by-sentence. Blend Hindi and English dynamically, exactly how a real person from Delhi or Mumbai talks when they are speaking passionately.
        • Keep the vibe intact: Preserve all the humor, energy, punchlines, and emotional beats from the original script.
        • Maintain the context: The core message and meaning must remain 100% unchanged.
        Formatting Rules:
        • Return only the final Hinglish text. Do not include any translation notes, explanations, introductory remarks, or markdown formatting blocks.
        • Use standard English alphabet characters for the Hindi words (e.g., write "Yeh toh kamaal hai" instead of Devnagari script)""" 
        f"Text : \n {state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {'final_output' : response.content.strip()}


# now your states and nodes are ready and 
# now it is time to create a graph. 
# For creating the graph you have to connect these nodes for that you have to use the edges,
# edges ae very inportant to create the workflows




from langgraph.graph import StateGraph , START , END

# create the graph
graph = StateGraph(PipelineState)

# add the nodes in the graph

graph.add_node("editor" , editor_node)
graph.add_node("scriptWriter" , scriptwriter_node)
graph.add_node("translator" , translator_node)


# Add Edges (Sequential)

graph.add_edge(START , "editor")
graph.add_edge( "editor", "scriptWriter" )
graph.add_edge("scriptWriter" , "translator")
graph.add_edge("translator" , END)

# compile the graph

app = graph.compile()

result = app.invoke({
    "raw_input" : "AI is changing everything very fastly. everyday new framework coming like langgraph. "
    "people think it is hard to learn coding but i think if you try daily you can code easily. "
    "dont give up because future is fully automated and we need to learn this technical things."
})



print("=" * 60)
print("1. ORIGINAL RAW INPUT:")
print("-" * 60)
print(result["raw_input"])
print("\n" + "=" * 60)

print("2. EDITED TEXT (Cleaned Grammar & Readability):")
print("-" * 60)
print(result["edited_text"])
print("\n" + "=" * 60)

print("3. VIDEO SCRIPT TEXT (Funny, Human & Engaging Style):")
print("-" * 60)
print(result["script_text"])
print("\n" + "=" * 60)

print("4. FINAL OUTPUT (Natural High-Energy Hinglish Translation):")
print("-" * 60)
print(result["final_output"])
print("=" * 60)