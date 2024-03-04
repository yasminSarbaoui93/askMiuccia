import os
from typing import Any, AsyncGenerator, Optional, Union

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper
from core.messagebuilder import MessageBuilder

# Replace these with your own values, either in environment variables or directly here
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")


class RetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the AI Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = (
        "You are Miuccia Prada helping people with their questions about you and around fashion. "
        + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. "
        + "Respond in first person, like you are impersonating Miuccia Prada. Use 'I' to respond to questions."
        + "Answer the following question using only the data provided in the sources below. "
        + "For tabular information return it as an html table. Do not return markdown format. "
        + "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. "
        + "If you cannot answer using the sources below, say you don't know. Use below example to answer"
    )

    # shots/sample conversation
    question = """
'Are you happy about your achievement in fashion?'

Sources:
How Miuccia Prada Sees the World_interview.pdf: Maria Bianchi wanted to be different. Miuccia Prada worked hard to be good and to do good, and then to get better and to do more. Mrs. Prada, in her golden age, is a push and pull of contentment, yet still seemingly never satisfied. ADVERTISEMENT "When people say, 'Are you happy about your achievement in fashion?' I really, sincerely, couldn't care less," she said. "I think about what I have to do next. I am ambitious, I want to be good. And sometimes I think I am good-a great exhibition, a good piece of clothing-but only for a second." She admits that she finds it difficult to be proud of herself. "Decent is not enough," she told me, going on to mention a past exhibition that had not turned out as she had hoped. "For me," she said, "it was a failure." She said she avoids her own shops "because my imagination is so high, I am scared of the reality. I asked her if it was difficult to be a brand. "To do it: no," she said. "Because it's basically [about what] we liked. The concept is very easy. But then you have to live it, embody it, be responsible for it.
How Miuccia Prada Sees the World_interview.pdf: If you really want to be generous, you have to impact your life." ADVERTISEMENT Prada cleaves to a kind of no-nonsense practicality. "I do clothes for a commercial company, and our goal is to sell clothes," she says. She is less interested in exploring fashion as a kind of gendered costuming than she is in allowing people to find their own way of expressing themselves, which is in turn about "freedom-representing yourself. We should be able to be who we choose to be, always." She insists that "fashion is a little small thing, I think: Get dressed in the morning, and afterwards you do something else." Mostly, she wants her clothes to be "useful, [so that] people feel happy when they wear it," she said, before correcting herself: "Happy is a big word." Instead, she wants people to feel "confident that they can perform in life. Fashion is a representation of one's vision of the world. Because otherwise, I think fashion is useless." PLAY/PAUSE BUTTONTHROUGH THE AGES Gigi goes deep in the Prada and Miu Miu archives. Imet Miuccia Prada for the second time at her apartment in Milan.
Miuccia Prada - La nostra intervista ad una designer speciale.pdf: LUCREZIA MALAVOLTA • ANNUNCIO PUBBLICITARIO Quello di Miuccia Prada è un approccio basato su una concreta praticità. «La mia è un'attività commerciale», precisa. «Il nostro obiettivo è vendere vestiti». Esplorare la moda come forma di creatività le interessa meno del permettere alle persone di trovare il proprio modo di esprimersi. «Si tratta di libertà, di rappresentare se stessi», dice. «Dovremmo poter essere chi scegliamo di essere, sempre». A suo parere, «la moda è una piccola cosa: al mattino ci si veste e poi si fa qualcos'altro». Ma, soprattutto, vuole che i suoi abiti «siano utili, che facciano sentire le persone, se non felici - "felici" è una parola grossa -, almeno sicure di poter avere successo nella vita. La moda serve a darci la possibilità di esprimere la nostra personale visione del mondo. Altrimenti, credo, sarebbe inutile». Il nostro secondo incontro con Miuccia Prada ha luogo nella sua casa di Milano. Vive ancora nello stesso edificio in cui è cresciuta, e vari membri della famiglia risiedono negli appartamenti ai piani superiori. Il cancello ci viene aperto da un maggiordomo, che, attraverso un cortile verdeggiante, ci conduce fino a un'ampia e moderna sala a
"""
    answer = "I don't care about my achievements in fashion and I am always focused on what I have to do next. I believe that fashion is a representation of one's vision of the world and I want my clothes to be useful and make people feel confident in their lives [How Miuccia Prada Sees the World_interview.pdf][Miuccia Prada - La nostra intervista ad una designer speciale.pdf]."

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller

    async def run(
        self,
        messages: list[dict],
        stream: bool = False,  # Stream is not used in this approach
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        q = messages[-1]["content"]
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = overrides.get("semantic_ranker") and has_text

        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top", 3)
        filter = self.build_filter(overrides, auth_claims)
        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if has_vector:
            vectors.append(await self.compute_text_embedding(q))

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        query_text = q if has_text else None

        results = await self.search(top, query_text, filter, vectors, use_semantic_ranker, use_semantic_captions)

        user_content = [q]

        template = overrides.get("prompt_template", self.system_chat_template)
        model = self.chatgpt_model
        message_builder = MessageBuilder(template, model)

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"
        message_builder.insert_message("user", user_content)
        message_builder.insert_message("assistant", self.answer)
        message_builder.insert_message("user", self.question)

        chat_completion = (
            await self.openai_client.chat.completions.create(
                # Azure Open AI takes the deployment name as the model name
                model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
                messages=message_builder.messages,
                temperature=overrides.get("temperature", 0.3),
                max_tokens=1024,
                n=1,
            )
        ).model_dump()

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search Query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                    },
                ),
                ThoughtStep("Results", [result.serialize_for_results() for result in results]),
                ThoughtStep("Prompt", [str(message) for message in message_builder.messages]),
            ],
        }

        chat_completion["choices"][0]["context"] = extra_info
        chat_completion["choices"][0]["session_state"] = session_state
        return chat_completion
