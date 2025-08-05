from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat.chat_models import GigaChat

deduplicate_prompt_template_en = """You will be given a list of statements. You need to de-duplicate them and return a list of indices of the unique statements. Note: Two statements are considered duplicates only if they express *exactly the same thing*. If there are no duplicate statements in the list, return the complete list of indices.

You should return a List(int), where each item in the list is the index of a unique, non-duplicated statement that has been retained. For example:
[1, 3, 5]

Below is the list of statements you need to de-duplicate:
{statements}

Please begin the extraction now. Output only the integer list, without any conversational text or explanations."""


def deduplicate(extracted_dict: dict) -> dict:
    citations = extracted_dict['citations']
    citation_groups: dict[str, list[str | dict]] = {}
    for _c in citations:
        if 'url' not in _c:
            continue
        else:
            if _c['url'] not in citation_groups:
                citation_groups[_c['url']] = []
            citation_groups[_c['url']].append(_c)

    citations_groups_deduped: dict[str, dict[str, list[str | None] | None]] = {}

    deduplicate_prompt = ChatPromptTemplate([
        ("human", deduplicate_prompt_template_en),
    ])
    llm = GigaChat(
            model="GigaChat-2-Max",
            verify_ssl_certs=False,
            profanity_check=False
        )
    deduplicate_chain = deduplicate_prompt | llm | JsonOutputParser()

    for ref_idx, group in citation_groups.items():
        if len(group) == 1:
            citations_groups_deduped[group[0]['url']] = {
                'facts': [group[0]['fact']],
                'url_content': None
            }
            continue

        statements = '\n'.join([f'{i+1}. {_c["fact"]}' for i, _c in enumerate(group)])

        retries = 0

        deduped_idx = []
        while retries < 3:
            retries += 1
            try:
                deduped_idx = deduplicate_chain.invoke({"statements": statements})
                break
            except Exception as e:
                print(repr(e))
                continue

        # if the model failed to deduplicate, use the default deduplication
        if not deduped_idx or 0 in deduped_idx or len(deduped_idx) > len(group):
            deduped_idx = [i+1 for i in range(len(group))]

        # deduplicate the citations by url
        citations_groups_deduped[group[0]['url']] = {
            'facts': [group[i-1]['fact'] for i in deduped_idx],
            'url_content': None
        }

    deduplicated_dict = extracted_dict.copy()
    deduplicated_dict['citations_deduped'] = citations_groups_deduped
    return deduplicated_dict
