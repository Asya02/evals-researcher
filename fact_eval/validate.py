import time

from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat.chat_models import GigaChat

validate_prompt_template_en = """You will be provided with a reference and some statements. Please determine whether each statement is 'supported', 'unsupported', or 'unknown' with respect to the reference. Please note:
First, assess whether the reference contains any valid content. If the reference contains no valid information, such as a 'page not found' message, then all statements should be considered 'unknown'.
If the reference is valid, for a given statement: if the facts or data it contains can be found entirely or partially within the reference, it is considered 'supported' (data accepts rounding); if all facts and data in the statement cannot be found in the reference, it is considered 'unsupported'.

You should return the result in a JSON list format, where each item in the list contains the statement's index and the judgment result, for example:
[
    {{
        "idx": 1,
        "result": "supported"
    }},
    {{
        "idx": 2,
        "result": "unsupported"
    }}
]

Below are the reference and statements:
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

Begin the assessment now. Output only the JSON list, without any conversational text or explanations."""


def validate_(data):
    url = data[0]
    ref = data[1]['url_content']
    facts = data[1]['facts']

    if ref is None:
        return {
            "url": url,
            "validate_res": [],
            "error": "no reference"
        }

    facts_str = '\n'.join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])

    retries = 0
    error = None

    validate_prompt = ChatPromptTemplate([
        ("human", validate_prompt_template_en),
    ])
    llm = GigaChat(
            model="GigaChat-2-Max",
            verify_ssl_certs=False,
            profanity_check=False
        )
    validate_chain = validate_prompt | llm | JsonOutputParser()
    while retries < 3:
        try:
            validate_res = validate_chain.invoke({"reference": ref, "statements": facts_str})

            for _v in validate_res:
                _v['idx'] -= 1
            assert len(validate_res) == len(facts)

            return {
                "url": url,
                "validate_res": validate_res,
                "error": None
            }
        except Exception as e:
            error = str(e)
            time.sleep(3)
            retries += 1

    return {
        "url": url,
        "validate_res": [],
        "error": error
    }


def validate(scraped_dict: dict) -> dict:
    # get the citations that need to be validated
    citations = [(k, v) for k, v in scraped_dict['citations_deduped'].items()]

    results = [validate_(citation) for citation in citations]
    validated_dict = scraped_dict.copy()
    for res in results:
        validated_dict['citations_deduped'][res['url']]['validate_res'] = res['validate_res']
        validated_dict['citations_deduped'][res['url']]['validate_error'] = res['error']
    return validated_dict
