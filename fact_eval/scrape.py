from gpt_researcher import GPTResearcher
from gpt_researcher.actions.web_scraping import scrape_urls
from gpt_researcher.utils.workers import WorkerPool

researcher = GPTResearcher(query="")


async def scrape_(citations):
    scraped_data, images = await scrape_urls(urls=citations, cfg=researcher.cfg, worker_pool=WorkerPool(1))

    results = []
    for result in scraped_data:
        url = result.get('url', '')
        title = result.get('title', '')
        content = result.get('raw_content', '')
        url_content = f"{title}\n\n{content}"
        results.append({
            'url': url,
            'url_content': url_content
        })

    return results


async def scrape(deduplicated_dict: dict) -> dict:
    citations = list([k for k, v in deduplicated_dict['citations_deduped'].items() if 'url_content' not in v or not v['url_content']])

    results = await scrape_(citations)
    scraped_dict = deduplicated_dict.copy()
    # update the url_content
    for res in results:
        scraped_dict['citations_deduped'][res['url']]['url_content'] = res['url_content']
    return scraped_dict
