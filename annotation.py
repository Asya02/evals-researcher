import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    return


@app.cell
def _():
    import json

    with open('data/processed/research_reports_2025-06-16_17-13-29.json', encoding="utf-8") as f:
        examples = json.load(f)
    examples[0]
    return (examples,)


@app.cell
def _(examples):
    from molabel import SimpleLabel
    from IPython.display import HTML
    import markdown


    from markdown.extensions.tables import TableExtension

    def render_example(example):
        # Используем расширение для таблиц
        html = markdown.markdown(
            "**Вопрос: **" + example['inputs']['task']['query'] + '\n\n\n**Отчет:**\n\n' + example['outputs']['report'], 
            extensions=[TableExtension()]
        )
        return html

    # Create annotation widget
    widget = SimpleLabel(
        examples=examples,
        render=render_example,
        notes=True
    )

    # Display in notebook
    widget
    return (widget,)


@app.cell
def _(widget):
    # Get annotations after labeling
    annotations = widget.get_annotations()
    annotations[0]
    return


@app.cell
def _():
    import pickle
    return (pickle,)


@app.cell
def _(mo):
    mo.md(
        r"""
    with open('data/annotations/annotations_2025-07-03.pkl', 'wb') as file:
        pickle.dump(annotations, file)
    """
    )
    return


@app.cell
def _(pickle):
    with open('data/annotations/annotations_2025-07-03.pkl', 'rb') as file_:
        annotations_ = pickle.load(file_)
    annotations_[0]
    return (annotations_,)


@app.cell
def _(annotations_):
    errors = []
    for annotation in annotations_:
        errors.extend(annotation['_notes'].split('\n'))
    len(errors)
    return (errors,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Clustering""")
    return


@app.cell
def _():
    from bertopic.backend import BaseEmbedder
    from langchain_gigachat.embeddings import GigaChatEmbeddings

    class CustomEmbedder(BaseEmbedder):
        def __init__(self, embedding_model):
            super().__init__()
            self.embedding_model = embedding_model

        def embed(self, documents, verbose=False):
            embeddings = self.embedding_model.embed_documents(documents)
            return embeddings 

    # Create custom backend
    emb_m = GigaChatEmbeddings(model="EmbeddingsGigaR", verify_ssl_certs=False)
    custom_embedder = CustomEmbedder(embedding_model=emb_m)
    return (custom_embedder,)


@app.cell
def _():
    from umap import UMAP

    umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.0, metric='cosine')
    return (umap_model,)


@app.cell
def _():
    from hdbscan import HDBSCAN

    hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    return (hdbscan_model,)


@app.cell
def _():
    # from bertopic.representation import MaximalMarginalRelevance

    # representation_model = MaximalMarginalRelevance(diversity=0.3)
    return


@app.cell
def _(custom_embedder, errors, hdbscan_model, umap_model):
    from bertopic import BERTopic
    import numpy as np

    topic_model = BERTopic(embedding_model=custom_embedder, calculate_probabilities=True, verbose=True, hdbscan_model=hdbscan_model, umap_model=umap_model)
    embeddings = custom_embedder.embed(errors)
    embeddings = np.array(embeddings)
    return embeddings, topic_model


@app.cell
def _(embeddings, errors, topic_model):
    topics, probs = topic_model.fit_transform(errors, embeddings)
    return


@app.cell
def _(topic_model):
    freq = topic_model.get_topic_info(); freq
    return


@app.cell
def _(topic_model):
    topic_model.get_topic(1)
    return


@app.cell
def _(topic_model):
    topic_model.visualize_topics()
    return


@app.cell
def _(topic_model):
    topic_model.visualize_hierarchy(top_n_topics=50)
    return


@app.cell
def _(topic_model):
    topic_model.visualize_heatmap()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
