from pathlib import Path
import json
import time
import streamlit as st
from rag import Index, Ollama, ingest, answer

st.set_page_config(page_title='EU Document AI Assistant', page_icon='📚', layout='wide')
st.title('EU Document AI Assistant')
st.caption('Explore documents, check the sources, generate answers with a local model.')
with st.sidebar:
    st.header('Documents and models')
    uploaded = st.file_uploader('Text-based PDF, TXT or Markdown', type=['pdf', 'txt', 'md'], accept_multiple_files=True)
    demo = st.checkbox('Use synthetic demo documents', value=not bool(uploaded))
    retrieval = st.selectbox('Retrieval', ['TF-IDF — local, no models', 'Embedding — Ollama'])
    embedding = st.text_input('Installed embedding model name', disabled=not retrieval.startswith('Embedding'))
    generate = st.checkbox('Generate an answer with Ollama')
    model = st.text_input('Installed chat model name', disabled=not generate)
    threshold = st.slider('Similarity threshold (not a probability)', 0., 1., .08, .01)
    k = st.slider('Passages to retrieve', 1, 8, 4)
    st.caption('Documents stay in session memory. Ollama is contacted on this computer only.')
    build = st.button('Index documents', type='primary')
    clear = st.button('Clear session')

if clear:
    for key in ['index', 'result', 'index_label']:
        st.session_state.pop(key, None)
    st.success('Index and answer removed from the session.')

if build:
    st.session_state.pop('index', None)
    st.session_state.pop('result', None)
    try:
        files = [(u.name, u.getvalue()) for u in uploaded]
        if demo:
            files += [(p.name, p.read_bytes()) for p in sorted((Path(__file__).parent / 'data/demo').glob('*.md'))]
        if len(files) > 20 or sum(len(b) for _, b in files) > 30 * 1024 * 1024:
            raise ValueError('Maximum 20 files and 30 MB total.')
        chunks = []
        with st.spinner('Reading and indexing…'):
            for name, data in files:
                chunks.extend(ingest(name, data))
            provider = Ollama(embedding_model=embedding) if retrieval.startswith('Embedding') else None
            st.session_state.index = Index(chunks, provider)
            st.session_state.index_label = f'{retrieval}: ' + ', '.join(name for name, _ in files)
        st.success(f'Indexed {len(st.session_state.index.chunks)} passages.')
    except Exception as exc:
        st.error(f'Indexing failed: {exc}')

if 'index' in st.session_state:
    st.caption('Active index: ' + st.session_state.index_label)
    st.caption('After changing documents or retrieval model, press Index documents again.')

if demo:
    st.info('The demo corpus is invented for technical testing. It contains no real EU legislation and is not legal advice.')
with st.form('question'):
    question = st.text_input('Question', placeholder='How long are Aurora project records retained?')
    submit = st.form_submit_button('Search and answer')
if submit:
    st.session_state.pop('result', None)
    if 'index' not in st.session_state:
        st.warning('Press “Index documents” first.')
    else:
        try:
            started = time.perf_counter()
            with st.spinner('Consulting the sources…'):
                result = answer(st.session_state.index, question, Ollama(chat_model=model) if generate else None, k, threshold)
            result['elapsed_seconds'] = round(time.perf_counter() - started, 3)
            st.session_state.result = result
        except Exception as exc:
            st.error(f'Answer not available: {exc}')
if 'result' in st.session_state:
    r = st.session_state.result
    st.subheader('Generated answer' if r['mode'] == 'generative' else 'Retrieved passages — no AI generation')
    st.text(r['answer'])
    st.caption(f"Time: {r['elapsed_seconds']} s · Citations can be checked; they are not an automatic guarantee of correctness.")
    for source in r['sources']:
        with st.expander(f"[{source['label']}] {source['source']} · page {source['page']} · similarity {source['score']:.3f}"):
            st.text(source['text'])
    st.download_button('Export answer and sources', json.dumps(r, ensure_ascii=False, indent=2), 'answer.json', 'application/json')
