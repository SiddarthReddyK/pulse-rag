from app.rag_pipeline import HashingEmbeddings, RAGPipeline
from tests.conftest import make_param


def _pipeline():
    return RAGPipeline(embeddings=HashingEmbeddings(), top_k=3)


def test_retrieves_relevant_chunk_for_ldl():
    context = _pipeline().retrieve("High LDL cholesterol")
    assert "LDL" in context


def test_abnormal_params_drive_retrieval():
    params = [
        make_param("LDL Cholesterol", 172, 0, 100, status="High"),
        make_param("Glucose", 90, 70, 100, status="Normal"),
    ]
    context = _pipeline().retrieve_for_abnormal_params(params)
    assert "LDL" in context


def test_all_normal_skips_retrieval():
    params = [make_param("Glucose", 90, 70, 100, status="Normal")]
    assert "normal range" in _pipeline().retrieve_for_abnormal_params(params)


def test_chunk_cap_is_respected():
    params = [make_param(f"Param {i}", 9, 0, 1, status="High") for i in range(10)]
    rag = _pipeline()
    capped = rag.retrieve_for_abnormal_params(params, max_chunks=2)
    uncapped = rag.retrieve_for_abnormal_params(params, max_chunks=50)
    assert 0 < len(capped) < len(uncapped)


def test_hashing_embeddings_are_deterministic_and_normalised():
    emb = HashingEmbeddings()
    a, b = emb.embed_query("high ldl"), emb.embed_query("high ldl")
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9
