"""Unit tests for the NLP layer."""
import numpy as np

from src.nlp.preprocess import STOPWORDS, tokenize, words
from src.nlp.skills import extract_skills
from src.nlp.vectorizer import TfidfVectorizer, cosine_sim, l2_normalize


def test_words_lowercases_and_keeps_everything():
    assert words("Python and SQL") == ["python", "and", "sql"]


def test_tokenize_drops_stopwords_and_short_tokens():
    toks = tokenize("We will build a Scalable Data Platform in 2025")
    assert "we" not in toks and "will" not in toks      # stopwords gone
    assert "a" not in toks                               # 1-char gone
    assert "2025" not in toks                            # pure number gone
    assert "scalable" in toks and "platform" in toks
    assert STOPWORDS  # bundled list is non-empty


def test_extract_skills_matches_multiword_and_dedupes_substring():
    skills = extract_skills("Experience with apache spark and machine learning.")
    assert "apache spark" in skills
    assert "machine learning" in skills
    assert "spark" not in skills                         # consumed by "apache spark"


def test_extract_skills_returns_sorted_unique():
    skills = extract_skills("python python sql aws")
    assert skills == sorted(set(skills))


def test_tfidf_shape_and_vocabulary():
    docs = [["python", "sql"], ["python", "spark"], ["spark", "sql"]]
    v = TfidfVectorizer(min_df=1)
    X = v.fit_transform(docs)
    assert X.shape == (3, len(v.vocabulary_))
    assert set(v.vocabulary_) == {"python", "sql", "spark"}


def test_tfidf_rows_are_l2_normalized():
    docs = [["python", "sql", "python"], ["spark", "sql"]]
    X = TfidfVectorizer(min_df=1).fit_transform(docs)
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0)


def test_tfidf_rare_term_has_higher_idf_than_common_term():
    docs = [["common", "rare"], ["common"], ["common"]]
    v = TfidfVectorizer(min_df=1).fit(docs)
    assert v.idf_[v.vocabulary_["rare"]] > v.idf_[v.vocabulary_["common"]]


def test_cosine_identical_vectors_is_one():
    docs = [["python", "sql", "aws"]]
    X = TfidfVectorizer(min_df=1).fit_transform(docs)
    sim = cosine_sim(X[0], X[0])
    assert np.isclose(sim[0, 0], 1.0)


def test_l2_normalize_leaves_zero_rows_alone():
    out = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.isclose(np.linalg.norm(out[1]), 1.0)
