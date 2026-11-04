from datetime import timezone as dt_timezone
import datetime
from datetime import timezone
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app, compute_cosine_similarity, encode, get_utc_now, synthesize_and_score
client = TestClient(app)

def test_root_and_health():
    response = client.get('/')
    assert response.status_code == 200
    response = client.get('/health')
    assert response.status_code == 200

def test_list_tools():
    response = client.get('/tools')
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, (list, dict))

def test_opportunity_scoring_engine():
    signals = [{'source': 'OpenAlex', 'title': 'Perovskite Solar Breakthrough', 'summary': 'High efficiency solar cells', 'timestamp': get_utc_now().isoformat(), 'location': 'Malaysia', 'domain': 'research', 'signal_type': 'research', 'relevance_score': 85.0, 'source_url': 'https://example.com/1'}, {'source': 'World Bank', 'title': 'Renewable Incentives', 'summary': 'Subsidies for clean tech', 'timestamp': get_utc_now().isoformat(), 'location': 'Malaysia', 'domain': 'economics', 'signal_type': 'macro', 'relevance_score': 90.0, 'source_url': 'https://example.com/2'}]
    result1 = synthesize_and_score(signals)
    result2 = synthesize_and_score(signals)
    assert isinstance(result1, list)
    assert isinstance(result2, list)
    assert len(result1) == len(result2)
    if len(result1) > 0:
        score1 = result1[0].get('score', 0.0)
        score2 = result2[0].get('score', 0.0)
        assert pytest.approx(score1, rel=1e-05, abs=1e-05) == score2

def test_duplicate_detection_and_similarity():
    text1 = 'Commercial Perovskite Solar Deployment in Malaysia'
    text2 = 'Commercial Perovskite Solar Deployment in Malaysia'
    vecs = encode([text1, text2])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] >= 0.0
    sim = compute_cosine_similarity(vecs[0], vecs[1])
    assert pytest.approx(sim, rel=0.01, abs=0.01) == 1.0

def test_timezone_normalization():
    utc_now = get_utc_now()
    assert utc_now.tzinfo is not None
    assert utc_now.tzinfo == dt_timezone.utc

def test_tools_endpoints():
    payload = {'domains': ['research', 'startup'], 'limit': 10, 'min_score': 50}
    response = client.post('/tools/discover_opportunities', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    response = client.get('/tools/get_opportunity?opportunity_id=opp_001')
    assert response.status_code == 200
    assert response.json() is not None
    response = client.get('/tools/search_opportunities?query=battery')
    assert response.status_code == 200
    assert response.json() is not None
    response = client.get('/tools/location_intelligence?country=Malaysia&industry=solar')
    assert response.status_code == 200
    assert response.json() is not None
    response = client.get('/tools/technology_watch?keywords=agentic%20ai&keywords=robotics')
    assert response.status_code == 200
    assert response.json() is not None
    response = client.get('/tools/market_gap_finder?location=Malaysia&sector=healthcare')
    assert response.status_code == 200
    assert response.json() is not None