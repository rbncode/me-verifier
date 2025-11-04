import os
import pytest
from flask import Flask
from api.app import app as flask_app
import io
from unittest.mock import patch

@pytest.fixture
def app():
    os.environ['MODEL_PATH'] = 'tests/models/'
    flask_app.config.update({
        "TESTING": True,
    })

    with flask_app.app_context():
        from api.app import load_models
        load_models()

    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_health_check(client):
    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json == {"status": "ok"}

def test_verify_no_image(client):
    response = client.post('/verify')
    assert response.status_code == 400
    assert 'error' in response.json

def test_verify_invalid_file_type(client):
    data = {
        'image': (io.BytesIO(b"some file content"), 'test.txt')
    }
    response = client.post('/verify', content_type='multipart/form-data', data=data)
    assert response.status_code == 400
    assert 'Invalid file type' in response.json['error']

@patch('api.app.preprocess_image')
def test_verify_valid_image(mock_preprocess_image, client):
    import numpy as np
    mock_preprocess_image.return_value = (np.random.rand(1, 512), None)

    data = {
        'image': (io.BytesIO(b"a valid image"), 'test.jpg')
    }
    response = client.post('/verify', content_type='multipart/form-data', data=data)

    assert response.status_code == 200
    json_data = response.get_json()
    assert 'model_version' in json_data
    assert 'is_me' in json_data
    assert 'score' in json_data
    assert 'threshold' in json_data
    assert 'timing_ms' in json_data

def test_verify_file_too_large(client):
    original_max_mb = os.getenv('MAX_MB', '10')
    os.environ['MAX_MB'] = '0.00001'

    large_content = b'a' * 20 
    data = {
        'image': (io.BytesIO(large_content), 'large_file.jpg')
    }

    response = client.post('/verify', content_type='multipart/form-data', data=data)

    assert response.status_code == 413
    assert 'File size exceeds the limit' in response.json['error']
    os.environ['MAX_MB'] = original_max_mb
