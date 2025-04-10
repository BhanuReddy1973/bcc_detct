# API Documentation

## Overview
This document provides comprehensive documentation for the BCC detection API, including endpoints, request/response formats, and usage examples.

## API Endpoints

### 1. Health Check
```http
GET /health
```
Response:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-04-09T12:00:00Z"
}
```

### 2. Model Information
```http
GET /model/info
```
Response:
```json
{
    "model_name": "bcc_detection_v1",
    "model_version": "1.0.0",
    "input_shape": [512, 512, 3],
    "output_classes": ["BCC", "Non-malignant"],
    "confidence_threshold": 0.5
}
```

### 3. Prediction
```http
POST /predict
Content-Type: multipart/form-data
```
Request:
- `file`: Image file (TIF, PNG, JPG)
- `confidence_threshold`: Optional float (0.0-1.0)
- `return_patches`: Optional boolean

Response:
```json
{
    "prediction": "BCC",
    "confidence": 0.85,
    "processing_time": 1.23,
    "patches": [
        {
            "coordinates": [0, 0, 512, 512],
            "confidence": 0.92,
            "tissue_percentage": 0.75
        }
    ]
}
```

### 4. Batch Prediction
```http
POST /predict/batch
Content-Type: application/json
```
Request:
```json
{
    "files": [
        {
            "url": "https://example.com/image1.tif",
            "confidence_threshold": 0.6
        },
        {
            "url": "https://example.com/image2.tif",
            "confidence_threshold": 0.7
        }
    ]
}
```

## Error Handling

### Error Codes
| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 413 | Payload Too Large |
| 415 | Unsupported Media Type |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

### Error Response Format
```json
{
    "error": {
        "code": 400,
        "message": "Invalid image format",
        "details": "Supported formats: TIF, PNG, JPG"
    }
}
```

## Authentication

### API Key Authentication
```http
POST /predict
Authorization: Bearer <api_key>
```

### JWT Authentication
```http
POST /predict
Authorization: Bearer <jwt_token>
```

## Rate Limiting
- 100 requests per minute per API key
- 1000 requests per day per API key

## SDK Examples

### Python SDK
```python
from bcc_detection import BCCClient

client = BCCClient(api_key="your_api_key")

# Single prediction
result = client.predict("path/to/image.tif")

# Batch prediction
results = client.predict_batch([
    "path/to/image1.tif",
    "path/to/image2.tif"
])
```

### JavaScript SDK
```javascript
const bccClient = new BCCClient({
    apiKey: 'your_api_key'
});

// Single prediction
const result = await bccClient.predict('path/to/image.tif');

// Batch prediction
const results = await bccClient.predictBatch([
    'path/to/image1.tif',
    'path/to/image2.tif'
]);
```

## Best Practices
1. **Error Handling**
   - Implement retry logic
   - Handle timeouts
   - Validate responses
   - Log errors

2. **Performance**
   - Use batch endpoints
   - Compress images
   - Cache results
   - Monitor latency

3. **Security**
   - Rotate API keys
   - Use HTTPS
   - Validate inputs
   - Monitor usage

## Testing
```python
import pytest
from bcc_detection import BCCClient

def test_prediction():
    client = BCCClient(api_key="test_key")
    result = client.predict("test_image.tif")
    assert result["confidence"] > 0.5
```

## Next Steps
After API integration, proceed to:
- [Maintenance Guide](../../stage8_maintenance/README.md)
- [Performance Optimization Guide](../../stage9_optimization/README.md) 